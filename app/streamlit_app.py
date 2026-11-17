import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.predict import predict_cycle

st.set_page_config(page_title="QC Injection Molding · Cikarang", layout="wide")
st.title("Deteksi Dini Cacat Injection Molding")
st.caption("Sistem pendukung keputusan real-time untuk operator lantai produksi (Kasus 01)")

col_preset1, col_preset2, col_preset3 = st.columns(3)
if "preset" not in st.session_state:
    st.session_state.preset = "normal"

if col_preset1.button("Isi Contoh: Siklus Normal (PP)"):
    st.session_state.preset = "normal"
if col_preset2.button("Isi Contoh: Risiko Tinggi (ABS Lembab)"):
    st.session_state.preset = "risk"
if col_preset3.button("Isi Contoh: Input Tidak Wajar (Uji Validasi)"):
    st.session_state.preset = "invalid"

defaults = {
    "normal": ("IM-01", "pagi", "PP", 231.0, 40.0, 930.0, 80.0, 19.0, 33.0, 0.015),
    "risk": ("IM-03", "malam", "ABS", 256.0, 48.0, 760.0, 108.0, 12.0, 26.0, 0.210),
    "invalid": ("IM-02", "siang", "PC", 575.0, 90.0, 0.0, 80.0, 18.0, -1.0, 0.090),
}[st.session_state.preset]

with st.form("form_siklus"):
    c1, c2, c3 = st.columns(3)
    mesin = c1.selectbox("Mesin", ["IM-01", "IM-02", "IM-03", "IM-04"], index=["IM-01", "IM-02", "IM-03", "IM-04"].index(defaults[0]))
    shift = c2.selectbox("Shift", ["pagi", "siang", "malam"], index=["pagi", "siang", "malam"].index(defaults[1]))
    bahan = c3.selectbox("Jenis Bahan", ["PP", "ABS", "PC"], index=["PP", "ABS", "PC"].index(defaults[2]))

    c4, c5, c6 = st.columns(3)
    suhu_leleh = c4.number_input("Suhu Leleh Barrel (°C)", value=float(defaults[3]), step=1.0)
    suhu_cetakan = c5.number_input("Suhu Cetakan (°C)", value=float(defaults[4]), step=1.0)
    tekanan = c6.number_input("Tekanan Injeksi (bar)", value=float(defaults[5]), step=5.0)

    c7, c8, c9, c10 = st.columns(4)
    kecepatan = c7.number_input("Kecepatan Injeksi (mm/s)", value=float(defaults[6]), step=1.0)
    pendinginan = c8.number_input("Waktu Pendinginan (detik)", value=float(defaults[7]), step=0.5)
    siklus = c9.number_input("Waktu Siklus Total (detik)", value=float(defaults[8]), step=0.5)
    kelembaban = c10.number_input("Kelembaban Bahan (%)", value=float(defaults[9]), step=0.005, format="%.3f")

    submitted = st.form_submit_button("Evaluasi Siklus Cetak", type="primary")

if submitted:
    payload = {
        "mesin": mesin,
        "shift": shift,
        "bahan": bahan,
        "suhu_leleh_C": suhu_leleh,
        "suhu_cetakan_C": suhu_cetakan,
        "tekanan_injeksi_bar": tekanan,
        "kecepatan_injeksi_mm_s": kecepatan,
        "waktu_pendinginan_s": pendinginan,
        "waktu_siklus_s": siklus,
        "kelembaban_bahan_pct": kelembaban,
    }
    res = predict_cycle(payload)

    if res["peringatan_input"]:
        for w in res["peringatan_input"]:
            st.warning(f"Validasi Sensor/Input: {w}")

    m1, m2 = st.columns(2)
    m1.metric("Probabilitas Cacat", f"{res['probabilitas_cacat']*100:.2f}%", f"Batas Bisnis: {res['threshold_bisnis']*100:.2f}%")
    if res["prediksi_cacat"] == 1:
        m2.error(f"Keputusan: {res['status_qc']}")
    else:
        m2.success(f"Keputusan: {res['status_qc']}")

    st.subheader("Saran Tindakan Operator")
    for s in res["saran_tindakan"]:
        st.write(f"- {s}")