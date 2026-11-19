import inspect
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import src.predict as pred_mod  # noqa: E402

# ---------------------------------------------------------------------------
# KONFIGURASI HALAMAN & GAYA CSS MODERN
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="QC Injection Molding · AI Monitor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #F1F5F9 0%, #E2E8F0 100%);
    }

    /* Styling resmi untuk st.container(border=True) agar membungkus widget dengan rapi */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border-radius: 16px !important;
        border: 1px solid #CBD5E1 !important;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06) !important;
        padding: 1.15rem 1.35rem !important;
        margin-bottom: 0.9rem !important;
    }

    /* Menyamakan tinggi label seluruh input widget agar kotak input sejajar horizontal */
    [data-testid="stWidgetLabel"] {
        min-height: 2.5rem !important;
        display: flex !important;
        align-items: flex-end !important;
        margin-bottom: 0.25rem !important;
    }
    [data-testid="stWidgetLabel"] p {
        font-weight: 700 !important;
        color: #0F172A !important;
        font-size: 0.86rem !important;
        line-height: 1.25 !important;
    }

    /* Hero Banner */
    .hero-card {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 55%, #0284C7 100%);
        padding: 1.5rem 1.9rem;
        border-radius: 18px;
        color: #FFFFFF;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.18);
        margin-bottom: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.22);
        border: 1px solid rgba(125, 211, 252, 0.45);
        color: #E0F2FE;
        font-size: 0.76rem;
        font-weight: 700;
        padding: 0.22rem 0.75rem;
        border-radius: 999px;
        letter-spacing: 0.04em;
        margin-bottom: 0.55rem;
    }
    .hero-title {
        font-size: 1.7rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
        line-height: 1.2;
    }
    .hero-sub {
        font-size: 0.94rem;
        color: #CBD5E1;
        margin: 0;
    }
    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.95rem;
    }
    .stat-pill {
        background: rgba(15, 23, 42, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 0.32rem 0.8rem;
        border-radius: 10px;
        font-size: 0.81rem;
        color: #F8FAFC;
    }

    /* Header Section di dalam Container */
    .section-header {
        font-size: 1.06rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.15rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }
    .section-sub {
        font-size: 0.83rem;
        color: #64748B;
        margin-bottom: 0.65rem;
    }

    /* Kartu Keterangan Seragam di bawah tiap input parameter */
    .param-desc {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.48rem 0.62rem;
        font-size: 0.75rem;
        color: #334155;
        line-height: 1.38;
        margin-top: 0.35rem;
        margin-bottom: 0.65rem;
        min-height: 4.1rem;
    }
    .param-desc b {
        color: #0F172A;
    }

    /* Material Info Banner */
    .mat-banner {
        background: linear-gradient(90deg, #EFF6FF 0%, #F0F9FF 100%);
        border-left: 4px solid #0284C7;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        font-size: 0.85rem;
        color: #1E3A8A;
        margin-top: 0.25rem;
    }

    /* Result Cards */
    .result-pass {
        background: linear-gradient(135deg, #065F46 0%, #059669 60%, #10B981 100%);
        color: #FFFFFF;
        padding: 1.4rem 1.6rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(5, 150, 105, 0.25);
        margin-bottom: 1rem;
    }
    .result-reject {
        background: linear-gradient(135deg, #991B1B 0%, #DC2626 55%, #F97316 100%);
        color: #FFFFFF;
        padding: 1.4rem 1.6rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(220, 38, 38, 0.28);
        margin-bottom: 1rem;
    }
    .result-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.9;
    }
    .result-status {
        font-size: 1.75rem;
        font-weight: 800;
        margin: 0.2rem 0 0.35rem 0;
    }

    /* KPI Mini Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
        margin: 0.85rem 0 1rem 0;
    }
    .kpi-card {
        background: #F8FAFC;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        border: 1px solid #E2E8F0;
    }
    .kpi-title {
        font-size: 0.76rem;
        color: #64748B;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.3rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 0.15rem;
    }

    /* Warning & Action Cards */
    .warn-card {
        background: #FFFBEB;
        border-left: 5px solid #F59E0B;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        color: #92400E;
        font-size: 0.88rem;
        margin-bottom: 0.65rem;
    }
    .action-item {
        background: #F8FAFC;
        border-left: 4px solid #2563EB;
        border-radius: 10px;
        padding: 0.72rem 0.95rem;
        margin-bottom: 0.5rem;
        color: #1E293B;
        font-size: 0.9rem;
    }

    /* Primary Button */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #0284C7 100%);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1.2rem;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.28);
        transition: all 0.2s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.38);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# PEMUATAN ARTIFACTS & ADAPTER INFERENSI
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(ROOT_DIR / "artifacts" / "model.joblib")
    with open(ROOT_DIR / "artifacts" / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


model, meta = load_artifacts()
final_name = meta.get("model_final", "HistGradientBoostingClassifier")
threshold_bisnis = float(
    meta.get("threshold_bisnis")
    or meta.get("threshold")
    or meta.get("selected_threshold")
    or meta["validation_results"][final_name]["threshold"]
)
test_pr_auc = meta.get("test_results_opened_once", {}).get(final_name, {}).get("pr_auc", 0.6727)

MATERIAL_GUIDE = {
    "PP": {
        "nama": "Polypropylene (PP) · Non-Higroskopis",
        "suhu_leleh": "230 °C",
        "suhu_cetakan": "40 °C",
        "kelembaban": "< 0,05% (tidak sensitif uap air)",
    },
    "ABS": {
        "nama": "Acrylonitrile Butadiene Styrene (ABS) · Higroskopis",
        "suhu_leleh": "240 °C",
        "suhu_cetakan": "60 °C",
        "kelembaban": "< 0,08% (wajib pre-drying, risiko silver streak)",
    },
    "PC": {
        "nama": "Polycarbonate (PC) · Sangat Higroskopis",
        "suhu_leleh": "300 °C",
        "suhu_cetakan": "90 °C",
        "kelembaban": "< 0,04% (wajib dehumidifying dryer)",
    },
}


def run_inference(payload: dict) -> dict:
    """Memanggil fungsi prediksi dari src.predict sesuai tanda tangan parameternya."""
    model_path = ROOT_DIR / "artifacts" / "model.joblib"
    meta_path = ROOT_DIR / "artifacts" / "metadata.json"

    for fn_name in ("predict_cycle", "predict_single", "predict_one", "predict"):
        fn = getattr(pred_mod, fn_name, None)
        if callable(fn):
            sig = inspect.signature(fn)
            kwargs = {}
            for pname, param in list(sig.parameters.items())[1:]:
                if pname == "model_path":
                    kwargs[pname] = model_path
                elif pname == "metadata_path":
                    kwargs[pname] = meta_path
                elif pname == "model":
                    kwargs[pname] = model
                elif param.default is inspect.Parameter.empty:
                    break
            res = fn(payload, **kwargs)
            if isinstance(res, dict):
                return res

    prob = float(model.predict_proba(pd.DataFrame([payload]))[0, 1])
    is_defect = prob >= threshold_bisnis
    return {
        "probabilitas_cacat": prob,
        "threshold": threshold_bisnis,
        "prediksi_cacat": int(is_defect),
        "keputusan": "SISIHKAN (POTENSI CACAT)" if is_defect else "LOLOS QC",
        "warnings": [],
        "rekomendasi": [],
    }


def build_recommendations(payload: dict, is_defect: bool) -> list[str]:
    """Menyusun rekomendasi tindakan konkret bagi operator lantai produksi."""
    if not is_defect:
        return [
            "Pertahankan parameter operasi saat ini dan lanjutkan siklus produksi normal.",
            "Lakukan pengecekan visual berkala setiap 50 siklus cetak sesuai SOP QC.",
        ]

    tips = []
    bahan = payload["bahan"]
    suhu_leleh = payload["suhu_leleh_C"]
    if suhu_leleh > 350:
        suhu_leleh = (suhu_leleh - 32.0) * 5.0 / 9.0

    std_leleh = {"PP": 230.0, "ABS": 240.0, "PC": 300.0}[bahan]
    std_cetakan = {"PP": 40.0, "ABS": 60.0, "PC": 90.0}[bahan]

    if bahan in ("ABS", "PC") and payload["kelembaban_bahan_pct"] > 0.08:
        tips.append(
            f"Keringkan resin {bahan} di dehumidifying dryer hingga kelembaban < 0,08% "
            f"(terbaca {payload['kelembaban_bahan_pct']:.3f}%) untuk mencegah cacat silver streak."
        )
    if abs(suhu_leleh - std_leleh) > 12:
        arah = "Turunkan" if suhu_leleh > std_leleh else "Naikkan"
        tips.append(
            f"{arah} suhu barrel/leleh mendekati titik optimal {bahan} (~{std_leleh:.0f} °C; terbaca {suhu_leleh:.1f} °C)."
        )
    if abs(payload["suhu_cetakan_C"] - std_cetakan) > 10:
        arah = "Turunkan" if payload["suhu_cetakan_C"] > std_cetakan else "Naikkan"
        tips.append(
            f"{arah} suhu cetakan (mold) mendekati standar {bahan} (~{std_cetakan:.0f} °C; terbaca {payload['suhu_cetakan_C']:.1f} °C)."
        )
    if 0 < payload["tekanan_injeksi_bar"] < 850:
        tips.append(
            f"Naikkan tekanan injeksi ke rentang 880–980 bar (terbaca {payload['tekanan_injeksi_bar']:.0f} bar) untuk mencegah short shot."
        )
    if payload["waktu_pendinginan_s"] < 15:
        tips.append(
            f"Tambah durasi pendinginan menjadi 16–20 detik (terbaca {payload['waktu_pendinginan_s']:.1f} dtk) agar komponen tidak melengkung."
        )
    if payload["mesin"] == "IM-03":
        tips.append("Periksa kalibrasi heater band dan check-ring pada mesin IM-03 (riwayat tingkat cacat tertinggi).")
    if not tips:
        tips.append("Sisihkan komponen untuk inspeksi dimensi/visual oleh staf QC sebelum dikemas.")
    return tips


# ---------------------------------------------------------------------------
# INISIALISASI STATE & PRESET SKENARIO
# ---------------------------------------------------------------------------
DEFAULTS = {
    "mesin": "IM-01",
    "shift": "pagi",
    "bahan": "PP",
    "suhu_leleh_C": 230.0,
    "suhu_cetakan_C": 40.0,
    "tekanan_injeksi_bar": 930.0,
    "kecepatan_injeksi_mm_s": 50.0,
    "waktu_pendinginan_s": 19.0,
    "waktu_siklus_s": 33.0,
    "kelembaban_bahan_pct": 0.015,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def apply_preset(values: dict):
    for k, v in values.items():
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# SIDEBAR INFORMASI SISTEM & BIAYA BISNIS
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏭 Panel Kontrol QC")
    st.caption("Pabrik Komponen Plastik Cikarang · Kasus 01")
    st.divider()

    st.markdown("**📊 Parameter Model Aktif**")
    st.info(
        f"**Model:** `{final_name}`\n\n"
        f"**PR-AUC Test:** `{test_pr_auc:.4f}`\n\n"
        f"**Threshold Bisnis:** `{threshold_bisnis:.3f}`"
    )

    st.markdown("**💰 Matriks Biaya Keputusan (20 : 1)**")
    st.warning(
        "• **Cacat Lolos (FN):** `Rp500.000` / unit\n"
        "• **Bagus Disisihkan (FP):** `Rp25.000` / unit\n\n"
        "Ambang batas rendah (`0,065`) memprioritaskan pencegahan produk cacat lolos ke pelanggan."
    )

    st.markdown("**🧪 Standar Operasi Material**")
    st.markdown(
        """
        | Bahan | Leleh | Cetakan | Kelembaban |
        | :--- | :---: | :---: | :---: |
        | **PP** | 230°C | 40°C | < 0,05% |
        | **ABS** | 240°C | 60°C | < 0,08% |
        | **PC** | 300°C | 90°C | < 0,04% |
        """
    )


# ---------------------------------------------------------------------------
# HEADER UTAMA (HERO BANNER)
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-badge">REAL-TIME QUALITY CONTROL · MKB UPI</div>
        <div class="hero-title">Sistem Deteksi Dini Cacat Injection Molding</div>
        <p class="hero-sub">
            Evaluasi risiko cacat siklus cetak secara instan dan rekomendasi penyetelan parameter mesin bagi operator lantai produksi.
        </p>
        <div class="pill-row">
            <div class="stat-pill">🤖 Model: <b>{final_name}</b></div>
            <div class="stat-pill">🎯 Threshold Bisnis: <b>{threshold_bisnis:.3f}</b></div>
            <div class="stat-pill">📈 PR-AUC Test: <b>{test_pr_auc:.4f}</b></div>
            <div class="stat-pill">🛡️ Auto-Guard Sensor: <b>Aktif</b></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# TOMBOL PRESET CEPAT (UNTUK DEMO & PENGUJIAN)
# ---------------------------------------------------------------------------
st.markdown("#### ⚡ Simulasi Cepat Skenario Lantai Produksi")
p_col1, p_col2, p_col3 = st.columns(3)

with p_col1:
    if st.button("🟢 Skenario 1 · Siklus Normal (PP)", use_container_width=True):
        apply_preset(
            {
                "mesin": "IM-01",
                "shift": "pagi",
                "bahan": "PP",
                "suhu_leleh_C": 230.0,
                "suhu_cetakan_C": 40.0,
                "tekanan_injeksi_bar": 930.0,
                "kecepatan_injeksi_mm_s": 50.0,
                "waktu_pendinginan_s": 19.0,
                "waktu_siklus_s": 33.0,
                "kelembaban_bahan_pct": 0.015,
            }
        )
        st.rerun()

with p_col2:
    if st.button("🔴 Skenario 2 · Risiko Tinggi (ABS Lembab)", use_container_width=True):
        apply_preset(
            {
                "mesin": "IM-03",
                "shift": "malam",
                "bahan": "ABS",
                "suhu_leleh_C": 258.0,
                "suhu_cetakan_C": 45.0,
                "tekanan_injeksi_bar": 760.0,
                "kecepatan_injeksi_mm_s": 38.0,
                "waktu_pendinginan_s": 12.0,
                "waktu_siklus_s": 26.0,
                "kelembaban_bahan_pct": 0.210,
            }
        )
        st.rerun()

with p_col3:
    if st.button("🟡 Skenario 3 · Anomali Sensor (°F & Dropout)", use_container_width=True):
        apply_preset(
            {
                "mesin": "IM-02",
                "shift": "siang",
                "bahan": "PC",
                "suhu_leleh_C": 575.0,
                "suhu_cetakan_C": 88.0,
                "tekanan_injeksi_bar": 0.0,
                "kecepatan_injeksi_mm_s": 48.0,
                "waktu_pendinginan_s": 20.0,
                "waktu_siklus_s": -1.0,
                "kelembaban_bahan_pct": 0.040,
            }
        )
        st.rerun()

st.write("")

# ---------------------------------------------------------------------------
# FORMULIR INPUT PARAMETER & HASIL PREDIKSI (TERBUNGKUS RAPI DALAM CONTAINER)
# ---------------------------------------------------------------------------
left_col, right_col = st.columns([1.18, 1.0], gap="large")

with left_col:
    # --- SECTION BOX 1 ---
    with st.container(border=True):
        st.markdown('<div class="section-header">🎛️ 1. Identitas Mesin & Jenis Resin</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Pilih unit mesin cetak, jadwal shift kerja, dan jenis polimer plastik yang diproses.</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3, vertical_alignment="top")
        with c1:
            mesin = st.selectbox(
                "Mesin Cetak (ID Unit)",
                ["IM-01", "IM-02", "IM-03", "IM-04"],
                key="mesin",
                help="Kode mesin injection molding di pabrik Cikarang. Mesin IM-03 memiliki sensitivitas termal lebih tinggi sehingga diawasi lebih ketat.",
            )
            st.markdown(
                '<div class="param-desc"><b>Kode Unit Mesin:</b> Lini cetak aktif (<code>IM-03</code> memiliki riwayat laju cacat lebih tinggi).</div>',
                unsafe_allow_html=True,
            )
        with c2:
            shift = st.selectbox(
                "Shift Kerja Operator",
                ["pagi", "siang", "malam"],
                key="shift",
                help="Jadwal kerja operator. Shift malam cenderung mengalami fluktuasi suhu sekitar dan laju cacat lebih tinggi.",
            )
            st.markdown(
                '<div class="param-desc"><b>Jadwal Shift:</b> Waktu operasi produksi (<code>malam</code> lebih rentan fluktuasi proses).</div>',
                unsafe_allow_html=True,
            )
        with c3:
            bahan = st.selectbox(
                "Jenis Bahan Resin",
                ["PP", "ABS", "PC"],
                key="bahan",
                help="Jenis resin polimer: PP (non-higroskopis), ABS (higroskopis), dan PC (sangat higroskopis).",
            )
            st.markdown(
                '<div class="param-desc"><b>Polimer Plastik:</b> Jenis bijih resin (<code>ABS</code> & <code>PC</code> menyerap kelembaban udara).</div>',
                unsafe_allow_html=True,
            )

        guide = MATERIAL_GUIDE[bahan]
        st.markdown(
            f"""
            <div class="mat-banner">
                <b>{guide['nama']}</b><br/>
                Acuan Optimal &rarr; Suhu Leleh: <b>{guide['suhu_leleh']}</b> · Cetakan: <b>{guide['suhu_cetakan']}</b> · Kelembaban: <b>{guide['kelembaban']}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- SECTION BOX 2 ---
    with st.container(border=True):
        st.markdown('<div class="section-header">🌡️ 2. Parameter Termal & Kelembaban Resin</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Kondisi suhu pelelehan plastik, suhu rongga cetakan, serta kadar uap air bijih plastik.</div>',
            unsafe_allow_html=True,
        )
        t1, t2, t3 = st.columns(3, vertical_alignment="top")
        with t1:
            suhu_leleh_C = st.number_input(
                "Suhu Leleh (°C)",
                step=1.0,
                format="%.1f",
                key="suhu_leleh_C",
                help="Suhu lelehan plastik di barrel sebelum injeksi. Ideal: PP 230°C, ABS 240°C, PC 300°C. Input > 350 otomatis dikonversi dari °F ke °C.",
            )
            st.markdown(
                '<div class="param-desc"><b>Suhu Leleh (Barrel):</b> Temperatur lelehan resin (Ideal: <b>230–300 °C</b>; &gt;350 auto-konversi °F).</div>',
                unsafe_allow_html=True,
            )
        with t2:
            suhu_cetakan_C = st.number_input(
                "Suhu Cetakan (°C)",
                step=1.0,
                format="%.1f",
                key="suhu_cetakan_C",
                help="Suhu dinding rongga cetakan (mold). Ideal: PP 40°C, ABS 60°C, PC 90°C. Terlalu dingin memicu garis alir/short shot.",
            )
            st.markdown(
                '<div class="param-desc"><b>Suhu Cetakan (Mold):</b> Suhu rongga cetak (Ideal: PP 40°C, ABS 60°C, PC 90°C).</div>',
                unsafe_allow_html=True,
            )
        with t3:
            kelembaban_bahan_pct = st.number_input(
                "Kelembaban (%)",
                min_value=0.0,
                step=0.005,
                format="%.3f",
                key="kelembaban_bahan_pct",
                help="Persentase kadar air pada bijih resin sebelum masuk hopper. Jika > 0,08% pada ABS/PC, uap air memicu cacat permukaan silver streak.",
            )
            st.markdown(
                '<div class="param-desc"><b>Kelembaban Bahan:</b> Kadar air resin di hopper (Batas aman ABS/PC: <code>&lt; 0,080%</code>).</div>',
                unsafe_allow_html=True,
            )

    # --- SECTION BOX 3 ---
    with st.container(border=True):
        st.markdown('<div class="section-header">⚙️ 3. Parameter Injeksi & Waktu Siklus</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Tekanan dan kecepatan pengisian cetakan serta durasi pendinginan komponen.</div>',
            unsafe_allow_html=True,
        )
        m1, m2 = st.columns(2, vertical_alignment="top")
        with m1:
            tekanan_injeksi_bar = st.number_input(
                "Tekanan Injeksi (bar)",
                step=5.0,
                format="%.1f",
                key="tekanan_injeksi_bar",
                help="Tekanan sekrup saat mendorong lelehan mengisi cetakan. Rentang normal: 850–1.050 bar. Nilai < 850 bar berisiko cetakan tidak terisi penuh (short shot).",
            )
            st.markdown(
                '<div class="param-desc"><b>Tekanan Injeksi:</b> Gaya dorong lelehan mengisi mold (Normal: <b>850–1.050 bar</b>; &lt;850 risiko <i>short shot</i>).</div>',
                unsafe_allow_html=True,
            )

            waktu_pendinginan_s = st.number_input(
                "Waktu Pendinginan (detik)",
                step=0.5,
                format="%.1f",
                key="waktu_pendinginan_s",
                help="Durasi komponen didinginkan di dalam cetakan sebelum dikeluarkan. Normal: 15–22 detik. Terlalu singkat memicu lengkungan (warpage).",
            )
            st.markdown(
                '<div class="param-desc"><b>Waktu Pendinginan:</b> Durasi pembekuan produk di cetakan (Normal: <b>15–22 dtk</b>; &lt;15 risiko melengkung).</div>',
                unsafe_allow_html=True,
            )

        with m2:
            kecepatan_injeksi_mm_s = st.number_input(
                "Kecepatan Injeksi (mm/s)",
                step=1.0,
                format="%.1f",
                key="kecepatan_injeksi_mm_s",
                help="Laju maju sekrup saat menyuntikkan lelehan plastik. Rentang normal: 40–60 mm/s. Terlalu cepat memicu burn mark; terlalu lambat memicu pembekuan dini.",
            )
            st.markdown(
                '<div class="param-desc"><b>Kecepatan Injeksi:</b> Laju gerak maju sekrup injeksi (Normal: <b>40–60 mm/s</b>; menjaga pengisian merata).</div>',
                unsafe_allow_html=True,
            )

            waktu_siklus_s = st.number_input(
                "Waktu Siklus Total (detik)",
                step=0.5,
                format="%.1f",
                key="waktu_siklus_s",
                help="Total durasi satu siklus lengkap dari tutup cetakan, injeksi, pendinginan, hingga buka cetakan. Normal: 28–38 detik (harus lebih lama dari waktu pendinginan).",
            )
            st.markdown(
                '<div class="param-desc"><b>Waktu Siklus Total:</b> Durasi keseluruhan 1 siklus cetak lengkap (Normal: <b>28–38 dtk</b>).</div>',
                unsafe_allow_html=True,
            )

    submit_btn = st.button("🔍 Evaluasi Kualitas Siklus Sekarang", type="primary", use_container_width=True)

# Jalankan inferensi otomatis saat halaman dimuat maupun saat tombol ditekan
payload = {
    "mesin": mesin,
    "shift": shift,
    "bahan": bahan,
    "suhu_leleh_C": float(suhu_leleh_C),
    "suhu_cetakan_C": float(suhu_cetakan_C),
    "tekanan_injeksi_bar": float(tekanan_injeksi_bar),
    "kecepatan_injeksi_mm_s": float(kecepatan_injeksi_mm_s),
    "waktu_pendinginan_s": float(waktu_pendinginan_s),
    "waktu_siklus_s": float(waktu_siklus_s),
    "kelembaban_bahan_pct": float(kelembaban_bahan_pct),
}

result = run_inference(payload)
prob = float(result.get("probabilitas_cacat", result.get("probability", 0.0)))
tau = float(result.get("threshold", threshold_bisnis))
is_defect = bool(result.get("prediksi_cacat", prob >= tau))
warnings_list = result.get("warnings", [])
rekomendasi = result.get("rekomendasi") or build_recommendations(payload, is_defect)

with right_col:
    # --- SECTION BOX HASIL ---
    with st.container(border=True):
        st.markdown('<div class="section-header">📋 Hasil Keputusan & Panduan Operator</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Rekomendasi otomatis berbasis risiko biaya bisnis (FN Rp500 rb vs FP Rp25 rb).</div>',
            unsafe_allow_html=True,
        )

        # 1. Peringatan Validasi Input / Sensor (jika ada)
        if warnings_list:
            for w in warnings_list:
                st.markdown(f'<div class="warn-card">⚠️ <b>Validasi Sensor Otomatis:</b> {w}</div>', unsafe_allow_html=True)

        # 2. Kartu Keputusan Utama
        if is_defect:
            st.markdown(
                f"""
                <div class="result-reject">
                    <div class="result-label">Keputusan Quality Control</div>
                    <div class="result-status">🚨 SISIHKAN · POTENSI CACAT</div>
                    <div>Probabilitas cacat <b>{prob*100:.2f}%</b> melewati batas toleransi bisnis (<b>{tau*100:.1f}%</b>). Segera pisahkan produk untuk inspeksi manual!</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-pass">
                    <div class="result-label">Keputusan Quality Control</div>
                    <div class="result-status">✅ LOLOS QC · PRODUKSI NORMAL</div>
                    <div>Probabilitas cacat <b>{prob*100:.2f}%</b> berada di bawah ambang batas bisnis (<b>{tau*100:.1f}%</b>). Parameter proses dalam kondisi stabil.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 3. Kartu KPI Ringkas
        rasio_risiko = prob / tau if tau > 0 else 0.0
        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-title">Probabilitas Cacat</div>
                    <div class="kpi-value" style="color:{'#DC2626' if is_defect else '#059669'};">{prob*100:.2f}%</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Threshold Bisnis</div>
                    <div class="kpi-value">{tau*100:.1f}%</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Indeks Risiko</div>
                    <div class="kpi-value">{rasio_risiko:.2f}×</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4. Bar Visual Tingkat Risiko
        st.caption(f"Visualisasi Skor Risiko Siklus ({prob*100:.2f}% vs Ambang Batas {tau*100:.1f}%)")
        st.progress(min(max(prob, 0.0), 1.0))

        # 5. Rekomendasi Tindakan Operator
        st.markdown("**🛠️ Rekomendasi Tindakan Operator:**")
        for idx, tip in enumerate(rekomendasi, start=1):
            st.markdown(f'<div class="action-item"><b>{idx}.</b> {tip}</div>', unsafe_allow_html=True)