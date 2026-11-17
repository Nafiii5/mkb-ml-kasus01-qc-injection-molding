import json
import joblib
import numpy as np
import pandas as pd
from src.cleaning import STANDAR_BAHAN, VALID_BAHAN, VALID_MESIN, VALID_SHIFT

RENTANG_WAJAR = {
    "suhu_leleh_C": (190.0, 340.0),
    "suhu_cetakan_C": (15.0, 120.0),
    "tekanan_injeksi_bar": (400.0, 1400.0),
    "kecepatan_injeksi_mm_s": (20.0, 150.0),
    "waktu_pendinginan_s": (5.0, 40.0),
    "waktu_siklus_s": (12.0, 60.0),
    "kelembaban_bahan_pct": (0.0, 0.60),
}


def validate_and_warn(payload: dict) -> list[str]:
    """Memeriksa masukan kosong, mustahil secara fisika, atau di luar rentang operasi."""
    warnings = []

    bahan = str(payload.get("bahan", "")).replace(".", "").strip().upper()
    mesin = str(payload.get("mesin", "")).strip().upper()
    shift = str(payload.get("shift", "")).strip().lower()

    if bahan not in VALID_BAHAN:
        raise ValueError(f"Bahan '{payload.get('bahan')}' tidak dikenali. Pilih: PP, ABS, atau PC.")
    if mesin not in VALID_MESIN:
        raise ValueError(f"Mesin '{payload.get('mesin')}' tidak dikenali. Pilih: IM-01 s.d. IM-04.")
    if shift not in VALID_SHIFT:
        raise ValueError(f"Shift '{payload.get('shift')}' tidak dikenali. Pilih: pagi, siang, atau malam.")

    for col, (low, high) in RENTANG_WAJAR.items():
        val = payload.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            warnings.append(f"Parameter '{col}' kosong -> diimputasi otomatis dengan median bahan {bahan}.")
        else:
            val_f = float(val)
            if col == "suhu_leleh_C" and val_f > 350.0:
                warnings.append(
                    f"Suhu leleh {val_f} terdeteksi dalam °F -> dikonversi otomatis ke {(val_f - 32)*5/9:.1f} °C."
                )
            elif val_f <= 0 and col in {"tekanan_injeksi_bar", "waktu_siklus_s", "waktu_pendinginan_s"}:
                warnings.append(f"Nilai mustahil pada '{col}' ({val_f}) -> dianggap sensor dropout & diimputasi.")
            elif not (low <= val_f <= high):
                warnings.append(
                    f"Parameter '{col}' ({val_f}) di luar rentang operasi normal [{low} - {high}]."
                )

    t_cool = payload.get("waktu_pendinginan_s")
    t_cyc = payload.get("waktu_siklus_s")
    if t_cool is not None and t_cyc is not None and float(t_cyc) > 0 and float(t_cool) >= float(t_cyc):
        warnings.append("Waktu pendinginan >= waktu siklus total (tidak wajar secara fisika).")

    return warnings


def generate_operator_actions(payload: dict, prob: float, threshold: float) -> list[str]:
    """Menghasilkan rekomendasi perbaikan pengaturan mesin berdasarkan penyimpangan parameter."""
    actions = []
    bahan = str(payload["bahan"]).replace(".", "").strip().upper()
    std_melt = STANDAR_BAHAN[bahan]["suhu_leleh"]
    std_mold = STANDAR_BAHAN[bahan]["suhu_cetakan"]
    is_higro = STANDAR_BAHAN[bahan]["higroskopis"] == 1

    suhu_leleh = payload.get("suhu_leleh_C")
    if suhu_leleh is not None:
        suhu_leleh = float(suhu_leleh)
        if suhu_leleh > 350.0:
            suhu_leleh = (suhu_leleh - 32.0) * 5.0 / 9.0
        if abs(suhu_leleh - std_melt) > 10.0:
            arah = "Turunkan" if suhu_leleh > std_melt else "Naikkan"
            actions.append(f"{arah} suhu barrel menuju standar {bahan} (~{std_melt:.0f} °C).")

    suhu_cetakan = payload.get("suhu_cetakan_C")
    if suhu_cetakan is not None and abs(float(suhu_cetakan) - std_mold) > 8.0:
        arah = "Turunkan" if float(suhu_cetakan) > std_mold else "Naikkan"
        actions.append(f"{arah} suhu cetakan menuju standar {bahan} (~{std_mold:.0f} °C).")

    kelembaban = payload.get("kelembaban_bahan_pct")
    if is_higro and kelembaban is not None and float(kelembaban) > 0.08:
        actions.append(
            f"Keringkan resin {bahan} di dehumidifying dryer (kelembaban {float(kelembaban):.3f}% > 0,08%, risiko silver streak)."
        )

    tekanan = payload.get("tekanan_injeksi_bar")
    if tekanan is not None and 0 < float(tekanan) < 830.0:
        actions.append("Naikkan tekanan injeksi ke kisaran 880-980 bar untuk mencegah short shot / sink mark.")

    pendinginan = payload.get("waktu_pendinginan_s")
    if pendinginan is not None and 0 < float(pendinginan) < 15.0:
        actions.append("Tambah waktu pendinginan minimal 16-20 detik agar komponen tidak melengkung (warpage).")

    if str(payload.get("mesin", "")).upper() == "IM-03":
        actions.append("Periksa kalibrasi nozzle & check-ring mesin IM-03 (riwayat tingkat cacat lebih tinggi).")

    if not actions:
        if prob >= threshold:
            actions.append("Sisihkan komponen untuk inspeksi QC manual dan periksa kestabilan parameter siklus.")
        else:
            actions.append("Parameter proses berada dalam batas normal. Lanjutkan produksi.")

    return actions


def predict_cycle(payload: dict, model_path="artifacts/model.joblib", meta_path="artifacts/metadata.json") -> dict:
    warnings = validate_and_warn(payload)
    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    threshold = float(meta["optimal_threshold"])
    df_input = pd.DataFrame([payload])
    prob = float(model.predict_proba(df_input)[0, 1])
    is_defect = prob >= threshold

    return {
        "probabilitas_cacat": round(prob, 4),
        "threshold_bisnis": threshold,
        "prediksi_cacat": int(is_defect),
        "status_qc": "SISIHKAN (POTENSI CACAT)" if is_defect else "LOLOS QC",
        "peringatan_input": warnings,
        "saran_tindakan": generate_operator_actions(payload, prob, threshold),
    }
