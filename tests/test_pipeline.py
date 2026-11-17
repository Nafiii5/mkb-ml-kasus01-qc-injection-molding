import pandas as pd
from src.cleaning import standardize_categories, fix_sensor_anomalies
from src.predict import predict_cycle


def test_cleaning_standardization_and_fahrenheit():
    df = pd.DataFrame(
        [
            {
                "bahan": "P.P",
                "shift": "Pagi",
                "mesin": "im-01",
                "suhu_leleh_C": 446.0,
                "tekanan_injeksi_bar": 0.0,
                "waktu_siklus_s": -1.0,
            }
        ]
    )
    out = fix_sensor_anomalies(standardize_categories(df))
    assert out.loc[0, "bahan"] == "PP"
    assert out.loc[0, "shift"] == "pagi"
    assert out.loc[0, "mesin"] == "IM-01"
    assert abs(out.loc[0, "suhu_leleh_C"] - 230.0) < 0.01
    assert pd.isna(out.loc[0, "tekanan_injeksi_bar"])
    assert pd.isna(out.loc[0, "waktu_siklus_s"])


def test_predict_unusual_inputs():
    payload = {
        "mesin": "IM-02",
        "shift": "siang",
        "bahan": "PC",
        "suhu_leleh_C": 575.0,
        "suhu_cetakan_C": 90.0,
        "tekanan_injeksi_bar": 0.0,
        "kecepatan_injeksi_mm_s": 80.0,
        "waktu_pendinginan_s": 18.0,
        "waktu_siklus_s": -1.0,
        "kelembaban_bahan_pct": None,
    }
    res = predict_cycle(payload)
    assert 0.0 <= res["probabilitas_cacat"] <= 1.0
    assert len(res["peringatan_input"]) >= 3
    assert len(res["saran_tindakan"]) >= 1
