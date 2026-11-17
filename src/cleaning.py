import numpy as np
import pandas as pd

STANDAR_BAHAN = {
    "PP": {"suhu_leleh": 230.0, "suhu_cetakan": 40.0, "higroskopis": 0},
    "ABS": {"suhu_leleh": 240.0, "suhu_cetakan": 60.0, "higroskopis": 1},
    "PC": {"suhu_leleh": 300.0, "suhu_cetakan": 90.0, "higroskopis": 1},
}

VALID_MESIN = {"IM-01", "IM-02", "IM-03", "IM-04"}
VALID_SHIFT = {"pagi", "siang", "malam"}
VALID_BAHAN = {"PP", "ABS", "PC"}


def standardize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Menyeragamkan penulisan kategori bahan, shift, dan mesin."""
    out = df.copy()
    if "bahan" in out.columns:
        out["bahan"] = (
            out["bahan"]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.strip()
            .str.upper()
        )
    if "shift" in out.columns:
        out["shift"] = out["shift"].astype(str).str.strip().str.lower()
    if "mesin" in out.columns:
        out["mesin"] = out["mesin"].astype(str).str.strip().str.upper()
    return out


def fix_sensor_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Memperbaiki kesalahan satuan Fahrenheit dan mengubah nilai sentinel menjadi NaN."""
    out = df.copy()

    if "suhu_leleh_C" in out.columns:
        mask_f = out["suhu_leleh_C"] > 350.0
        out.loc[mask_f, "suhu_leleh_C"] = (out.loc[mask_f, "suhu_leleh_C"] - 32.0) * 5.0 / 9.0

    if "tekanan_injeksi_bar" in out.columns:
        out.loc[out["tekanan_injeksi_bar"] <= 0.0, "tekanan_injeksi_bar"] = np.nan

    if "waktu_siklus_s" in out.columns:
        out.loc[out["waktu_siklus_s"] <= 0.0, "waktu_siklus_s"] = np.nan

    if "suhu_cetakan_C" in out.columns:
        out.loc[out["suhu_cetakan_C"] < 12.0, "suhu_cetakan_C"] = np.nan

    if "kelembaban_bahan_pct" in out.columns:
        out.loc[out["kelembaban_bahan_pct"] < 0.0, "kelembaban_bahan_pct"] = np.nan

    return out


def clean_raw_dataset(raw_path: str, clean_path: str | None = None) -> pd.DataFrame:
    """Membersihkan dataset mentah secara menyeluruh untuk penyimpanan data/clean.csv."""
    df = pd.read_csv(raw_path)
    df = df.drop_duplicates().reset_index(drop=True)
    df = standardize_categories(df)
    df = fix_sensor_anomalies(df)

    df = df[
        df["bahan"].isin(VALID_BAHAN)
        & df["shift"].isin(VALID_SHIFT)
        & df["mesin"].isin(VALID_MESIN)
    ].reset_index(drop=True)

    if clean_path:
        df.to_csv(clean_path, index=False)
    return df
