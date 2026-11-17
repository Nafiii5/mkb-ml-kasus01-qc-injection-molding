import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.cleaning import STANDAR_BAHAN, standardize_categories, fix_sensor_anomalies


class InjectionMoldingFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer Scikit-Learn untuk:
    1. Imputasi nilai hilang berbasis median per jenis bahan (fit hanya pada train).
    2. Pembuatan fitur fisika proses injection molding untuk melampaui baseline.
    """

    def __init__(self, use_engineered_features: bool = True):
        self.use_engineered_features = use_engineered_features

    def fit(self, X: pd.DataFrame, y=None):
        df = fix_sensor_anomalies(standardize_categories(X))
        self.medians_by_bahan_ = {}
        num_cols = [
            "suhu_leleh_C",
            "suhu_cetakan_C",
            "tekanan_injeksi_bar",
            "kecepatan_injeksi_mm_s",
            "waktu_pendinginan_s",
            "waktu_siklus_s",
            "kelembaban_bahan_pct",
        ]
        self.global_medians_ = {col: float(df[col].median()) for col in num_cols}

        for bahan in ["PP", "ABS", "PC"]:
            sub = df[df["bahan"] == bahan]
            self.medians_by_bahan_[bahan] = {
                col: float(sub[col].median()) if len(sub) > 0 else self.global_medians_[col]
                for col in num_cols
            }

        valid_diff = df["waktu_siklus_s"] - df["waktu_pendinginan_s"]
        self.median_non_cooling_ = float(valid_diff.dropna().median()) if valid_diff.notna().any() else 14.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = fix_sensor_anomalies(standardize_categories(X))

        missing_kelembaban = df["kelembaban_bahan_pct"].isna().astype(float)
        missing_pendinginan = df["waktu_pendinginan_s"].isna().astype(float)

        num_cols = list(self.global_medians_.keys())
        for col in num_cols:
            if df[col].isna().any():
                mapped_med = df["bahan"].map(
                    lambda b: self.medians_by_bahan_.get(b, self.global_medians_)[col]
                )
                df[col] = df[col].fillna(mapped_med).fillna(self.global_medians_[col])

        invalid_cycle = df["waktu_siklus_s"] <= df["waktu_pendinginan_s"]
        df.loc[invalid_cycle, "waktu_siklus_s"] = (
            df.loc[invalid_cycle, "waktu_pendinginan_s"] + self.median_non_cooling_
        )

        base_cols = [
            "mesin",
            "shift",
            "bahan",
            "suhu_leleh_C",
            "suhu_cetakan_C",
            "tekanan_injeksi_bar",
            "kecepatan_injeksi_mm_s",
            "waktu_pendinginan_s",
            "waktu_siklus_s",
            "kelembaban_bahan_pct",
        ]

        if not self.use_engineered_features:
            return df[base_cols].copy()

        std_melt = df["bahan"].map(lambda b: STANDAR_BAHAN[b]["suhu_leleh"])
        std_mold = df["bahan"].map(lambda b: STANDAR_BAHAN[b]["suhu_cetakan"])
        is_higro = df["bahan"].map(lambda b: STANDAR_BAHAN[b]["higroskopis"]).astype(float)

        df["dev_suhu_leleh"] = df["suhu_leleh_C"] - std_melt
        df["abs_dev_suhu_leleh"] = df["dev_suhu_leleh"].abs()
        df["dev_suhu_cetakan"] = df["suhu_cetakan_C"] - std_mold
        df["abs_dev_suhu_cetakan"] = df["dev_suhu_cetakan"].abs()
        df["selisih_suhu_leleh_cetakan"] = df["suhu_leleh_C"] - df["suhu_cetakan_C"]

        df["is_higroskopis"] = is_higro
        df["risiko_kelembaban_higro"] = is_higro * df["kelembaban_bahan_pct"]
        df["kelembaban_berlebih_higro"] = is_higro * np.maximum(0.0, df["kelembaban_bahan_pct"] - 0.08)

        df["rasio_pendinginan"] = df["waktu_pendinginan_s"] / df["waktu_siklus_s"]
        df["waktu_non_pendinginan"] = df["waktu_siklus_s"] - df["waktu_pendinginan_s"]

        df["defisit_tekanan_bar"] = np.maximum(0.0, 850.0 - df["tekanan_injeksi_bar"])
        df["abs_dev_kecepatan"] = (df["kecepatan_injeksi_mm_s"] - 80.0).abs()
        df["daya_injeksi_relatif"] = (df["tekanan_injeksi_bar"] * df["kecepatan_injeksi_mm_s"]) / 1000.0

        df["is_mesin_im03"] = (df["mesin"] == "IM-03").astype(float)
        df["is_shift_malam"] = (df["shift"] == "malam").astype(float)
        df["is_missing_kelembaban"] = missing_kelembaban
        df["is_missing_pendinginan"] = missing_pendinginan

        return df.drop(columns=["waktu_produksi"], errors="ignore")
