import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.cleaning import clean_raw_dataset
from src.features import InjectionMoldingFeatureEngineer

COST_FN = 500_000  # Cacat lolos ke pelanggan (Rp500.000)
COST_FP = 25_000   # Bagus ikut disisihkan (Rp25.000)


def build_preprocessor(use_engineered: bool = True, scale_numeric: bool = False):
    cat_cols = ["mesin", "shift", "bahan"]
    base_num = [
        "suhu_leleh_C",
        "suhu_cetakan_C",
        "tekanan_injeksi_bar",
        "kecepatan_injeksi_mm_s",
        "waktu_pendinginan_s",
        "waktu_siklus_s",
        "kelembaban_bahan_pct",
    ]
    eng_num = [
        "dev_suhu_leleh",
        "abs_dev_suhu_leleh",
        "dev_suhu_cetakan",
        "abs_dev_suhu_cetakan",
        "selisih_suhu_leleh_cetakan",
        "is_higroskopis",
        "risiko_kelembaban_higro",
        "kelembaban_berlebih_higro",
        "rasio_pendinginan",
        "waktu_non_pendinginan",
        "defisit_tekanan_bar",
        "abs_dev_kecepatan",
        "daya_injeksi_relatif",
        "is_mesin_im03",
        "is_shift_malam",
        "is_missing_kelembaban",
        "is_missing_pendinginan",
    ]
    num_cols = base_num + eng_num if use_engineered else base_num

    num_transformer = StandardScaler() if scale_numeric else "passthrough"
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ],
        remainder="drop",
    )


def find_optimal_business_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    thresholds = np.linspace(0.01, 0.50, 99)
    best = None

    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = int(fn * COST_FN + fp * COST_FP)
        avg_cost_per_part = total_cost / len(y_true)

        if best is None or total_cost < best["total_cost_rp"]:
            best = {
                "threshold": round(float(th), 4),
                "total_cost_rp": total_cost,
                "avg_cost_per_part_rp": round(float(avg_cost_per_part), 2),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
                "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            }
    return best


def evaluate_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = int(fn * COST_FN + fp * COST_FP)
    return {
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "total_cost_rp": total_cost,
        "avg_cost_per_part_rp": round(float(total_cost / len(y_true)), 2),
    }


def main():
    Path("artifacts").mkdir(exist_ok=True)
    df = clean_raw_dataset("data/raw.csv", "data/clean.csv")

    X = df.drop(columns=["cacat"])
    y = df["cacat"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    baseline_pipe = Pipeline(
        [
            ("fe", InjectionMoldingFeatureEngineer(use_engineered_features=False)),
            ("prep", build_preprocessor(use_engineered=False, scale_numeric=True)),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    baseline_pipe.fit(X_train, y_train)
    val_prob_base = baseline_pipe.predict_proba(X_val)[:, 1]
    val_pr_auc_base = average_precision_score(y_val, val_prob_base)
    base_th_info = find_optimal_business_threshold(y_val, val_prob_base)

    candidates = {
        "HistGradientBoosting": Pipeline(
            [
                ("fe", InjectionMoldingFeatureEngineer(use_engineered_features=True)),
                ("prep", build_preprocessor(use_engineered=True, scale_numeric=False)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        min_samples_leaf=30,
                        l2_regularization=1.0,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("fe", InjectionMoldingFeatureEngineer(use_engineered_features=True)),
                ("prep", build_preprocessor(use_engineered=True, scale_numeric=False)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=14,
                        min_samples_leaf=10,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    best_name = None
    best_pipe = None
    best_val_pr_auc = -1.0
    best_val_th_info = None
    val_summary = {
        "Baseline_LogisticRegression": {
            "pr_auc": round(float(val_pr_auc_base), 4),
            **base_th_info,
        }
    }

    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        prob_val = pipe.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, prob_val)
        th_info = find_optimal_business_threshold(y_val, prob_val)
        val_summary[name] = {"pr_auc": round(float(pr_auc), 4), **th_info}

        if pr_auc > best_val_pr_auc:
            best_val_pr_auc = pr_auc
            best_name = name
            best_pipe = pipe
            best_val_th_info = th_info

    test_prob_base = baseline_pipe.predict_proba(X_test)[:, 1]
    test_metrics_base = evaluate_at_threshold(y_test, test_prob_base, base_th_info["threshold"])

    test_prob_final = best_pipe.predict_proba(X_test)[:, 1]
    test_metrics_final = evaluate_at_threshold(y_test, test_prob_final, best_val_th_info["threshold"])

    joblib.dump(baseline_pipe, "artifacts/baseline_model.joblib")
    joblib.dump(best_pipe, "artifacts/model.joblib")

    metadata = {
        "kasus": "01 - Pengendalian Mutu Injection Molding",
        "model_final": best_name,
        "dataset_clean_rows": int(len(df)),
        "split_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "business_cost_rp": {"false_negative": COST_FN, "false_positive": COST_FP},
        "optimal_threshold": best_val_th_info["threshold"],
        "validation_results": val_summary,
        "test_results_opened_once": {
            "Baseline_LogisticRegression": test_metrics_base,
            best_name: test_metrics_final,
        },
        "input_ranges": {
            "suhu_leleh_C": [190.0, 340.0],
            "suhu_cetakan_C": [15.0, 120.0],
            "tekanan_injeksi_bar": [400.0, 1400.0],
            "kecepatan_injeksi_mm_s": [20.0, 150.0],
            "waktu_pendinginan_s": [5.0, 40.0],
            "waktu_siklus_s": [12.0, 60.0],
            "kelembaban_bahan_pct": [0.0, 0.60],
        },
    }

    with open("artifacts/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Pelatihan selesai. Metadata tersimpan di artifacts/metadata.json")
    print(json.dumps(metadata["test_results_opened_once"], indent=2))


if __name__ == "__main__":
    main()
