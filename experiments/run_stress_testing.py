"""
experiments/run_stress_testing.py
Stage 5 - Observational Robustness & Stress-Testing Suite

Usage:
    python experiments/run_stress_testing.py

Outputs:
    results/stress_test_degradation.csv
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stress_test.degradations import (
    cadence_dropout_sweep,
    photometric_noise_sweep,
    redshift_sweep,
)
from src.anomaly.detectors import (
    GlobalIsolationForest,
    MultiClassIsolationForest,
    GlobalLOF,
    MultiClassLOF,
)

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "clean_train.parquet"
TEST_PATH  = PROJECT_ROOT / "data" / "processed" / "clean_test.parquet"
RESULTS_PATH = PROJECT_ROOT / "results" / "stress_test_degradation.csv"

LABEL_COL = "classALeRCE"
TRANSIENT_CLASSES = ["SLSN", "SNII", "SNIa", "SNIbc"]
CADENCE_LEVELS  = [0.20, 0.40, 0.60, 0.80]
NOISE_LEVELS    = [1.5, 2.0, 3.0]
REDSHIFT_FRACS  = [0.25, 0.50, 0.75, 1.00]
RANDOM_STATE = 42


def load_data():
    print("[INFO] Loading data...")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df  = pd.read_parquet(TEST_PATH)
    y_train = train_df[LABEL_COL].values
    X_train = train_df.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number]).values
    y_test  = test_df[LABEL_COL].values
    X_test  = test_df.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number]).values
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def fit_models(X_train, y_train):
    models = {
        "Global IF":  GlobalIsolationForest(random_state=RANDOM_STATE),
        "MCIF":       MultiClassIsolationForest(random_state=RANDOM_STATE),
        "Global LOF": GlobalLOF(n_neighbors=20),
        "MCLOF":      MultiClassLOF(n_neighbors=20),
    }
    for name, model in models.items():
        print(f"  [FIT] {name} ...", end=" ", flush=True)
        model.fit(X_train, y_train)
        print("done")
    return models


def evaluate_auroc(models, X_test, y_test, anomaly_class):
    y_binary = (y_test == anomaly_class).astype(int)
    if y_binary.sum() == 0:
        return {name: float("nan") for name in models}
    aurocs = {}
    for name, model in models.items():
        try:
            scores = model.score_samples(X_test)
            aurocs[name] = round(roc_auc_score(y_binary, -scores), 6)
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            aurocs[name] = float("nan")
    return aurocs


def run_experiment(degradation_type, level_label, X_train_deg, X_test_deg, y_train, y_test, baseline_aurocs):
    print(f"\n[EXPERIMENT] {degradation_type} | level={level_label}")
    models = fit_models(X_train_deg, y_train)
    rows = []
    for cls in TRANSIENT_CLASSES:
        aurocs = evaluate_auroc(models, X_test_deg, y_test, cls)
        best_model = max(aurocs, key=lambda k: aurocs[k] if aurocs[k] == aurocs[k] else -1)
        best_auroc = aurocs[best_model]
        baseline = baseline_aurocs.get(cls, {})
        baseline_best = max(baseline.values()) if baseline else float("nan")
        delta = round(best_auroc - baseline_best, 6) if best_auroc == best_auroc else float("nan")
        rows.append({
            "degradation_type": degradation_type,
            "level": level_label,
            "anomaly_class": cls,
            "auroc_global_if":  aurocs.get("Global IF", float("nan")),
            "auroc_global_lof": aurocs.get("Global LOF", float("nan")),
            "auroc_mcif":       aurocs.get("MCIF", float("nan")),
            "auroc_mclof":      aurocs.get("MCLOF", float("nan")),
            "best_model": best_model,
            "best_auroc": best_auroc,
            "delta_vs_baseline": delta,
        })
        print(f"  {cls}: {best_model}={best_auroc:.4f}  delta={delta:+.4f}")
    return rows


def main():
    print("=" * 70)
    print("  Deep_Astro - Stage 5: Observational Stress-Testing Suite")
    print("=" * 70)

    X_train, y_train, X_test, y_test = load_data()

    print("\n[BASELINE] Fitting models on clean data...")
    baseline_models = fit_models(X_train, y_train)
    baseline_aurocs = {cls: evaluate_auroc(baseline_models, X_test, y_test, cls) for cls in TRANSIENT_CLASSES}
    print("\n[BASELINE] Summary:")
    for cls, aurocs in baseline_aurocs.items():
        best = max(aurocs, key=lambda k: aurocs[k] if aurocs[k] == aurocs[k] else -1)
        print(f"  {cls}: {best} = {aurocs[best]:.4f}")

    all_results = []

    print("\n== EXPERIMENT A: Cadence Dropout ==")
    cadence_train = cadence_dropout_sweep(X_train, CADENCE_LEVELS, RANDOM_STATE)
    cadence_test  = cadence_dropout_sweep(X_test,  CADENCE_LEVELS, RANDOM_STATE)
    for level in CADENCE_LEVELS:
        all_results.extend(run_experiment("cadence_dropout", f"{int(level*100)}%",
            cadence_train[level], cadence_test[level], y_train, y_test, baseline_aurocs))

    print("\n== EXPERIMENT B: Photometric Noise ==")
    noise_train = photometric_noise_sweep(X_train, NOISE_LEVELS, RANDOM_STATE)
    noise_test  = photometric_noise_sweep(X_test,  NOISE_LEVELS, RANDOM_STATE)
    for scale in NOISE_LEVELS:
        all_results.extend(run_experiment("photometric_noise", f"x{scale}",
            noise_train[scale], noise_test[scale], y_train, y_test, baseline_aurocs))

    print("\n== EXPERIMENT C: Redshift Removal ==")
    redshift_train = redshift_sweep(X_train, REDSHIFT_FRACS, RANDOM_STATE)
    redshift_test  = redshift_sweep(X_test,  REDSHIFT_FRACS, RANDOM_STATE)
    for frac in REDSHIFT_FRACS:
        all_results.extend(run_experiment("redshift_removal", f"{int(frac*100)}%_missing",
            redshift_train[frac], redshift_test[frac], y_train, y_test, baseline_aurocs))

    results_df = pd.DataFrame(all_results)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\n[DONE] Results saved to {RESULTS_PATH}")
    print(results_df.to_string(index=False))
    print("\n[Stage 5 Complete]")


if __name__ == "__main__":
    main()
