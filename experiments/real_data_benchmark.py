"""Real-Data Benchmark Experiment on Cleaned ZTF ALeRCE Dataset.

Replicates and extends Table 2 from Gupta et al. (2025):
Compares:
1. Global Isolation Forest (Baseline)
2. Global Local Outlier Factor (LOF Baseline)
3. Multi-Class Isolation Forest (MCIF - Paper)
4. Multi-Class LOF (MCLOF - Our Extension)
5. MCIF + Cluster-Standardized Thresholding (Our Objective 1 Innovation)

Across transient astronomical classes using one-vs-rest anomaly testing.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.anomaly.detectors import (
    GlobalIsolationForest,
    MultiClassIsolationForest,
    GlobalLOF,
    MultiClassLOF,
)
from src.thresholding.adaptive import ClusterStandardizedScorer
from src.evaluation.metrics import compute_auroc, compute_aucpr


def run_real_data_benchmark(
    train_parquet_path: str = "data/processed/clean_train.parquet",
    test_parquet_path: str = "data/processed/clean_test.parquet",
    target_category: str = "Transient",
    n_estimators: int = 150,
    n_neighbors: int = 20,
    random_state: int = 42,
):
    print("=" * 85)
    print(f"        DEEP_ASTRO REAL-DATA BENCHMARK: {target_category.upper()} CATEGORY        ")
    print("=" * 85)

    print(f"Loading preprocessed data...")
    train_df = pd.read_parquet(train_parquet_path)
    test_df = pd.read_parquet(test_parquet_path)

    # Filter to target category
    train_cat = train_df[train_df["hierClass"] == target_category].copy().reset_index(drop=True)
    test_cat = test_df[test_df["hierClass"] == target_category].copy().reset_index(drop=True)

    classes_in_cat = sorted(train_cat["classALeRCE"].unique())
    print(f"Target Category: {target_category}")
    print(f"Classes evaluated: {classes_in_cat}")
    print(f"Available samples: Train={len(train_cat):,}, Test={len(test_cat):,}")

    meta_cols = ["oid", "classALeRCE", "hierClass"]
    feature_cols = [c for c in train_cat.columns if c not in meta_cols and np.issubdtype(train_cat[c].dtype, np.number)]

    results = []

    print("-" * 85)
    print(f"{'Target Anomaly':<16} | {'Global IF':<10} | {'Global LOF':<10} | {'MCIF (Paper)':<12} | {'MCLOF (Ours)':<12} | {'MCIF+Adapt':<10}")
    print("-" * 85)

    for outlier_class in classes_in_cat:
        # Define normal classes in this category
        normal_classes = [c for c in classes_in_cat if c != outlier_class]

        # Training set: Only normal classes
        train_sub = train_cat[train_cat["classALeRCE"].isin(normal_classes)].copy().reset_index(drop=True)
        # Validation set: Stratified sample of normal classes for calibrating adaptive thresholds
        val_sub = train_sub.sample(frac=0.2, random_state=random_state)
        train_sub = train_sub.drop(val_sub.index).reset_index(drop=True)
        val_sub = val_sub.reset_index(drop=True)

        # Test set: Normal classes (0) vs Outlier class (1)
        test_sub = test_cat.copy().reset_index(drop=True)
        y_test_binary = np.where(test_sub["classALeRCE"] == outlier_class, 1, 0)

        # Check if outlier exists in test
        n_outliers = np.sum(y_test_binary == 1)
        n_normals = np.sum(y_test_binary == 0)
        if n_outliers == 0 or n_normals == 0:
            print(f"Skipping {outlier_class}: Insufficient test samples (Outliers={n_outliers}, Normals={n_normals})")
            continue

        X_train = train_sub[feature_cols].values
        y_train = train_sub["classALeRCE"].values

        X_val = val_sub[feature_cols].values
        y_val = val_sub["classALeRCE"].values

        X_test = test_sub[feature_cols].values

        # 1. Global Isolation Forest
        gif = GlobalIsolationForest(n_estimators=n_estimators, random_state=random_state)
        gif.fit(X_train)
        scores_gif = gif.score(X_test)
        auroc_gif = compute_auroc(y_test_binary, scores_gif)

        # 2. Global LOF
        k = min(n_neighbors, len(X_train) - 1)
        glof = GlobalLOF(n_neighbors=k)
        glof.fit(X_train)
        scores_glof = glof.score(X_test)
        auroc_glof = compute_auroc(y_test_binary, scores_glof)

        # 3. MCIF (Paper Baseline)
        mcif = MultiClassIsolationForest(n_estimators=n_estimators, random_state=random_state)
        mcif.fit(X_train, y_train)
        scores_mcif = mcif.score(X_test)
        auroc_mcif = compute_auroc(y_test_binary, scores_mcif)

        # 4. Multi-Class LOF (MCLOF - Our Innovation)
        mclof = MultiClassLOF(n_neighbors=n_neighbors)
        mclof.fit(X_train, y_train)
        scores_mclof = mclof.score(X_test)
        auroc_mclof = compute_auroc(y_test_binary, scores_mclof)

        # 5. MCIF + Cluster-Standardized Adaptive Thresholding (Objective 1)
        val_matrix = mcif.score_per_class(X_val)
        test_matrix = mcif.score_per_class(X_test)
        adaptive_scorer = ClusterStandardizedScorer()
        adaptive_scorer.fit(val_matrix, y_val, normal_classes)
        scores_adapt = adaptive_scorer.score(test_matrix)
        auroc_adapt = compute_auroc(y_test_binary, scores_adapt)

        results.append({
            "Category": target_category,
            "Anomaly_Class": outlier_class,
            "AUROC_Global_IF": auroc_gif,
            "AUROC_Global_LOF": auroc_glof,
            "AUROC_MCIF": auroc_mcif,
            "AUROC_MCLOF": auroc_mclof,
            "AUROC_MCIF_Adaptive": auroc_adapt,
            "Num_Outliers_Test": int(n_outliers),
            "Num_Normals_Test": int(n_normals),
        })

        print(f"{outlier_class:<16} | {auroc_gif:<10.4f} | {auroc_glof:<10.4f} | {auroc_mcif:<12.4f} | {auroc_mclof:<12.4f} | {auroc_adapt:<10.4f}")

    print("-" * 85)

    # Save results to CSV
    os.makedirs("results", exist_ok=True)
    out_csv = f"results/benchmark_{target_category.lower()}.csv"
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_csv, index=False)
    print(f"Results successfully saved to {out_csv}")
    print("=" * 85)
    return res_df


if __name__ == "__main__":
    category = "Transient"
    if len(sys.argv) > 1:
        category = sys.argv[1]
    run_real_data_benchmark(target_category=category)
