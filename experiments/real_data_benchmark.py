"""Full 14-Class Real-Data Benchmark on ZTF ALeRCE Dataset.

Covers all 3 Astronomical Categories:
- Transient: SLSN, SNII, SNIa, SNIbc
- Stochastic: AGN, Blazar, CV/Nova, QSO, YSO
- Periodic: CEP, DSCT, E, RRL, LPV

Compares:
1. Global Isolation Forest
2. Global LOF (Our Baseline)
3. MCIF (Base Paper)
4. Multi-Class LOF (MCLOF - Our Extension)
5. MCIF + Cluster-Standardized Adaptive Thresholding (Objective 1)
"""

import os
import sys
import numpy as np
import pandas as pd

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


def run_category_benchmark(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    category: str,
    feature_cols: list,
    max_train_per_class: int = 1500,
    n_estimators: int = 150,
    n_neighbors: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Runs one-vs-rest anomaly detection across all classes in an astronomical category."""
    rng = np.random.RandomState(random_state)
    train_cat = train_df[train_df["hierClass"] == category].copy().reset_index(drop=True)
    test_cat = test_df[test_df["hierClass"] == category].copy().reset_index(drop=True)

    classes_in_cat = sorted(train_cat["classALeRCE"].unique())
    print("\n" + "=" * 90)
    print(f" CATEGORY: {category.upper()} ({len(classes_in_cat)} Classes: {', '.join(classes_in_cat)})")
    print(f" Available Samples: Train={len(train_cat):,}, Test={len(test_cat):,}")
    print("-" * 90)
    print(f"{'Target Outlier':<16} | {'Global IF':<10} | {'Global LOF':<10} | {'MCIF (Paper)':<12} | {'MCLOF (Ours)':<12} | {'MCIF+Adapt':<10} | {'Top Method'}")
    print("-" * 90)

    category_results = []

    for outlier_class in classes_in_cat:
        normal_classes = [c for c in classes_in_cat if c != outlier_class]

        # 1. Prepare Training Subset (normals only, capped per class for laptop performance)
        train_sub_dfs = []
        val_sub_dfs = []
        for nc in normal_classes:
            nc_df = train_cat[train_cat["classALeRCE"] == nc].copy()
            if len(nc_df) > max_train_per_class:
                nc_df = nc_df.sample(n=max_train_per_class, random_state=random_state)
            # 80/20 train/val split
            val_part = nc_df.sample(frac=0.2, random_state=random_state)
            train_part = nc_df.drop(val_part.index)
            train_sub_dfs.append(train_part)
            val_sub_dfs.append(val_part)

        train_sub = pd.concat(train_sub_dfs, ignore_index=True)
        val_sub = pd.concat(val_sub_dfs, ignore_index=True)

        # 2. Prepare Testing Set (normals=0 vs outlier=1)
        test_sub = test_cat.copy().reset_index(drop=True)
        y_test_binary = np.where(test_sub["classALeRCE"] == outlier_class, 1, 0)

        n_outliers = np.sum(y_test_binary == 1)
        n_normals = np.sum(y_test_binary == 0)
        if n_outliers == 0 or n_normals == 0:
            continue

        X_train = train_sub[feature_cols].values
        y_train = train_sub["classALeRCE"].values

        X_val = val_sub[feature_cols].values
        y_val = val_sub["classALeRCE"].values

        X_test = test_sub[feature_cols].values

        # 3. Fit and Score Detectors
        # A. Global Isolation Forest
        gif = GlobalIsolationForest(n_estimators=n_estimators, random_state=random_state)
        gif.fit(X_train)
        auroc_gif = compute_auroc(y_test_binary, gif.score(X_test))

        # B. Global LOF
        k = min(n_neighbors, len(X_train) - 1)
        glof = GlobalLOF(n_neighbors=k)
        glof.fit(X_train)
        auroc_glof = compute_auroc(y_test_binary, glof.score(X_test))

        # C. MCIF (Paper Baseline)
        mcif = MultiClassIsolationForest(n_estimators=n_estimators, random_state=random_state)
        mcif.fit(X_train, y_train)
        auroc_mcif = compute_auroc(y_test_binary, mcif.score(X_test))

        # D. MCLOF (Our Extension)
        mclof = MultiClassLOF(n_neighbors=n_neighbors)
        mclof.fit(X_train, y_train)
        auroc_mclof = compute_auroc(y_test_binary, mclof.score(X_test))

        # E. MCIF + Adaptive Thresholding (Objective 1)
        val_matrix = mcif.score_per_class(X_val)
        test_matrix = mcif.score_per_class(X_test)
        adaptive_scorer = ClusterStandardizedScorer()
        adaptive_scorer.fit(val_matrix, y_val, normal_classes)
        auroc_adapt = compute_auroc(y_test_binary, adaptive_scorer.score(test_matrix))

        # Determine winner
        scores_map = {
            "Global IF": auroc_gif,
            "Global LOF": auroc_glof,
            "MCIF": auroc_mcif,
            "MCLOF": auroc_mclof,
            "MCIF+Adapt": auroc_adapt,
        }
        best_method = max(scores_map, key=scores_map.get)

        row = {
            "Category": category,
            "Anomaly_Class": outlier_class,
            "AUROC_Global_IF": auroc_gif,
            "AUROC_Global_LOF": auroc_glof,
            "AUROC_MCIF": auroc_mcif,
            "AUROC_MCLOF": auroc_mclof,
            "AUROC_MCIF_Adaptive": auroc_adapt,
            "Best_Model": best_method,
            "Best_AUROC": scores_map[best_method],
            "Num_Outliers": int(n_outliers),
            "Num_Normals": int(n_normals),
        }
        category_results.append(row)

        print(f"{outlier_class:<16} | {auroc_gif:<10.4f} | {auroc_glof:<10.4f} | {auroc_mcif:<12.4f} | {auroc_mclof:<12.4f} | {auroc_adapt:<10.4f} | {best_method}")

    return pd.DataFrame(category_results)


def run_full_14_class_benchmark():
    train_parquet = "data/processed/clean_train.parquet"
    test_parquet = "data/processed/clean_test.parquet"

    print("Loading cleaned datasets...")
    train_df = pd.read_parquet(train_parquet)
    test_df = pd.read_parquet(test_parquet)

    meta_cols = ["oid", "classALeRCE", "hierClass"]
    feature_cols = [c for c in train_df.columns if c not in meta_cols and np.issubdtype(train_df[c].dtype, np.number)]

    categories = ["Transient", "Stochastic", "Periodic"]
    all_results = []

    for cat in categories:
        df_cat = run_category_benchmark(train_df, test_df, cat, feature_cols)
        all_results.append(df_cat)

    master_df = pd.concat(all_results, ignore_index=True)

    os.makedirs("results", exist_ok=True)
    master_csv = "results/benchmark_all_14_classes.csv"
    master_df.to_csv(master_csv, index=False)

    print("\n" + "=" * 90)
    print("                     BENCHMARK SUMMARY ACROSS ALL 14 CLASSES                    ")
    print("=" * 90)
    print(f"Mean AUROC across all 14 classes:")
    print(f"  • Global Isolation Forest: {master_df['AUROC_Global_IF'].mean():.4f}")
    print(f"  • Global LOF (Our Base):   {master_df['AUROC_Global_LOF'].mean():.4f}")
    print(f"  • MCIF (Paper Baseline):   {master_df['AUROC_MCIF'].mean():.4f}")
    print(f"  • MCLOF (Our Extension):   {master_df['AUROC_MCLOF'].mean():.4f}")
    print(f"  • MCIF + Adaptive (Obj 1): {master_df['AUROC_MCIF_Adaptive'].mean():.4f}")
    print("-" * 90)
    print("Win distribution across 14 classes:")
    print(master_df["Best_Model"].value_counts().to_string())
    print("=" * 90)
    print(f"Master results saved to {master_csv}")


if __name__ == "__main__":
    run_full_14_class_benchmark()
