"""Dataset Cleaning and Preprocessing Pipeline for Deep_Astro.

Performs:
1. Removal of unidentified / contaminated classes (e.g. Periodic-Other).
2. Leakage-free feature imputation (median imputation fitted on train split).
3. Outlier-robust scaling (RobustScaler with IQR bounds).
4. Export to clean compressed formats (Parquet + NPZ) and stratified splits.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


def clean_benchmark_dataset(
    raw_dir: str = "data/benchmark_alerce/data",
    output_dir: str = "data/processed",
    subsample_train_size: int = 5000,
    random_state: int = 42,
):
    """Cleans, imputes, normalizes, and packages the benchmark transient dataset."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(random_state)

    train_path = os.path.join(raw_dir, "train_data_filtered.pkl")
    test_path = os.path.join(raw_dir, "test_data_filtered.pkl")

    print(f"Loading raw datasets from {raw_dir}...")
    train_df = pd.read_pickle(train_path)
    test_df = pd.read_pickle(test_path)

    print(f"  Raw train shape: {train_df.shape}")
    print(f"  Raw test shape:  {test_df.shape}")

    # 1. Filter out unclassified / noisy class
    print("\n[Step 1] Filtering unclassified noise ('Periodic-Other')...")
    train_df = train_df[train_df["classALeRCE"] != "Periodic-Other"].reset_index(drop=True)
    test_df = test_df[test_df["classALeRCE"] != "Periodic-Other"].reset_index(drop=True)

    print(f"  Cleaned train shape: {train_df.shape}")
    print(f"  Cleaned test shape:  {test_df.shape}")

    # 2. Separate metadata and numerical feature columns
    meta_cols = ["oid", "classALeRCE", "hierClass"]
    feature_cols = [c for c in train_df.columns if c not in meta_cols and np.issubdtype(train_df[c].dtype, np.number)]
    print(f"\n[Step 2] Extracted {len(feature_cols)} numerical astronomical features.")

    # 3. Leakage-free imputation & robust scaling
    print("[Step 3] Imputing missing values and applying RobustScaler...")
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(5.0, 95.0))

    X_train_raw = train_df[feature_cols].values
    X_test_raw = test_df[feature_cols].values

    # Fit ONLY on training data to prevent data leakage
    X_train_clean = imputer.fit_transform(X_train_raw)
    X_train_scaled = scaler.fit_transform(X_train_clean)

    # Transform test data using training statistics
    X_test_clean = imputer.transform(X_test_raw)
    X_test_scaled = scaler.transform(X_test_clean)

    # Reassemble clean DataFrames
    clean_train_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
    clean_test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)

    for col in meta_cols:
        clean_train_df[col] = train_df[col]
        clean_test_df[col] = test_df[col]

    # 4. Save clean master files
    train_out_parquet = os.path.join(output_dir, "clean_train.parquet")
    test_out_parquet = os.path.join(output_dir, "clean_test.parquet")
    print(f"\n[Step 4] Saving clean Parquet files to {output_dir}...")
    clean_train_df.to_parquet(train_out_parquet, index=False)
    clean_test_df.to_parquet(test_out_parquet, index=False)

    # 5. Create Stratified Laptop-Optimized Subset (5k train, nature-matched test)
    print("\n[Step 5] Creating lightweight stratified subsets for fast laptop experimentation...")
    classes = np.unique(clean_train_df["classALeRCE"])
    stratified_train_idx = []

    samples_per_class = max(15, subsample_train_size // len(classes))
    for c in classes:
        c_idx = np.where(clean_train_df["classALeRCE"] == c)[0]
        n_take = min(len(c_idx), samples_per_class)
        chosen = rng.choice(c_idx, size=n_take, replace=False)
        stratified_train_idx.extend(chosen)

    stratified_train_idx = np.array(stratified_train_idx)
    rng.shuffle(stratified_train_idx)

    X_train_sub = X_train_scaled[stratified_train_idx]
    y_train_sub = clean_train_df["classALeRCE"].values[stratified_train_idx]
    hier_train_sub = clean_train_df["hierClass"].values[stratified_train_idx]

    sub_out_npz = os.path.join(output_dir, "stratified_train_5k.npz")
    np.savez_compressed(
        sub_out_npz,
        X=X_train_sub,
        y=y_train_sub,
        hier=hier_train_sub,
        features=feature_cols,
    )
    print(f"  Saved {len(stratified_train_idx)} stratified training samples to {sub_out_npz}")

    # Save summary metadata
    summary = {
        "num_features": len(feature_cols),
        "feature_names": feature_cols,
        "classes": list(classes),
        "train_samples_total": len(clean_train_df),
        "test_samples_total": len(clean_test_df),
        "train_samples_stratified": len(stratified_train_idx),
        "class_counts_train": clean_train_df["classALeRCE"].value_counts().to_dict(),
        "class_counts_test": clean_test_df["classALeRCE"].value_counts().to_dict(),
    }
    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved summary metadata to {summary_path}")
    print("\n[✓] Dataset cleaning and preprocessing completed successfully!")


if __name__ == "__main__":
    clean_benchmark_dataset()
