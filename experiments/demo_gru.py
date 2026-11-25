"""End-to-End GRU + MCIF Live Demo Script for Deep_Astro.

Demonstrates the complete sequential pipeline:
1. Build synthetic multi-passband light curves (N, 656, 4) with variable lengths & zero-padding.
2. Build and train the Dual-Stream GRU Encoder (src/models/encoder.py).
3. Extract 100D latent vectors via extract_latent_space().
4. Fit MultiClassIsolationForest (MCIF) on the 100D latent embeddings.
5. Score normal test transients vs. unseen anomalous shapes.
"""

import os
import sys
import numpy as np

# Suppress verbose TensorFlow C++ info logs for a cleaner demo output
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Ensure repository root is on sys.path (allows running via VS Code Play button)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.encoder import build_transient_encoder, extract_latent_space
from src.anomaly.detectors import MultiClassIsolationForest
from src.thresholding.adaptive import ClusterStandardizedScorer
import tensorflow as tf


def _make_light_curve(
    rng: np.random.RandomState,
    shape_type: str,
    ntimesteps: int = 656,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generates one padded light curve (656, 4) and context vector (2,)."""
    seq_len = int(rng.randint(45, 95))
    t = np.linspace(0.0, 1.0, seq_len)
    wavelength = rng.choice([0.4827, 0.6233], size=seq_len)  # ZTF g and r bands

    if shape_type == "SNIa-like":
        flux = 2.0 * np.exp(-((t - 0.25) ** 2) / 0.015) + rng.normal(0, 0.05, seq_len)
        redshift = rng.uniform(0.05, 0.20)
    elif shape_type == "SNII-like":
        rise = 1.0 / (1.0 + np.exp(-25.0 * (t - 0.15)))
        plateau = np.exp(-1.5 * np.maximum(0.0, t - 0.20))
        flux = 1.6 * rise * plateau + rng.normal(0, 0.05, seq_len)
        redshift = rng.uniform(0.02, 0.15)
    elif shape_type == "Periodic-like":
        flux = 1.0 + 0.8 * np.sin(2.0 * np.pi * 4.0 * t) + rng.normal(0, 0.05, seq_len)
        redshift = rng.uniform(0.0, 0.02)
    elif shape_type == "Anomaly-FastSpike":
        # Ultra-fast, high-luminosity kilonova-like flare at mid-epoch
        flux = 6.0 * np.exp(-((t - 0.55) ** 2) / 0.0008) + rng.normal(0, 0.05, seq_len)
        redshift = rng.uniform(0.01, 0.04)
    elif shape_type == "Anomaly-DoubleBurst":
        # Unusual double-peaked exotic transient
        flux = (
            4.5 * np.exp(-((t - 0.20) ** 2) / 0.003)
            + 5.0 * np.exp(-((t - 0.80) ** 2) / 0.003)
            + rng.normal(0, 0.05, seq_len)
        )
        redshift = rng.uniform(0.30, 0.50)
    else:
        raise ValueError(f"Unknown shape_type: {shape_type}")

    error = np.full(seq_len, 0.05, dtype=np.float32)

    lc = np.zeros((ntimesteps, 4), dtype=np.float32)
    lc[:seq_len, 0] = flux
    lc[:seq_len, 1] = error
    lc[:seq_len, 2] = t
    lc[:seq_len, 3] = wavelength

    ebv = rng.uniform(0.01, 0.12)
    context = np.array([ebv, redshift], dtype=np.float32)
    return lc, context, seq_len


def build_demo_dataset(n_train_per_class: int = 35, n_val_per_class: int = 10, random_state: int = 42):
    """Builds small train, validation, and test datasets of padded sequential light curves."""
    rng = np.random.RandomState(random_state)
    common_classes = ["SNIa-like", "SNII-like", "Periodic-like"]

    def _sample_split(count_per_class: int):
        lcs, ctxs, y_idx = [], [], []
        for idx, cls_name in enumerate(common_classes):
            for _ in range(count_per_class):
                lc, ctx, _ = _make_light_curve(rng, cls_name)
                lcs.append(lc)
                ctxs.append(ctx)
                y_idx.append(idx)
        y_idx = np.array(y_idx)
        return (
            np.array(lcs, dtype=np.float32),
            np.array(ctxs, dtype=np.float32),
            np.eye(len(common_classes), dtype=np.float32)[y_idx],
            np.array([common_classes[i] for i in y_idx]),
        )

    lc_train, ctx_train, y_train_onehot, y_train_labels = _sample_split(n_train_per_class)
    lc_val, ctx_val, _, y_val_labels = _sample_split(n_val_per_class)

    # Test set: 6 normal events (2 per common class) + 4 unseen anomalies
    test_specs = [
        ("SNIa-like", False),
        ("SNIa-like", False),
        ("SNII-like", False),
        ("SNII-like", False),
        ("Periodic-like", False),
        ("Periodic-like", False),
        ("Anomaly-FastSpike", True),
        ("Anomaly-FastSpike", True),
        ("Anomaly-DoubleBurst", True),
        ("Anomaly-DoubleBurst", True),
    ]
    lc_test, ctx_test, test_names, test_is_anom, test_lens = [], [], [], [], []
    for cls_name, is_anom in test_specs:
        lc, ctx, seq_len = _make_light_curve(rng, cls_name)
        lc_test.append(lc)
        ctx_test.append(ctx)
        test_names.append(cls_name)
        test_is_anom.append(is_anom)
        test_lens.append(seq_len)

    return (
        lc_train, ctx_train, y_train_onehot, y_train_labels,
        lc_val, ctx_val, y_val_labels,
        np.array(lc_test, dtype=np.float32), np.array(ctx_test, dtype=np.float32),
        test_names, test_is_anom, test_lens, common_classes,
    )


def run_gru_demo():
    tf.keras.utils.set_random_seed(42)
    print("=" * 84)
    print("         DEEP_ASTRO: DUAL-STREAM GRU + MCIF + ADAPTIVE END-TO-END DEMO")
    print("=" * 84)

    (
        lc_train, ctx_train, y_train_onehot, y_train_labels,
        lc_val, ctx_val, y_val_labels,
        lc_test, ctx_test, test_names, test_is_anom, test_lens, common_classes,
    ) = build_demo_dataset(n_train_per_class=35, n_val_per_class=10, random_state=42)

    print(f"Train light curves: {lc_train.shape} (zero-padded to 656 timesteps)")
    print(f"Train context:      {ctx_train.shape} ([extinction, redshift])")
    print(f"Known classes:      {common_classes}")

    # 1. Build Dual-Stream GRU Encoder
    print("\n[Step 1] Building Dual-Stream GRU Classifier...")
    model = build_transient_encoder(
        latent_dim=100,
        ntimesteps=656,
        num_classes=len(common_classes),
        contextual_features=2,
        n_features=4,
    )
    print(f"  Total trainable parameters: {model.count_params():,}")

    # 2. Train GRU on Common Classes
    print("\n[Step 2] Training GRU for 10 epochs on common transient shapes...")
    history = model.fit(
        [lc_train, ctx_train],
        y_train_onehot,
        epochs=10,
        batch_size=16,
        shuffle=True,
        verbose=1,
    )
    final_acc = history.history["accuracy"][-1]
    print(f"  Final training accuracy: {final_acc * 100:.1f}%")

    # 3. Discard Softmax Head & Extract 100D Latent Space
    print("\n[Step 3] Extracting 100D penultimate latent representations...")
    z_train = extract_latent_space(model, [lc_train, ctx_train])
    z_val = extract_latent_space(model, [lc_val, ctx_val])
    z_test = extract_latent_space(model, [lc_test, ctx_test])
    print(f"  Raw sequence shape: {lc_train.shape} ({656 * 4:,} floats/event)")
    print(f"  Compressed latent:  {z_train.shape} (100 floats/event -> 96.2% reduction)")

    # 4. Fit MCIF on train embeddings & calibrate ClusterStandardizedScorer on validation embeddings
    print("\n[Step 4] Fitting MCIF (train) + ClusterStandardizedScorer (val) on 100D embeddings...")
    mcif = MultiClassIsolationForest(n_estimators=150, random_state=42)
    mcif.fit(z_train, y_train_labels)

    val_matrix = mcif.score_per_class(z_val)
    scorer = ClusterStandardizedScorer()
    scorer.fit(val_matrix, y_val_labels, mcif.classes)

    val_z = scorer.score(val_matrix)
    threshold_z = float(np.percentile(val_z, 95))

    test_raw = mcif.score(z_test)
    test_z = scorer.score(mcif.score_per_class(z_test))

    # 5. Display Per-Event Results
    print(f"\n[Step 5] Scoring Test Events (Adaptive z-score Threshold = {threshold_z:+.2f}):")
    print("-" * 84)
    print(f"{'Event':<6} | {'True Class':<29} | {'Steps':<5} | {'MCIF Raw':<9} | {'Adapt z':<8} | {'Decision'}")
    print("-" * 84)
    for i, (name, is_anom, seq_len, raw_s, z_s) in enumerate(
        zip(test_names, test_is_anom, test_lens, test_raw, test_z), start=1
    ):
        status = "FLAGGED ANOMALY" if z_s > threshold_z else "Normal"
        tag = "[Unseen]" if is_anom else "[Known] "
        print(f"#{i:<5} | {tag} {name:<20} | {seq_len:<5} | {raw_s:>+8.4f}  | {z_s:>+7.2f}  | {status}")
    print("=" * 84)


if __name__ == "__main__":
    run_gru_demo()
