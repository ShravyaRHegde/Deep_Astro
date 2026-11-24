"""
src/stress_test/degradations.py
Stage 5 - Observational Robustness & Stress-Testing Suite
"""

import numpy as np
from typing import List

FLUX_FEATURE_SLICE = slice(0, 90)
REDSHIFT_COL = 100


def cadence_dropout(X, drop_fraction, random_state=42):
    """Simulate cloudy nights by zeroing drop_fraction of features per sample."""
    if not 0.0 <= drop_fraction < 1.0:
        raise ValueError(f"drop_fraction must be in [0, 1), got {drop_fraction}")
    rng = np.random.default_rng(random_state)
    X_degraded = X.copy().astype(float)
    n_samples, n_features = X_degraded.shape
    n_drop = max(1, int(n_features * drop_fraction))
    for i in range(n_samples):
        drop_cols = rng.choice(n_features, size=n_drop, replace=False)
        X_degraded[i, drop_cols] = 0.0
    return X_degraded


def cadence_dropout_sweep(X, drop_levels=[0.20, 0.40, 0.60, 0.80], random_state=42):
    """Apply cadence_dropout at multiple drop levels."""
    return {level: cadence_dropout(X, level, random_state=random_state) for level in drop_levels}


def add_photometric_noise(X, noise_scale, random_state=42):
    """Add scaled Gaussian noise to flux/magnitude features."""
    if noise_scale < 1.0:
        raise ValueError(f"noise_scale must be >= 1.0, got {noise_scale}")
    rng = np.random.default_rng(random_state)
    X_noisy = X.copy().astype(float)
    flux_cols = X_noisy[:, FLUX_FEATURE_SLICE]
    feature_stds = np.std(flux_cols, axis=0) + 1e-8
    noise_amplitude = (noise_scale - 1.0) * feature_stds
    noise = rng.normal(loc=0.0, scale=noise_amplitude, size=flux_cols.shape)
    X_noisy[:, FLUX_FEATURE_SLICE] = flux_cols + noise
    return X_noisy


def photometric_noise_sweep(X, noise_levels=[1.5, 2.0, 3.0], random_state=42):
    """Apply noise at multiple scale levels."""
    return {scale: add_photometric_noise(X, scale, random_state=random_state) for scale in noise_levels}


def remove_redshift(X, fraction=1.0, noisy=False, noise_std=0.3, random_state=42):
    """Simulate missing spectroscopic host redshift."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    rng = np.random.default_rng(random_state)
    X_masked = X.copy().astype(float)
    n_samples = X_masked.shape[0]
    n_mask = int(n_samples * fraction)
    if n_mask == 0:
        return X_masked
    mask_indices = rng.choice(n_samples, size=n_mask, replace=False)
    if noisy:
        z_values = X_masked[mask_indices, REDSHIFT_COL]
        X_masked[mask_indices, REDSHIFT_COL] = z_values + rng.normal(loc=0.0, scale=noise_std, size=n_mask)
    else:
        X_masked[mask_indices, REDSHIFT_COL] = 0.0
    return X_masked


def redshift_sweep(X, fractions=[0.25, 0.50, 0.75, 1.00], random_state=42):
    """Apply redshift removal at multiple missing fractions."""
    return {frac: remove_redshift(X, fraction=frac, noisy=False, random_state=random_state) for frac in fractions}
