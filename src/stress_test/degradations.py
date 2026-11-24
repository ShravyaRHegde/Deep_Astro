"""Observational noise, missing cadence, and redshift degradation simulators."""

import numpy as np


def simulate_cadence_dropouts(
    lc_matrix: np.ndarray,
    dropout_fraction: float = 0.4,
    random_state: int = 42,
) -> np.ndarray:
    """Simulates missed observations due to bad weather or survey cadence interruptions.
    
    Args:
        lc_matrix: Array of shape (N, N_timesteps, 4) with entries [flux, error, time, wavelength].
        dropout_fraction: Fraction of non-zero observations to drop (0.0 to 0.9).
        random_state: Random seed for reproducibility.
        
    Returns:
        Degraded light curve array with masked observations.
    """
    rng = np.random.RandomState(random_state)
    degraded = lc_matrix.copy()
    n_samples, n_timesteps, n_features = degraded.shape

    for i in range(n_samples):
        # Identify active (non-zero) timesteps
        active_indices = np.where(np.abs(degraded[i, :, 0]) > 1e-6)[0]
        if len(active_indices) > 2:
            n_drop = int(len(active_indices) * dropout_fraction)
            drop_idx = rng.choice(active_indices, size=n_drop, replace=False)
            degraded[i, drop_idx, :] = 0.0

    return degraded


def simulate_photometric_noise(
    lc_matrix: np.ndarray,
    noise_multiplier: float = 2.0,
    random_state: int = 42,
) -> np.ndarray:
    """Escalates photometric measurement errors and injects proportional Gaussian noise into flux values.
    
    Args:
        lc_matrix: Array of shape (N, N_timesteps, 4).
        noise_multiplier: Factor by which to escalate uncertainty (e.g., 1.5, 2.0, 3.0).
        random_state: Random seed.
        
    Returns:
        Noisy light curve array.
    """
    rng = np.random.RandomState(random_state)
    noisy = lc_matrix.copy()

    # Column 0 is flux, Column 1 is uncertainty
    flux = noisy[:, :, 0]
    error = noisy[:, :, 1]
    active_mask = np.abs(flux) > 1e-6

    # Scale error column
    noisy[:, :, 1] = error * noise_multiplier

    # Perturb flux by added Gaussian scatter
    added_noise = rng.normal(loc=0.0, scale=error * (noise_multiplier - 1.0))
    noisy[:, :, 0] = np.where(active_mask, flux + added_noise, 0.0)

    return noisy


def simulate_redshift_degradation(
    context_matrix: np.ndarray,
    mode: str = "photo_z",
    scatter_factor: float = 0.05,
    random_state: int = 42,
) -> np.ndarray:
    """Simulates real-world alert scenarios where spectroscopic host redshift is unavailable.
    
    Args:
        context_matrix: Array of shape (N, 2) containing [Milky Way extinction, host redshift].
        mode: 'exact' (no change), 'photo_z' (photometric redshift with scatter), or 'no_z' (zeroed).
        scatter_factor: Photometric redshift error sigma_z / (1 + z).
        random_state: Random seed.
        
    Returns:
        Degraded context array.
    """
    rng = np.random.RandomState(random_state)
    degraded = context_matrix.copy()

    if mode == "exact":
        return degraded
    elif mode == "no_z":
        degraded[:, 1] = 0.0
        return degraded
    elif mode == "photo_z":
        z_true = degraded[:, 1]
        sigma = scatter_factor * (1.0 + z_true)
        z_noisy = z_true + rng.normal(0.0, sigma)
        degraded[:, 1] = np.maximum(0.0, z_noisy)
        return degraded
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from 'exact', 'photo_z', 'no_z'.")
