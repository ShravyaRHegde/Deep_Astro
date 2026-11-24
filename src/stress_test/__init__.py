"""Observational stress-testing package for Deep_Astro (Objective 3)."""
from .degradations import (
    simulate_cadence_dropouts,
    simulate_photometric_noise,
    simulate_redshift_degradation,
)

__all__ = [
    "simulate_cadence_dropouts",
    "simulate_photometric_noise",
    "simulate_redshift_degradation",
]
