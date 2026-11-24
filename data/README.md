# Data Directory Guide

This folder manages astronomical datasets used in the Deep_Astro pipeline.

## 1. Directory Structure

```text
data/
├── README.md               # Data setup and acquisition guidelines
├── raw/                    # Raw simulation light-curve files (.csv / .tar.gz)
├── processed/              # Formatted numpy time series arrays (X_data, y_data)
└── latent/                 # Pre-computed 100D embeddings for fast benchmarking
```

## 2. Dataset Sources

1. **ZTF / PLAsTiCC Simulations**:
   - Simulated Zwicky Transient Facility light curves matching the observing logs of the ZTF survey.
   - 17 total classes: 12 common classes used for training, 5 rare anomalous classes reserved exclusively for evaluation.
2. **Tabular Feature Baseline**:
   - Tabular light-curve features extracted from transient archives, used for rapid model comparison without running sequential deep neural networks.

## 3. Storage & Memory Note

Large data files (`.csv`, `.h5`, `.tar.gz`) are excluded from Git tracking via `.gitignore`. Store raw light curve dumps locally in `data/raw/`.
