# Deep_Astro: Project Status & Engineering Log

**Current Release Milestone**: `v0.1-baseline`  
**Repository**: [https://github.com/ShravyaRHegde/Deep_Astro](https://github.com/ShravyaRHegde/Deep_Astro)  
**Lead Researcher**: Shravya R Hegde  

---

## 1. Executive Summary of Work Completed Till Now

This document provides a complete audit of all engineering and research tasks completed for the foundational baseline (`v0.1-baseline`) of the Deep_Astro project.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          DEEP_ASTRO WORK COMPLETED AUDIT                               │
├──────────────────────────┬─────────────────────────────┬───────────────────────────────┤
│ Foundational Layer       │ Data & Preprocessing Layer  │ Modeling & Benchmark Layer    │
├──────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ [✓] System Architecture  │ [✓] ZTF Dataset Download    │ [✓] Global Isolation Forest   │
│ [✓] Corporate-grade Docs │ [✓] Noise Filtering         │ [✓] Multi-Class IF (MCIF)     │
│ [✓] Memory Safety Rules  │ [✓] Leakage-Free Imputer    │ [✓] Global LOF Density Base   │
│ [✓] Git Remote & Tags    │ [✓] RobustScaler IQ Bounds  │ [✓] Multi-Class LOF (MCLOF)   │
│ [✓] Dependency Stack     │ [✓] Stratified 5K Splits    │ [✓] Dynamic Thresholding (Obj1)│
│                          │ [✓] Parquet/NPZ Formats     │ [✓] 14-Class Real Benchmark   │
└──────────────────────────┴─────────────────────────────┴───────────────────────────────┘
```

---

## 2. Detailed Task Breakdown

### A. Environment & Architecture
- **Dependency Stack**: Installed and validated `tensorflow 2.21.0`, `keras 3.15.1`, `scikit-learn 1.8.0`, `astropy 8.0.3`, `pyerfa 2.0.1`, `astromcad 1.2.7`, `pandas 3.0.1`, `numpy 2.4.3`, `scipy 1.17.1`, and `gdown 6.5.2`.
- **System Architecture**: Published [docs/SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md), containing plain-English analogies, 6-layer architecture flowcharts, and real-time alert sequence diagrams.
- **Compute Safety Strategy**: Published [docs/DATA_STRATEGY.md](DATA_STRATEGY.md) and [docs/DATASET_SPLIT_GUIDE.md](DATASET_SPLIT_GUIDE.md) explaining why stratified subsampling preserves anomaly detection accuracy and keeps laptop RAM usage under 250 MB.

### B. Data Ingestion & Cleaning
- **Downloaded Dataset**: Acquired the official ALeRCE ZTF dataset (`93,171` training transients, `23,180` test transients).
- **Cleaning Script**: Implemented [`src/data_cleaning.py`](../src/data_cleaning.py).
  - Filtered out unclassified `Periodic-Other` contamination (`92,311` train / `22,978` test clean objects remaining).
  - Extracted 186 numerical astrophysical features ($g-r$ color evolution, variability amplitude, period harmonics).
  - Fitted median imputation exclusively on training data to prevent data leakage.
  - Applied `RobustScaler` with $[5.0, 95.0]$ interquartile range to normalize extreme astronomical variations.
  - Exported master clean Parquet files (`clean_train.parquet`, `clean_test.parquet`) and a lightweight balanced split (`stratified_train_5k.npz`, 4,363 samples, 2.6 MB).
  - Generated metadata summary in `data/processed/dataset_summary.json`.

### C. Algorithm Implementation & Benchmarking
- **Core Detectors ([`src/anomaly/detectors.py`](../src/anomaly/detectors.py))**:
  - `GlobalIsolationForest`: Single pooled forest baseline.
  - `MultiClassIsolationForest`: 12-class MCIF algorithm from Gupta et al. (2025).
  - `GlobalLOF`: Local Outlier Factor density baseline.
  - `MultiClassLOF`: Class-specific reachability density estimator.
- **Dynamic Thresholding ([`src/thresholding/adaptive.py`](../src/thresholding/adaptive.py))**:
  - `ClusterStandardizedScorer`: Standardizes raw anomaly scores against nearest cluster mean and variance ($z = \frac{\text{score} - \mu_c}{\sigma_c}$) to uncover subtle anomalies (Objective 1).
  - `EmpiricalQuantileNormalizer`: Percentile rank normalization.
- **Full 14-Class Benchmark ([`experiments/real_data_benchmark.py`](../experiments/real_data_benchmark.py))**:
  - Evaluated one-vs-rest anomaly detection across all 14 astronomical classes in Transients, Stochastic, and Periodic categories.
  - Master results saved in [`results/benchmark_all_14_classes.csv`](../results/benchmark_all_14_classes.csv).
  - Detailed findings documented in [docs/BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).

---

## 3. Master Benchmark Results (Table 2 Extension)

| Category | Outlier Class | Global IF | Global LOF | MCIF (Paper) | MCLOF (Ours) | MCIF+Adaptive | Top Method |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Transient** | **SLSN** | 0.6593 | **0.6960** | 0.5395 | 0.6217 | 0.5262 | **Global LOF** (+13.9% over MCIF) |
| | **SNII** | **0.7804** | 0.7448 | 0.7542 | 0.5652 | 0.6997 | **Global IF** |
| | **SNIa** | 0.4032 | 0.3973 | **0.4535** | 0.4378 | 0.4455 | **MCIF** |
| | **SNIbc** | 0.5969 | 0.6226 | **0.6466** | 0.6042 | 0.6444 | **MCIF** |
| **Stochastic**| **AGN** | 0.5178 | **0.6013** | 0.5675 | 0.5570 | 0.5319 | **Global LOF** (+3.4% over MCIF) |
| | **Blazar** | 0.6277 | 0.5822 | 0.6787 | 0.5097 | **0.6850** | **MCIF + Adaptive** |
| | **CV/Nova** | **0.9521** | 0.8323 | 0.9413 | 0.8304 | 0.9472 | **Global IF** |
| | **QSO** | **0.5744** | 0.4767 | 0.5418 | 0.4647 | 0.5172 | **Global IF** |
| | **YSO** | **0.8627** | 0.6708 | 0.7336 | 0.5411 | 0.7673 | **Global IF** |
| **Periodic** | **CEP** | 0.4418 | 0.4643 | **0.6344** | 0.4503 | 0.6334 | **MCIF** |
| | **DSCT** | 0.3961 | 0.4359 | **0.4731** | 0.4080 | 0.4692 | **MCIF** |
| | **E** | 0.4980 | 0.6013 | **0.6184** | 0.5144 | 0.6142 | **MCIF** |
| | **LPV** | 0.9835 | 0.8988 | 0.9825 | 0.8666 | **0.9840** | **MCIF + Adaptive** |
| | **RRL** | 0.4594 | 0.4909 | **0.5437** | 0.3696 | 0.5265 | **MCIF** |

---

## 4. Work Ready for Other Team Members

The `main` branch is stable and tagged as `v0.1-baseline`. Collaborators can create new branches to tackle the remaining roadmap:

```bash
# Workflow for Collaborators
git checkout main
git pull origin main
git checkout -b feature/<collaborator-name>-<task>
```

### Next Open Tasks:
1. **Task 1: Cadence & Noise Stress Testing (Objective 3)**:
   - Use [`src/stress_test/degradations.py`](../src/stress_test/degradations.py) to simulate 20%, 40%, 60%, 80% cadence dropouts and photometric noise scaling.
   - Plot the AUROC degradation curve across models.
2. **Task 2: Plot Generation & Presentation Figures**:
   - Generate ROC curves, AUCPR curves, and comparison bar charts comparing MCIF vs LOF vs Global IF for presentation slides.
