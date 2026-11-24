# Dataset Splitting & Experimental Protocol Guide

## 1. Executive Summary

When experimenting with multiple machine learning algorithms (MCIF, Global Isolation Forest, Global LOF, Multi-Class LOF), you **do not need to train on the entire massive raw dataset** to achieve high accuracy. 

In fact, training anomaly detectors on massive datasets often **degrades detection accuracy** due to statistical "swamping." 

This guide explains how to split and subsample the dataset correctly so that:
1. Your laptop runs fast and cool without memory spikes (< 250 MB RAM).
2. The comparison between algorithms is 100% scientifically fair and rigorous.
3. The evaluation accuracy matches or exceeds published academic standards.

---

## 2. The Golden Rule of Model Comparison

> **Rule:** Every algorithm must be evaluated on the **exact same standardized test set**.

```
                           FULL DATASET
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
  Common Classes (12)                             Rare Anomalies (5)
  (SNIa, SNII, SLSN, AGN, etc.)                   (CaRT, KN, PISN, ILOT, uLens)
        │                                               │
   ┌────┴──────────────────────────┐                    │
   ▼                               ▼                    │
Train Split (Common Only)    Val Split (Common Only)    │
~3,000 - 5,000 samples       ~1,000 samples             │
   │                               │                    │
   │ [Fit Detectors]               │ [Calibrate Thresh] │
   │ • Global IF                   │ • Compute μc, σc   │
   │ • Global LOF                  │ • Fit Quantiles    │
   │ • MCIF                        │                    │
   │ • MCLOF                       │                    │
   └───────────────┬───────────────┘                    │
                   │                                    │
                   └───────────────┬────────────────────┘
                                   ▼
                       STANDARDIZED TEST SPLIT
                       (Evaluated on ALL 4 Algos)
                       • 12,040 Common Transients
                       • 54 Nature-Matched Anomalies
                       ─────────────────────────────
                       Total: 12,094 events (~8 MB 100D)
```

1. **Training Split (Common Classes Only)**:
   - Size: **3,000 to 5,000 samples** (stratified across the 12 common classes, ~250–400 per class).
   - Used exclusively to establish the "normal baseline" for MCIF, Global IF, Global LOF, and MCLOF.
2. **Validation Split (Common Classes Only)**:
   - Size: **~1,000 samples**.
   - Used to calibrate the class-dependent thresholds (mean $\mu_c$, std $\sigma_c$, and quantile cutoffs for Objective 1).
3. **Standardized Test Split (Common + Unseen Anomalies)**:
   - Size: **12,040 common transients + 54 anomalous events** (the realistic 220:1 ratio from Kessler et al. / PLAsTiCC).
   - **Crucial**: Pass this identical test split through every algorithm. Because the test split is identical, any difference in AUROC or Top-2000 Recall is strictly due to algorithmic superiority, not data luck.

---

## 3. Why Subsampling Training Data Preserves (and Enhances) Accuracy

### 3.1 The Isolation Forest Paradox (Swamping & Masking)
Isolation Forest works by recursively placing random axis-aligned splits until points are isolated. 
* If you train on **100,000 points**: normal points pack densely around outliers. Outliers require many splits to isolate (*swamping*), making them look like normal points.
* If you train on **subsamples of 256 to 1,000 points**: outliers stand isolated in space and are partitioned within 2 to 4 splits, maximizing discrimination.
* The original creators of Isolation Forest (*Liu et al., 2008*) proved that subsampling improves anomaly detection accuracy over using massive datasets.

### 3.2 Computational Complexity of Local Outlier Factor (LOF)
* Standard LOF performs $k$-nearest neighbor distance queries, with complexity scaling as $O(N^2)$ or $O(N \log N)$.
* On 100,000 points, LOF would freeze your laptop and take hours.
* On **3,000 to 5,000 points**, LOF runs in **under 15 seconds**, yielding clean local density estimates.

---

## 4. The 2-Stage Latent Feature Caching Workflow

To prevent your laptop from repeatedly crunching heavy raw light curves ($656 \times 4$ floats per event), follow this two-stage caching pipeline:

```
[Raw Light Curves] ──(Run ONCE in small batches)──► [GRU Encoder] ──► [Save latent_features.npz (~8 MB)]
                                                                               │
                                                                               ├──► Test Algo 1: Global IF (2 sec)
                                                                               ├──► Test Algo 2: MCIF (4 sec)
                                                                               ├──► Test Algo 3: Global LOF (10 sec)
                                                                               └──► Test Algo 4: MCLOF (12 sec)
```

1. **Step 1 (Run Once)**: Load raw light curves in mini-batches of 128 through the neural network encoder to extract 100D vectors ($z_s$). Save the vectors to `data/latent/embeddings.npz`.
2. **Step 2 (Instant Experimentation)**: Load `embeddings.npz`. Now, every anomaly detection model, threshold experiment, and cross-validation loop runs in seconds in RAM without touching the raw light curve files again.
