# Data Strategy & Compute Optimization

## 1. Context: Laptop Execution & Memory Constraints

A major concern when processing astronomical alert time series is computational demand. Astronomical data sets frequently contain tens of thousands of light curves, which can easily overwhelm personal computer RAM and cause freezes if loaded entirely into memory as raw multi-passband time series.

This document outlines how Deep_Astro handles large-scale transient data efficiently, maintaining high detection fidelity while running comfortably on a standard laptop.

---

## 2. Will Subsampling or Splitting Hurt Accuracy?

### Theoretical Foundation
For anomaly detection—specifically Isolation Forests and Local Outlier Factor (LOF)—**massive datasets do not automatically yield better anomaly boundaries, and can actually degrade detection accuracy**:

1. **Swamping and Masking in Isolation Forests**:
   - *Swamping* occurs when normal points cluster closely around an outlier, artificially lengthening the tree path length and making an anomaly appear normal.
   - *Masking* occurs when multiple outliers cluster near each other, partitioning together and disguising their anomalous nature.
   - Foundational anomaly detection literature demonstrates that Isolation Forests perform optimally with **subsamples of 256 to 512 instances per tree**.

2. **Latent Representation Compression**:
   - Raw light curves consist of sequential matrices ($N_{\text{timesteps}} \times 4$ dimensions).
   - Once mapped by the recurrent neural network encoder, each event is represented by a compact **100-dimensional vector**.
   - A dataset of 20,000 objects in a 100-dimensional float32 representation requires only:
     $$20{,}000 \times 100 \times 4\text{ bytes} \approx 8\text{ MB of RAM}$$
   - Therefore, the anomaly detection stage operates on lightweight vectors rather than heavy time-series arrays.

3. **Stratified Representative Sampling**:
   - Rather than loading arbitrary data chunks, we use stratified sampling that mirrors the true astronomical event distribution:
     - Common transients: ~12,040 objects
     - Rare anomalous events: ~54 objects (~220:1 ratio)
   - Running evaluation across repeated independent folds (e.g., 50-fold cross-validation) provides high statistical confidence without requiring massive memory allocations.

---

## 3. Recommended Workflow on Personal Hardware

1. **Feature Extraction in Batches**:
   When encoding raw light curves through the deep neural network, pass data in batches of 128 to 256 using Python generators. This keeps memory usage below 500 MB.
2. **Latent Vector Caching**:
   Save extracted 100-dimensional vectors to compact compressed files (`.npz` or `.parquet`). Subsequent anomaly detection and thresholding experiments execute in seconds without loading raw time series again.
3. **Multi-Model Parallelism**:
   MCIF trains separate Isolation Forests per class. Because each class subset is relatively small, fitting all 12 trees takes under 15 seconds on a standard multi-core laptop CPU.
