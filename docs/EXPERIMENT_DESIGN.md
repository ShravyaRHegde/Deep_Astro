# Experimental Design & Evaluation Protocol

## 1. Experimental Goals

This project investigates three core research directions:
1. **Dynamic Thresholding**: Evaluating whether class-conditioned thresholding improves recovery of challenging rare transients without inflating false positives.
2. **Algorithm Comparison**: Benchmarking MCIF against Global Isolation Forest, Global LOF, and Multi-Class LOF (MCLOF).
3. **Observational Robustness**: Stress-testing the anomaly detection pipeline against degraded astronomical alert conditions.

---

## 2. Benchmark Evaluation Protocol

### Realistic Population Sampling
In natural sky surveys, anomalous events are extremely rare compared to standard transients. The evaluation protocol mimics natural frequencies:
- **Common In-Distribution Transients**: 12,040 objects (SNIa, SNII, SLSN-I, AGN, etc.)
- **Anomalous Out-of-Distribution Transients**: 54 objects (Kilonovae, PISN, ILOT, CaRT, uLens)
- **Class Ratio**: ~220 : 1
- **Cross-Validation**: 50 independent stratified evaluation splits to compute mean and standard deviation of recovery rates.

### Primary Metrics
- **AUROC**: Measures overall class separability across all possible decision boundaries.
- **AUCPR**: Evaluates positive prediction purity under extreme class imbalance.
- **Recall @ Top-2000 Follow-Up**: The percentage of true anomalies detected when following up the top 2000 highest-ranked candidates (~15% of all alerts), mimicking telescope scheduling constraints.
- **Per-Class Anomaly Recall**: Breakdown across distinct anomaly categories, with specific focus on subtle events (e.g., CaRT).

---

## 3. Robustness Stress-Testing Suite

### 1. Cadence Dropouts
Simulates missing observations caused by telescope downtime, weather, and lunar cycles:
- Random dropout fractions: $20\%$, $40\%$, $60\%$, and $80\%$ of measured epochs removed.

### 2. Photometric Noise Escalation
Simulates poor atmospheric seeing or faint detections near survey limiting magnitude:
$$f_{\text{perturbed}} = f + \mathcal{N}(0, \alpha \cdot \epsilon) \quad \text{for } \alpha \in \{1.0, 1.5, 2.0, 3.0\}$$

### 3. Redshift Degradation
Evaluates performance when spectroscopic host redshift is unavailable:
- Mode 1: Exact spectroscopic host redshift ($z_{\text{spec}}$)
- Mode 2: Photometric redshift with uncertainty ($\sigma_z = 0.05(1+z)$)
- Mode 3: No redshift information available
