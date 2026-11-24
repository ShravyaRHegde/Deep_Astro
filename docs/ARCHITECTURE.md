# Technical Architecture Specification

## 1. System Overview

Deep_Astro implements a two-stage anomaly detection framework for astronomical time series:

1. **Stage 1: Supervised Sequential Representation Learning**
   - Encodes irregularly sampled, multi-band optical light curves into an organized latent feature space where common astronomical transients cluster by physical class.
2. **Stage 2: Multi-Class Anomaly Detection & Adaptive Scoring**
   - Models the boundary of each known class independently to detect unexpected cosmic phenomena without falsely classifying well-separated common classes as outliers.

---

## 2. Neural Network Encoder Pipeline

### Input Features
For each transient event $s$, observations are represented sequentially without requiring interpolation:
$$X_{sj} = [f_{sj}, \epsilon_{sj}, t_{sj}, \lambda_p]$$
- $f_{sj}$: Scaled photometric flux ($f / 500$)
- $\epsilon_{sj}$: Scaled measurement uncertainty ($\epsilon / 500$)
- $t_{sj}$: Normalized observer time relative to trigger
- $\lambda_p$: Central wavelength of the filter passband ($g=0.4827\,\mu\text{m}, r=0.6233\,\mu\text{m}$)

### Network Layers
- **Masking Layer**: Masks zero-padded timesteps up to maximum sequence length $N_T = 656$.
- **Recurrent Layers**: 2 sequential Gated Recurrent Unit (GRU) layers with 100 hidden units each.
- **Contextual Stream**: Processes host galaxy redshift and Galactic extinction via a Dense layer (10 units, ReLU).
- **Concatenation & Penultimate Layer**: Combines sequential and contextual features into a 100-neuron Dense layer with ReLU activation ($z_s = E(X_s)$).
- **Classification Output**: 12-neuron Dense layer with softmax activation, trained using class-weighted categorical cross-entropy.

---

## 3. Anomaly Detection Algorithms

### 3.1 Multi-Class Isolation Forest (MCIF)
Instead of fitting one global isolation forest across all classes, MCIF trains separate isolation forests $I_c$ for each known class $c \in \{1, \dots, C\}$ on the 100D latent embeddings:
$$A_{\text{MCIF}}(z_s) = \min_{c=1 \dots C} \left( -I_c(z_s) \right)$$
- If an object is consistent with any known class, its minimum anomaly score is small.
- If an event is out-of-distribution across all known classes, $A_{\text{MCIF}}(z_s)$ is elevated.

### 3.2 Multi-Class Local Outlier Factor (MCLOF)
To address varying cluster densities in the latent space, MCLOF evaluates local reachability density against class-specific nearest neighbors:
$$A_{\text{MCLOF}}(z_s) = \min_{c=1 \dots C} \text{LOF}_c(z_s)$$

---

## 4. Adaptive & Dynamic Thresholding

A static global threshold $\tau$ often misses low-amplitude or subtle rare classes (e.g., Calcium-Rich Transients) that share structural features with standard supernovae. 

### Methods Implemented
1. **Cluster-Standardized Anomaly Scoring**:
   Standardizes scores against the mean ($\mu_c$) and standard deviation ($\sigma_c$) of the closest class cluster:
   $$A_{\text{std}}(z_s) = \frac{-I_{c^*}(z_s) - \mu_{c^*}}{\sigma_{c^*}} \quad \text{where } c^* = \arg\min_c (-I_c(z_s))$$
2. **Empirical Quantile Normalization**:
   Maps scores into class-specific percentile ranks ($[0, 1]$), ensuring uniform sensitivity across dense and diffuse clusters.
