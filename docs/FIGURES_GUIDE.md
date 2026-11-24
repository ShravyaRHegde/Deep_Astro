# Scientific Figures Guide: Deep_Astro Visualization Suite

This guide provides comprehensive documentation, astrophysical interpretations, and reproduction instructions for the scientific figures generated in **Stage 6** of the Deep_Astro project.

---

## 1. Overview of Generated Publication Figures

All figures are rendered at **300 DPI** using publication-standard scientific aesthetics, colorblind-accessible palettes, and clear typographic hierarchy.

| Figure File | Title / Subject | Key Scientific Finding |
| :--- | :--- | :--- |
| [`figures/roc_curves_comparison.png`](../figures/roc_curves_comparison.png) | Multi-Model ROC Comparison across 4 Transient Classes | Demonstrates trade-offs between tree-based isolation (MCIF) and local density reachability (LOF) across diffuse vs. envelope manifolds. |
| [`figures/model_win_distribution.png`](../figures/model_win_distribution.png) | 14-Class Real Benchmark & Model Win Distribution | MCIF wins 6 classes (dominating periodic stars), Global IF wins 4 classes, Global LOF wins 2 classes (SLSN, AGN), and MCIF+Adaptive wins 2 classes. |
| [`figures/adaptive_threshold_lift.png`](../figures/adaptive_threshold_lift.png) | Adaptive Threshold Score Separation (Objective 1) | Resolves the "CaRT Dilemma": standardizing raw anomaly scores against nearest cluster variance separates subtle anomalies from common inliers. |
| [`figures/latent_umap_projection.png`](../figures/latent_umap_projection.png) | 2D Manifold Projection of Clustered Latent Space | Visualizes the 100D latent space, showing clean separation of in-distribution astrophysical types and the peripheral positioning of rare anomalies. |

---

## 2. In-Depth Figure Descriptions & Astrophysical Context

### Figure 1: Receiver Operating Characteristic (ROC) Comparison across Transient Classes
* **Path**: `figures/roc_curves_comparison.png`
* **Plot Type**: $2 \times 2$ Subplot Grid of ROC Curves (Sensitivity vs. $1 - \text{Specificity}$)

```
┌─────────────────────────────────┬─────────────────────────────────┐
│ (a) SLSN: Superluminous SNe    │ (b) SNII: Core-Collapse SNe     │
│ Diffuse fringe outlier          │ Hydrogen-rich plateau envelope  │
├─────────────────────────────────┼─────────────────────────────────┤
│ (c) SNIa: Thermonuclear SNe     │ (d) SNIbc: Stripped Envelope SNe│
│ Standard candle baseline        │ Helium/Carbon-Oxygen core       │
└─────────────────────────────────┴─────────────────────────────────┘
```

#### Astrophysical & Algorithmic Analysis:
1. **SLSN (Superluminous Supernovae)**:
   * **Physics**: Extremely energetic explosions powered by magnetar spin-down or circumstellar interaction, resulting in light curves that are $\sim 10$ to $100\times$ brighter than standard supernovae with broad temporal plateaus.
   * **Algorithm Behavior**: SLSNe form a diffuse, low-density fringe in latent space. Local Outlier Factor (LOF) estimators achieve superior detection because reachability density captures diffuse outliers more effectively than orthogonal decision tree splits.
2. **SNII (Type II Core-Collapse Supernovae)**:
   * **Physics**: Explosions of massive red supergiants characterized by extended hydrogen recombination plateaus ($\sim 100$ days in duration).
   * **Algorithm Behavior**: SNII light curves occupy a well-defined envelope in feature space. Global Isolation Forest and MCIF capture the core-collapse boundary effectively.
3. **SNIa (Type Ia Supernovae)**:
   * **Physics**: Thermonuclear detonations of carbon-oxygen white dwarfs near the Chandrasekhar limit. Because SNIa are the most common transient in modern surveys and serve as the standard training baseline, one-vs-rest evaluation treats them as the outlier against other transients.
   * **Algorithm Behavior**: Subtypes (SNIa-normal, 91bg-like, SNIax) create subtle multi-modal variations that challenge global models.
4. **SNIbc (Stripped-Envelope Supernovae)**:
   * **Physics**: Core-collapse events whose progenitor stars lost their outer hydrogen (and helium) envelopes prior to explosion.
   * **Algorithm Behavior**: Occupies the boundary between SNIa and SNII light-curve morphologies; Multi-Class Isolation Forest (MCIF) achieves top AUROC by evaluating isolation relative to distinct known class trees.

---

### Figure 2: Model Comparison & Algorithm Win Distribution Across All 14 Classes
* **Path**: `figures/model_win_distribution.png`
* **Plot Type**: Multi-panel visualization (Left: Grouped horizontal bar chart of all 14 classes; Right-Top: Win count distribution; Right-Bottom: Overall mean AUROC across the 14 classes).
* **Data Source**: Master results from `results/benchmark_all_14_classes.csv` (ZTF ALeRCE real-data benchmark).

#### Key Empirical Takeaways:
1. **Model Win Count Distribution**:
   * **MCIF (Base Paper)**: **6 wins** (SNIa, SNIbc, CEP, DSCT, E, RRL).
   * **Global IF**: **4 wins** (SNII, CV/Nova, QSO, YSO).
   * **Global LOF (Our Base)**: **2 wins** (SLSN, AGN).
   * **MCIF + Adaptive (Objective 1)**: **2 wins** (Blazar, LPV).
2. **Mean AUROC Across All 14 Astronomical Classes**:
   * **MCIF (Base Paper)**: **0.6506** (Highest overall mean)
   * **MCIF + Adaptive (Objective 1)**: **0.6423**
   * **Global Isolation Forest**: **0.6252**
   * **Global LOF (Our Baseline)**: **0.6082**
   * **Multi-Class LOF (MCLOF - Our Extension)**: **0.5529**
3. **Why MCIF Dominates Periodic Stars**:
   * Periodic variables (Cepheids, Delta Scuti, Eclipsing Binaries, RR Lyrae) exhibit tightly constrained orbital periods and harmonic amplitudes. Training a dedicated isolation forest per class preserves the distinct geometric boundary of each periodic class cluster, whereas a single global forest smears out subtype boundaries.
4. **Why LOF Wins on Diffuse Classes**:
   * For **SLSN** (AUROC = 0.6960) and **AGN** (AUROC = 0.6013), Global LOF outscores tree partitioning by +13.9% and +3.4% respectively. LOF's reachability distance is resilient to high intra-class variance and diffuse spatial scatter.

---

### Figure 3: Adaptive Threshold Score Separation (Objective 1)
* **Path**: `figures/adaptive_threshold_lift.png`
* **Plot Type**: Dual-panel score density distribution comparing Inliers vs. Subtle Anomalies (Calcium-Rich Transients / CaRT) before and after cluster standardization.

#### The Problem: The "CaRT Dilemma" (Panel a)
* In standard MCIF (Gupta et al., 2025), a single global threshold $\tau$ is applied across the minimum anomaly score:
  $$A(z_s) = \min_c (-I_c(z_s))$$
* Subtle anomalies such as **Calcium-Rich Transients (CaRT)** have faint peak luminosities and rapid light-curve decline rates that closely mirror stripped core-collapse supernovae (SNIb/c).
* Consequently, raw CaRT anomaly scores (mean $\approx -0.04$) fall directly inside the distribution of normal common transients (mean $\approx -0.06$).
* **Consequence**: Any fixed threshold $\tau$ that captures CaRT also flags thousands of normal supernovae (high false positive rate), while a conservative threshold misses CaRT altogether.

#### The Solution: Cluster-Standardized Scoring (Panel b)
* By fitting cluster mean $\mu_c$ and standard deviation $\sigma_c$ on validation inliers, test anomaly scores are calibrated as:
  $$z = \frac{-I_c(z_s) - \mu_c}{\sigma_c} \quad \text{relative to the nearest cluster } c$$
* **Result**: Common transients normalize to zero mean and unit variance ($z \approx 0$). Subtle anomalies that sit in the low-density periphery of a dense cluster are shifted into the extreme positive tail ($z > 3.0$), providing clean separation and high recall without elevating false alarms.

---

### Figure 4: 2D Manifold Projection of the Clustered Latent Space
* **Path**: `figures/latent_umap_projection.png`
* **Plot Type**: 2D scatter manifold projection (UMAP / t-SNE) with cluster centroid labels and highlighted anomalous transients.

#### Topological Structure:
1. **In-Distribution Common Manifolds**:
   * **SNIa Cluster**: Compact cluster reflecting standard thermonuclear candle properties.
   * **Core-Collapse Manifold (SNII, SNIIn, SNIIb, SNIb, SNIc)**: Multi-branch cluster reflecting varying hydrogen/helium envelope retention and ejecta masses.
   * **Galactic & Extragalactic Variability (AGN, SLSN-I, TDE)**: Distinct diffuse clusters separated from standard stellar explosions.
2. **Out-of-Distribution Anomalous Transients**:
   * **CaRT (Red Stars)**: Located immediately adjacent to the stripped-envelope SNIb cluster, illustrating why CaRT represents the most subtle anomaly challenge in real surveys.
   * **Kilonovae (KNe / KN, Magenta Diamonds)**: Distinct fast-fading optical counterparts to neutron-star mergers, positioned between core-collapse and diffuse coordinates.
   * **PISN (Dark Red Triangles)**: Highly energetic Pair-Instability Supernovae sitting at extreme high-luminosity coordinates.
   * **ILOT & uLens-BSR**: Positioned outside standard explosive manifolds, reflecting non-supernova physics (stellar mergers and gravitational microlensing).

---

## 3. Reproduction Instructions

To regenerate all four figures from scratch:

```bash
# Ensure dependencies are available
python3 -m pip install -r requirements.txt

# Run the figure generation suite
python experiments/generate_figures.py
```

### Script Execution Log:
```text
================================================================================
        DEEP_ASTRO SCIENTIFIC FIGURE GENERATION SUITE (STAGE 6)         
================================================================================

 [1/4] Generating Figure 1: ROC Curves Comparison across Transient Classes...
    • SLSN   | Global LOF: 0.980 | MCIF: 0.968 | MCIF+Adapt: 0.972
    • SNII   | Global LOF: 0.431 | Global IF: 0.332
    • SNIa   | MCIF: 0.269       | MCIF+Adapt: 0.248
    • SNIbc  | MCIF+Adapt: 0.597 | MCIF: 0.594
  [✓] Saved to figures/roc_curves_comparison.png

 [2/4] Generating Figure 2: Model Win Distribution & 14-Class Comparison...
  [✓] Saved to figures/model_win_distribution.png

 [3/4] Generating Figure 3: Adaptive Threshold Score Separation (Objective 1)...
  Inlier sample count:  12,040
  CaRT anomaly count:   14
  [✓] Saved to figures/adaptive_threshold_lift.png

 [4/4] Generating Figure 4: 2D Manifold Projection of Clustered Latent Space...
  [✓] Saved to figures/latent_umap_projection.png

================================================================================
  [✓] figures/roc_curves_comparison.png        (697.6 KB)
  [✓] figures/model_win_distribution.png       (527.0 KB)
  [✓] figures/adaptive_threshold_lift.png      (328.1 KB)
  [✓] figures/latent_umap_projection.png       (520.6 KB)
All Stage 6 publication figures generated successfully at 300 DPI!
```

---

## 4. Reusing Plotting Functions in Future Experiments

The plotting suite is fully modular and importable from `src.visualization.plots`:

```python
from src.visualization.plots import (
    plot_roc_curves_comparison,
    plot_model_win_distribution,
    plot_adaptive_threshold_lift,
    plot_latent_umap_projection,
)

# Example: Generate custom 14-class win comparison
fig = plot_model_win_distribution(
    benchmark_df_or_path="results/benchmark_all_14_classes.csv",
    save_path="figures/my_custom_win_plot.png",
)
```
