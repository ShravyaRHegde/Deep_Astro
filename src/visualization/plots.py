"""Scientific Visualization and Plotting Module for Deep_Astro.

Generates publication-quality figures for:
1. ROC curves across transient anomaly classes (SLSN, SNII, SNIa, SNIbc)
2. Benchmark model comparison and win distribution across all 14 classes
3. Adaptive threshold score separation before vs after cluster standardization
4. 2D manifold projection (UMAP / t-SNE) of the clustered latent space
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# -----------------------------------------------------------------------------
# Publication Styling Configuration
# -----------------------------------------------------------------------------
STYLE_CONFIG = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.edgecolor": "#2d3748",
    "axes.linewidth": 1.1,
    "axes.grid": True,
    "grid.color": "#e2e8f0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "xtick.color": "#2d3748",
    "ytick.color": "#2d3748",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#cbd5e1",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

# Color palettes for algorithms and categories
ALGO_COLORS = {
    "Global IF": "#2b6cb0",        # Deep Blue
    "Global LOF": "#d97706",       # Amber
    "MCIF": "#2e7d32",             # Emerald Green (Paper Baseline)
    "MCLOF": "#805ad5",            # Purple (Our Extension)
    "MCIF + Adaptive": "#e53e3e",  # Crimson Red (Objective 1)
    "MCIF+Adapt": "#e53e3e",
}

ALGO_STYLES = {
    "Global IF": {"ls": "--", "lw": 1.8},
    "Global LOF": {"ls": "-.", "lw": 1.8},
    "MCIF": {"ls": "-", "lw": 2.2},
    "MCLOF": {"ls": ":", "lw": 2.0},
    "MCIF + Adaptive": {"ls": "-", "lw": 2.5},
    "MCIF+Adapt": {"ls": "-", "lw": 2.5},
}


def apply_publication_style():
    """Applies clean publication-grade styling to Matplotlib."""
    plt.rcParams.update(STYLE_CONFIG)


# -----------------------------------------------------------------------------
# 1. ROC Curves Comparison (Figure 1)
# -----------------------------------------------------------------------------
def plot_roc_curves_comparison(
    roc_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, float]]],
    save_path: Optional[str] = "figures/roc_curves_comparison.png",
) -> plt.Figure:
    """Plots a 2x2 multi-panel ROC curve comparison for SLSN, SNII, SNIa, and SNIbc.
    
    Args:
        roc_data: Dictionary mapping class names to model curve dictionaries:
            {
                "SLSN": {
                    "Global IF": (fpr, tpr, auroc),
                    "Global LOF": (fpr, tpr, auroc),
                    ...
                },
                ...
            }
        save_path: Optional path to save the generated figure.
        
    Returns:
        The generated matplotlib Figure object.
    """
    apply_publication_style()

    target_classes = ["SLSN", "SNII", "SNIa", "SNIbc"]
    class_subtitles = {
        "SLSN": "Superluminous Supernovae (Diffuse Fringe Outlier)",
        "SNII": "Type II Supernovae (Core-Collapse Envelope)",
        "SNIa": "Type Ia Supernovae (Thermonuclear Baseline)",
        "SNIbc": "Type Ib/c Supernovae (Stripped Envelope SNe)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, cname in enumerate(target_classes):
        ax = axes[idx]
        ax.plot([0, 1], [0, 1], linestyle="--", color="#a0aec0", linewidth=1.3, label="Random Guess (AUC = 0.50)")

        if cname in roc_data:
            models_dict = roc_data[cname]
            for mname, (fpr, tpr, auc_val) in models_dict.items():
                col = ALGO_COLORS.get(mname, "#4a5568")
                style = ALGO_STYLES.get(mname, {"ls": "-", "lw": 1.8})
                label = f"{mname} (AUC = {auc_val:.3f})"
                ax.plot(fpr, tpr, label=label, color=col, linestyle=style["ls"], linewidth=style["lw"], alpha=0.9)

        ax.set_title(f"({chr(97 + idx)}) {cname}: {class_subtitles.get(cname, '')}", pad=10)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.set_aspect("equal")
        ax.legend(loc="lower right", frameon=True, fontsize=8.8)

        if idx in [2, 3]:
            ax.set_xlabel("False Positive Rate (1 - Specificity)")
        if idx in [0, 2]:
            ax.set_ylabel("True Positive Rate (Sensitivity)")

    fig.suptitle(
        "Receiver Operating Characteristic (ROC) Comparison across Transient Classes\n"
        "Multi-Class Isolation Forest vs. Local Outlier Factor vs. Dynamic Adaptive Thresholding",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        print(f"[✓] Saved ROC Curves Comparison to {save_path}")

    return fig


# -----------------------------------------------------------------------------
# 2. Model Win Distribution & 14-Class Comparison (Figure 2)
# -----------------------------------------------------------------------------
def plot_model_win_distribution(
    benchmark_df_or_path: Union[str, pd.DataFrame],
    save_path: Optional[str] = "figures/model_win_distribution.png",
) -> plt.Figure:
    """Plots full 14-class AUROC benchmark comparison and algorithm win distribution.
    
    Args:
        benchmark_df_or_path: Path to benchmark_all_14_classes.csv or DataFrame.
        save_path: Optional path to save the generated figure.
        
    Returns:
        The generated matplotlib Figure object.
    """
    apply_publication_style()

    if isinstance(benchmark_df_or_path, str):
        df = pd.read_csv(benchmark_df_or_path)
    else:
        df = benchmark_df_or_path.copy()

    # Model columns and display mapping
    model_cols = [
        ("AUROC_MCIF", "MCIF"),
        ("AUROC_MCIF_Adaptive", "MCIF + Adaptive"),
        ("AUROC_Global_IF", "Global IF"),
        ("AUROC_Global_LOF", "Global LOF"),
        ("AUROC_MCLOF", "MCLOF"),
    ]

    # Create 2-column layout: Left = Grouped Bar Chart, Right = Win & Mean AUROC Summary
    fig = plt.figure(figsize=(15, 8.5))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.7, 1.0], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.32)

    ax_main = fig.add_subplot(gs[:, 0])
    ax_wins = fig.add_subplot(gs[0, 1])
    ax_means = fig.add_subplot(gs[1, 1])

    # 1. Main Panel: Grouped Horizontal Bar Chart for all 14 Classes
    # Order classes by Category then Class Name
    category_order = {"Transient": 0, "Stochastic": 1, "Periodic": 2}
    df["Cat_Order"] = df["Category"].map(category_order)
    df = df.sort_values(by=["Cat_Order", "Anomaly_Class"], ascending=[True, True]).reset_index(drop=True)

    n_classes = len(df)
    y_pos = np.arange(n_classes)
    bar_height = 0.15

    for m_idx, (col_name, m_label) in enumerate(model_cols):
        offsets = y_pos + (m_idx - 2) * bar_height
        color = ALGO_COLORS.get(m_label, "#718096")
        values = df[col_name].values
        ax_main.barh(offsets, values, height=bar_height, label=m_label, color=color, alpha=0.92, edgecolor="white", linewidth=0.5)

    # Add category separation background spans
    cat_indices = {}
    for cat in ["Transient", "Stochastic", "Periodic"]:
        idxs = df[df["Category"] == cat].index
        if len(idxs) > 0:
            cat_indices[cat] = (idxs.min(), idxs.max())

    for cat, (start_idx, end_idx) in cat_indices.items():
        if cat == "Stochastic":
            ax_main.axhspan(start_idx - 0.45, end_idx + 0.45, color="#f7fafc", alpha=0.9, zorder=0)

    # Mark the winning model for each class with a badge/star
    for i, row in df.iterrows():
        best_auroc = row["Best_AUROC"]
        ax_main.text(
            best_auroc + 0.015,
            i,
            f"★ {row['Best_Model']}",
            va="center",
            ha="left",
            fontsize=7.8,
            fontweight="bold",
            color="#2d3748",
        )

    # Format Main Panel
    ax_main.set_yticks(y_pos)
    ylabels = [f"{row['Anomaly_Class']} ({row['Category']})" for _, row in df.iterrows()]
    ax_main.set_yticklabels(ylabels, fontsize=9.5)
    ax_main.invert_yaxis()
    ax_main.set_xlim(0.3, 1.08)
    ax_main.axvline(0.5, color="#e2e8f0", linestyle=":", linewidth=1.2)
    ax_main.set_xlabel("AUROC (One-vs-Rest Anomaly Separation)")
    ax_main.set_title("AUROC Comparison Across All 14 Astronomical Classes", pad=12)
    ax_main.legend(loc="lower right", frameon=True, fontsize=8.5, ncol=2)

    # 2. Right-Top Panel: Win Distribution
    win_counts = df["Best_Model"].value_counts()
    models_in_order = ["MCIF", "Global IF", "Global LOF", "MCIF+Adapt"]
    counts = [win_counts.get(m, 0) for m in models_in_order]
    display_names = ["MCIF (Paper)", "Global IF", "Global LOF (Ours)", "MCIF + Adapt (Obj 1)"]
    win_colors = [ALGO_COLORS.get(m, "#4a5568") for m in ["MCIF", "Global IF", "Global LOF", "MCIF + Adaptive"]]

    bars = ax_wins.bar(display_names, counts, color=win_colors, width=0.55, edgecolor="#2d3748", linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        ax_wins.text(bar.get_x() + bar.get_width() / 2, h + 0.15, f"{int(h)} Wins", ha="center", va="bottom", fontweight="bold", fontsize=9)

    ax_wins.set_ylim(0, max(counts) + 1.6)
    ax_wins.set_ylabel("Number of Winning Classes")
    ax_wins.set_title("Algorithm Win Distribution (14 Classes Total)", pad=10)
    ax_wins.tick_params(axis="x", rotation=12)

    # 3. Right-Bottom Panel: Overall Mean AUROC
    mean_aurocs = {
        "MCIF (Paper)": df["AUROC_MCIF"].mean(),
        "MCIF + Adapt": df["AUROC_MCIF_Adaptive"].mean(),
        "Global IF": df["AUROC_Global_IF"].mean(),
        "Global LOF": df["AUROC_Global_LOF"].mean(),
        "MCLOF (Ours)": df["AUROC_MCLOF"].mean(),
    }
    mean_names = list(mean_aurocs.keys())
    mean_vals = list(mean_aurocs.values())
    mean_colors = [ALGO_COLORS.get(m.replace(" (Paper)", "").replace(" (Ours)", ""), "#4a5568") for m in mean_names]

    bars2 = ax_means.barh(mean_names, mean_vals, color=mean_colors, height=0.55, edgecolor="#2d3748", linewidth=0.8)
    for bar in bars2:
        w = bar.get_width()
        ax_means.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", ha="left", va="center", fontweight="bold", fontsize=8.8)

    ax_means.set_xlim(0.4, 0.76)
    ax_means.set_xlabel("Mean Benchmark AUROC")
    ax_means.set_title("Mean AUROC Across All 14 Classes", pad=10)
    ax_means.invert_yaxis()

    fig.suptitle(
        "Deep_Astro Comprehensive 14-Class Real Data Benchmark (ZTF ALeRCE Light Curves)\n"
        "Multi-Class Isolation Forest vs. Baseline Density & Dynamic Thresholding Engines",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        print(f"[✓] Saved Model Win Distribution figure to {save_path}")

    return fig


# -----------------------------------------------------------------------------
# 3. Adaptive Threshold Score Separation (Figure 3)
# -----------------------------------------------------------------------------
def plot_adaptive_threshold_lift(
    raw_inliers: np.ndarray,
    raw_outliers: np.ndarray,
    std_inliers: np.ndarray,
    std_outliers: np.ndarray,
    outlier_name: str = "CaRT (Calcium-Rich Transient)",
    normal_name: str = "Common Transients (SNIa / SNII)",
    save_path: Optional[str] = "figures/adaptive_threshold_lift.png",
) -> plt.Figure:
    """Visualizes score separation before vs. after cluster standardization (Objective 1).
    
    Demonstrates resolution of the 'CaRT Dilemma':
    - Before: Subtle anomalies overlap heavily with normal common transient scores.
    - After: Standardizing relative to class cluster variance lifts subtle anomalies into the tail.
    
    Args:
        raw_inliers: Uncalibrated anomaly scores for common inliers.
        raw_outliers: Uncalibrated anomaly scores for subtle anomalies.
        std_inliers: Standardized anomaly scores (z-scores) for common inliers.
        std_outliers: Standardized anomaly scores (z-scores) for subtle anomalies.
        outlier_name: Descriptive label for the anomaly class.
        normal_name: Descriptive label for normal inliers.
        save_path: Optional path to save the generated figure.
        
    Returns:
        The generated matplotlib Figure object.
    """
    apply_publication_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=False)

    col_norm = "#2b6cb0"  # Blue for common inliers
    col_anom = "#e53e3e"  # Crimson red for subtle anomalies

    # 1. Left Panel: Before Cluster Standardization (Raw Minimum Score)
    ax1 = axes[0]
    bins_raw = np.linspace(
        min(raw_inliers.min(), raw_outliers.min()) - 0.05,
        max(raw_inliers.max(), raw_outliers.max()) + 0.05,
        45,
    )

    ax1.hist(raw_inliers, bins=bins_raw, density=True, alpha=0.55, color=col_norm, label=normal_name, edgecolor=col_norm)
    ax1.hist(raw_outliers, bins=bins_raw, density=True, alpha=0.75, color=col_anom, label=outlier_name, edgecolor=col_anom)

    # Static global threshold line (e.g., 95th percentile of inliers)
    tau_raw = np.percentile(raw_inliers, 95)
    ax1.axvline(tau_raw, color="#742a2a", linestyle="--", linewidth=2.0, label=f"Global Threshold $\\tau_{{95}}$ = {tau_raw:.3f}")

    # Annotate the Dilemma (Overlapping scores) neatly inside axes
    overlap_center = (np.mean(raw_inliers) + np.mean(raw_outliers)) / 2
    max_dens1 = max(ax1.get_ylim()[1], 10)
    ax1.annotate(
        "Severe Score Overlap\n('CaRT Dilemma')\nFixed threshold yields high FPs",
        xy=(overlap_center, max_dens1 * 0.25),
        xytext=(overlap_center + 0.045, max_dens1 * 0.65),
        arrowprops=dict(facecolor="#2d3748", arrowstyle="->", lw=1.2),
        ha="center",
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff5f5", ec="#feb2b2", lw=1.0),
    )

    ax1.set_title("(a) Before: Raw MCIF Minimum Score\n(Uncalibrated Cluster Densities)", pad=10)
    ax1.set_xlabel("Raw Anomaly Score: $\\min_c (-I_c(z_s))$")
    ax1.set_ylabel("Probability Density")
    ax1.legend(loc="upper left", frameon=True, fontsize=8.8)

    # 2. Right Panel: After Cluster Standardization (Cluster-Standardized z-Score)
    ax2 = axes[1]
    bins_std = np.linspace(
        min(std_inliers.min(), std_outliers.min()) - 0.5,
        max(std_inliers.max(), std_outliers.max()) + 0.5,
        45,
    )

    ax2.hist(std_inliers, bins=bins_std, density=True, alpha=0.55, color=col_norm, label=normal_name, edgecolor=col_norm)
    ax2.hist(std_outliers, bins=bins_std, density=True, alpha=0.75, color=col_anom, label=outlier_name, edgecolor=col_anom)

    # Standardized threshold line (e.g. z = 3.0 or 95th percentile)
    tau_std = np.percentile(std_inliers, 95)
    ax2.axvline(tau_std, color="#22543d", linestyle="--", linewidth=2.0, label=f"Calibrated Threshold $z_{{95}}$ = {tau_std:.2f}")

    # Annotate Clean Separation neatly without colliding with legend
    anom_mean_std = np.mean(std_outliers)
    max_dens2 = ax2.get_ylim()[1]
    ax2.annotate(
        "Clean Separation Lift!\n$z = \\frac{s - \\mu_c}{\\sigma_c} > 3.0$\nHigh Anomaly Recovery",
        xy=(anom_mean_std, max_dens2 * 0.4),
        xytext=(anom_mean_std + 2.5, max_dens2 * 0.75),
        arrowprops=dict(facecolor="#2d3748", arrowstyle="->", lw=1.2),
        ha="center",
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f0fff4", ec="#9ae6b4", lw=1.0),
    )

    ax2.set_title("(b) After: Cluster-Standardized Anomaly Score (Objective 1)\n($z$-Score Standardized Against Nearest Class Cluster)", pad=10)
    ax2.set_xlabel("Standardized Anomaly Score: $z = (s - \\mu_c) / \\sigma_c$")
    ax2.set_ylabel("Probability Density")
    ax2.legend(loc="upper left", frameon=True, fontsize=8.8)

    fig.suptitle(
        "Dynamic Thresholding: Resolving the Rare-Class Detection Dilemma\n"
        "Comparison of Inlier vs. Subtle Outlier Distributions Before and After Cluster Standardization",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        print(f"[✓] Saved Adaptive Threshold Lift figure to {save_path}")

    return fig


# -----------------------------------------------------------------------------
# 4. 2D Manifold Projection of Clustered Latent Space (Figure 4)
# -----------------------------------------------------------------------------
def plot_latent_umap_projection(
    x_2d: np.ndarray,
    y_classes: np.ndarray,
    is_anomaly: Optional[np.ndarray] = None,
    method_name: str = "UMAP",
    save_path: Optional[str] = "figures/latent_umap_projection.png",
) -> plt.Figure:
    """Plots 2D manifold projection (UMAP / t-SNE) of the clustered latent space.
    
    Shows well-separated clusters for known astronomical types and the positioning
    of rare out-of-distribution anomalous transients (CaRT, Kilonova, PISN, etc.).
    
    Args:
        x_2d: Array of 2D coordinates with shape (N, 2).
        y_classes: Array of class labels for all points.
        is_anomaly: Boolean or binary array indicating known vs anomalous classes.
        method_name: Name of dimensionality reduction technique ('UMAP' or 't-SNE').
        save_path: Optional path to save the generated figure.
        
    Returns:
        The generated matplotlib Figure object.
    """
    apply_publication_style()

    if is_anomaly is None:
        known_anomalies = {"CaRT", "KNe", "KN", "PISN", "ILOT", "uLens-BSR", "uLens"}
        is_anomaly = np.isin(y_classes, list(known_anomalies))

    fig, ax = plt.subplots(figsize=(13, 9))

    # Common class palette (muted astrophysical shades)
    common_palette = {
        "SNIa": "#3182ce",      # Blue (Thermonuclear standard)
        "SNIa-91bg": "#63b3ed",
        "SNIax": "#90cdf4",
        "SNIb": "#38a169",      # Green (Stripped envelope)
        "SNIc": "#48bb78",
        "SNIc-BL": "#68d391",
        "SNII": "#d69e2e",      # Gold / Ochre (Core-collapse)
        "SNIIn": "#ecc94b",
        "SNIIb": "#f6e05e",
        "TDE": "#805ad5",       # Purple (Tidal disruption)
        "SLSN-I": "#dd6b20",    # Orange (Superluminous)
        "SLSN": "#dd6b20",
        "AGN": "#718096",       # Slate Gray (Galactic variability)
    }

    # Anomalous class markers and high-contrast colors
    anomaly_palette = {
        "CaRT": ("#e53e3e", "*", 160),       # Red Star
        "KNe": ("#b83280", "D", 110),        # Magenta Diamond
        "KN": ("#b83280", "D", 110),
        "PISN": ("#c53030", "^", 120),       # Dark Red Triangle
        "ILOT": ("#9f7aea", "s", 110),       # Violet Square
        "uLens-BSR": ("#319795", "P", 120),  # Teal Plus
        "uLens": ("#319795", "P", 120),
    }

    # 1. Plot In-Distribution Common Classes
    common_mask = ~is_anomaly
    unique_common = np.unique(y_classes[common_mask])

    for c in unique_common:
        c_idx = np.where(common_mask & (y_classes == c))[0]
        color = common_palette.get(c, "#a0aec0")
        ax.scatter(
            x_2d[c_idx, 0],
            x_2d[c_idx, 1],
            c=color,
            label=c,
            s=22,
            alpha=0.68,
            edgecolors="none",
        )

        # Plot class centroid annotation
        centroid_x = np.median(x_2d[c_idx, 0])
        centroid_y = np.median(x_2d[c_idx, 1])
        ax.text(
            centroid_x,
            centroid_y,
            c,
            fontsize=8.5,
            fontweight="bold",
            color="#1a202c",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#cbd5e1", alpha=0.85, lw=0.8),
        )

    # 2. Plot Rare Out-of-Distribution Anomalies
    anom_mask = is_anomaly
    unique_anom = np.unique(y_classes[anom_mask])

    for a in unique_anom:
        a_idx = np.where(anom_mask & (y_classes == a))[0]
        color, marker, size = anomaly_palette.get(a, ("#e53e3e", "X", 120))
        ax.scatter(
            x_2d[a_idx, 0],
            x_2d[a_idx, 1],
            c=color,
            marker=marker,
            s=size,
            label=f"★ {a} (Outlier)",
            alpha=0.95,
            edgecolors="#1a202c",
            linewidths=1.2,
            zorder=5,
        )

    ax.set_title(
        f"2D {method_name} Projection of the 100-Dimensional Clustered Latent Space\n"
        "Common Astrophysical Manifolds vs. Unmodelled Rare Anomalies",
        pad=14,
        fontsize=13,
    )
    ax.set_xlabel(f"{method_name} Dimension 1", fontsize=11)
    ax.set_ylabel(f"{method_name} Dimension 2", fontsize=11)

    # Dedicated dual legend (Common vs Anomalies)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.22, 1.0),
        frameon=True,
        title="Astronomical Classes",
        title_fontsize=10,
        fontsize=8.5,
        ncol=1,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[✓] Saved 2D Latent Projection ({method_name}) to {save_path}")

    return fig
