"""Astronomical anomaly detection performance evaluation metrics."""

from typing import Dict, List
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def compute_auroc(y_true_binary: np.ndarray, y_scores: np.ndarray) -> float:
    """Computes Area Under the ROC Curve for anomaly separation."""
    return float(roc_auc_score(y_true_binary, y_scores))


def compute_aucpr(y_true_binary: np.ndarray, y_scores: np.ndarray) -> float:
    """Computes Area Under the Precision-Recall Curve under extreme class imbalance."""
    precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
    return float(auc(recall, precision))


def compute_recall_at_budget(
    y_true_binary: np.ndarray,
    y_scores: np.ndarray,
    top_k: int = 2000,
) -> float:
    """Computes anomaly recall when following up the top-K highest scoring events.
    
    Args:
        y_true_binary: Array where 1 indicates anomaly, 0 indicates common transient.
        y_scores: Continuous anomaly scores (higher = more anomalous).
        top_k: Number of candidate events followed up by telescope facilities.
        
    Returns:
        Fraction of true anomalies successfully recovered within top-K candidates.
    """
    total_anomalies = np.sum(y_true_binary == 1)
    if total_anomalies == 0:
        return 0.0

    ranked_indices = np.argsort(y_scores)[::-1]
    top_candidates = ranked_indices[:top_k]
    recovered_anomalies = np.sum(y_true_binary[top_candidates] == 1)

    return float(recovered_anomalies / total_anomalies)


def evaluate_per_class_recovery(
    y_true_binary: np.ndarray,
    y_scores: np.ndarray,
    class_names: np.ndarray,
    target_classes: List[str],
    top_k: int = 2000,
) -> Dict[str, float]:
    """Computes detection recall broken down across individual anomalous classes."""
    ranked_indices = np.argsort(y_scores)[::-1]
    top_candidates = set(ranked_indices[:top_k])

    per_class_recall = {}
    for c in target_classes:
        class_indices = np.where(class_names == c)[0]
        if len(class_indices) == 0:
            per_class_recall[c] = 0.0
            continue
        recovered = sum(1 for idx in class_indices if idx in top_candidates)
        per_class_recall[c] = float(recovered / len(class_indices))

    return per_class_recall
