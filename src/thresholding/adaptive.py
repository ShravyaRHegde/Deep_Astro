"""Class-conditioned dynamic and adaptive thresholding strategies (Objective 1)."""

import numpy as np
from scipy import stats


class ClusterStandardizedScorer:
    """Standardizes raw anomaly scores against the empirical distribution of the nearest class cluster.
    
    Addresses the 'CaRT Dilemma': Subtle anomalies near dense clusters receive a calibrated
    z-score rather than being suppressed by diffuse outlier scores from other classes.
    """

    def __init__(self):
        self.means = {}
        self.stds = {}
        self.classes = []

    def fit(self, val_matrix: np.ndarray, val_labels: np.ndarray, classes: np.ndarray):
        """Fits baseline mean and standard deviation per class from validation inliers.
        
        Args:
            val_matrix: Shape (N_val, N_classes) containing scores relative to each class.
            val_labels: Array of true class labels for validation samples.
            classes: Array of class names corresponding to columns of val_matrix.
        """
        self.classes = list(classes)
        for i, c in enumerate(self.classes):
            idx = np.where(val_labels == c)[0]
            if len(idx) > 0:
                class_scores = val_matrix[idx, i]
                self.means[c] = float(np.mean(class_scores))
                self.stds[c] = float(np.std(class_scores) + 1e-7)
            else:
                self.means[c] = 0.0
                self.stds[c] = 1.0
        return self

    def score(self, score_matrix: np.ndarray) -> np.ndarray:
        """Transforms a multi-class score matrix into standardized anomaly scores.
        
        Args:
            score_matrix: Shape (N_samples, N_classes)
            
        Returns:
            1D array of standardized anomaly scores.
        """
        num_samples = score_matrix.shape[0]
        calibrated_scores = np.zeros(num_samples)

        # Nearest class index for each object
        nearest_idx = np.argmin(score_matrix, axis=1)

        for i in range(num_samples):
            c_idx = nearest_idx[i]
            c_name = self.classes[c_idx]
            raw_score = score_matrix[i, c_idx]
            mu = self.means.get(c_name, 0.0)
            sig = self.stds.get(c_name, 1.0)
            calibrated_scores[i] = (raw_score - mu) / sig

        return calibrated_scores


class EmpiricalQuantileNormalizer:
    """Maps raw anomaly scores into empirical CDF percentiles ([0, 1]) per class cluster."""

    def __init__(self):
        self.ecdfs = {}
        self.classes = []

    def fit(self, val_matrix: np.ndarray, val_labels: np.ndarray, classes: np.ndarray):
        self.classes = list(classes)
        for i, c in enumerate(self.classes):
            idx = np.where(val_labels == c)[0]
            if len(idx) > 0:
                self.ecdfs[c] = np.sort(val_matrix[idx, i])
            else:
                self.ecdfs[c] = np.array([0.0, 1.0])
        return self

    def score(self, score_matrix: np.ndarray) -> np.ndarray:
        num_samples = score_matrix.shape[0]
        percentile_matrix = np.zeros_like(score_matrix)

        for col, c_name in enumerate(self.classes):
            ref_scores = self.ecdfs[c_name]
            # Empirical percentile rank
            percentile_matrix[:, col] = (
                np.searchsorted(ref_scores, score_matrix[:, col], side="right") / len(ref_scores)
            )

        # Overall anomaly score is the minimum percentile across all classes
        return np.min(percentile_matrix, axis=1)
