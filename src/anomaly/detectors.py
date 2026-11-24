"""Anomaly detection architectures for latent transient representations."""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class BaseAnomalyDetector(ABC):
    """Abstract base class for transient anomaly detectors."""

    @abstractmethod
    def fit(self, x_latent: np.ndarray, labels: np.ndarray = None):
        """Fits the detector to the training latent space."""
        pass

    @abstractmethod
    def score(self, x_latent: np.ndarray) -> np.ndarray:
        """Returns anomaly scores for latent samples (higher = more anomalous)."""
        pass


class GlobalIsolationForest(BaseAnomalyDetector):
    """Standard single Isolation Forest trained on pooled latent observations."""

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def fit(self, x_latent: np.ndarray, labels: np.ndarray = None):
        self.model.fit(x_latent)
        return self

    def score(self, x_latent: np.ndarray) -> np.ndarray:
        # Scikit-learn decision_function is positive for inliers, negative for outliers.
        # Negate so higher scores indicate greater anomaly likelihood.
        return -self.model.decision_function(x_latent)


class MultiClassIsolationForest(BaseAnomalyDetector):
    """Multi-Class Isolation Forest (MCIF).
    
    Trains an independent Isolation Forest for each known class cluster.
    Final anomaly score is the minimum anomaly score across all class detectors.
    """

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.classes = []
        self.forests = {}

    def fit(self, x_latent: np.ndarray, labels: np.ndarray):
        self.classes = np.unique(labels)
        self.forests = {}
        for c in self.classes:
            idx = np.where(labels == c)[0]
            forest = IsolationForest(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            )
            forest.fit(x_latent[idx])
            self.forests[c] = forest
        return self

    def score_per_class(self, x_latent: np.ndarray) -> np.ndarray:
        """Computes anomaly scores relative to each class detector."""
        class_scores = []
        for c in self.classes:
            detector = self.forests[c]
            class_scores.append(-detector.decision_function(x_latent))
        return np.array(class_scores).T  # Shape: (N_samples, N_classes)

    def score(self, x_latent: np.ndarray) -> np.ndarray:
        matrix = self.score_per_class(x_latent)
        return np.min(matrix, axis=1)


class GlobalLOF(BaseAnomalyDetector):
    """Standard Local Outlier Factor (LOF) trained across all common classes."""

    def __init__(self, n_neighbors: int = 20):
        self.n_neighbors = n_neighbors
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            novelty=True,
            n_jobs=-1,
        )

    def fit(self, x_latent: np.ndarray, labels: np.ndarray = None):
        self.model.fit(x_latent)
        return self

    def score(self, x_latent: np.ndarray) -> np.ndarray:
        raw = -self.model.decision_function(x_latent)
        return np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=-1.0)


class MultiClassLOF(BaseAnomalyDetector):
    """Multi-Class Local Outlier Factor (MCLOF).
    
    Fits class-specific local density estimators to handle distinct manifold densities.
    """

    def __init__(self, n_neighbors: int = 20):
        self.n_neighbors = n_neighbors
        self.classes = []
        self.lofs = {}

    def fit(self, x_latent: np.ndarray, labels: np.ndarray):
        self.classes = np.unique(labels)
        self.lofs = {}
        for c in self.classes:
            idx = np.where(labels == c)[0]
            # Ensure n_neighbors does not exceed class sample size
            k = min(self.n_neighbors, max(2, len(idx) - 1))
            lof = LocalOutlierFactor(
                n_neighbors=k,
                novelty=True,
                n_jobs=-1,
            )
            lof.fit(x_latent[idx])
            self.lofs[c] = lof
        return self

    def score_per_class(self, x_latent: np.ndarray) -> np.ndarray:
        class_scores = []
        for c in self.classes:
            detector = self.lofs[c]
            raw = -detector.decision_function(x_latent)
            class_scores.append(np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=-1.0))
        return np.array(class_scores).T

    def score(self, x_latent: np.ndarray) -> np.ndarray:
        matrix = self.score_per_class(x_latent)
        return np.min(matrix, axis=1)
