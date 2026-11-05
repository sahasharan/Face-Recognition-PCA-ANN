"""
src/ann_classifier.py
=====================
ANN classifier wrapper using sklearn's MLPClassifier.

Wraps ``sklearn.neural_network.MLPClassifier`` with:
* feature normalisation (StandardScaler)
* configurable hyperparameters from config.py
* unknown-person detection via probability thresholding
* model persistence
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.utils import get_logger, save_object, load_object

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

class ANNClassifier:
    """
    MLPClassifier wrapped with feature scaling and unknown-person detection.

    Parameters
    ----------
    hidden_layers : tuple[int, ...]
        Sizes of hidden layers, e.g. ``(256, 128)``.
    unknown_threshold : float
        If the maximum predicted class probability is below this value
        the prediction is labelled "Unknown Person".

    Attributes
    ----------
    scaler_ : StandardScaler
        Fitted feature normaliser.
    mlp_ : MLPClassifier
        Underlying neural network.
    is_fitted_ : bool
        Whether ``fit()`` has been called.
    """

    def __init__(
        self,
        hidden_layers: tuple = config.ANN_HIDDEN_LAYERS,
        unknown_threshold: float = config.UNKNOWN_THRESHOLD,
    ) -> None:
        self.hidden_layers = hidden_layers
        self.unknown_threshold = unknown_threshold

        self.scaler_: StandardScaler = StandardScaler()
        self.mlp_: MLPClassifier = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=config.ANN_ACTIVATION,
            solver=config.ANN_SOLVER,
            alpha=config.ANN_ALPHA,
            max_iter=config.ANN_MAX_ITER,
            early_stopping=config.ANN_EARLY_STOPPING,
            validation_fraction=config.ANN_VALIDATION_FRACTION,
            n_iter_no_change=config.ANN_N_ITER_NO_CHANGE,
            random_state=config.ANN_RANDOM_STATE,
        )
        self.is_fitted_: bool = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "ANNClassifier":
        """
        Fit the scaler then train the MLP.

        Parameters
        ----------
        X_train : np.ndarray, shape (N_train, k)
            PCA-projected training signatures.
        y_train : np.ndarray, shape (N_train,)
            Integer class labels.

        Returns
        -------
        ANNClassifier
            Self (fitted).
        """
        logger.info(
            "Training ANN on %d samples with %d features.", *X_train.shape
        )
        X_scaled = self.scaler_.fit_transform(X_train)
        self.mlp_.fit(X_scaled, y_train)
        self.is_fitted_ = True
        logger.info(
            "ANN training complete. Iterations: %d.", self.mlp_.n_iter_
        )
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return hard class predictions (integer labels).

        Parameters
        ----------
        X : np.ndarray, shape (N, k)

        Returns
        -------
        np.ndarray, shape (N,)
        """
        self._check_fitted()
        return self.mlp_.predict(self.scaler_.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probability matrix.

        Parameters
        ----------
        X : np.ndarray, shape (N, k)

        Returns
        -------
        np.ndarray, shape (N, n_classes)
        """
        self._check_fitted()
        return self.mlp_.predict_proba(self.scaler_.transform(X))

    def predict_with_unknown(
        self, X: np.ndarray, class_names: list[str]
    ) -> list[dict]:
        """
        Predict with unknown-person detection.

        For each sample:
        * If max class probability ≥ ``unknown_threshold`` → return class name.
        * Otherwise → return "Unknown Person".

        Parameters
        ----------
        X : np.ndarray, shape (N, k)
        class_names : list[str]
            Ordered class names matching the label encoder output.

        Returns
        -------
        list[dict]
            Each dict has keys:
            ``label``      (str)   — predicted class or "Unknown Person"
            ``confidence`` (float) — max probability score
            ``is_unknown`` (bool)
        """
        self._check_fitted()
        proba = self.predict_proba(X)                          # (N, n_classes)
        results = []
        for probs in proba:
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            if conf < self.unknown_threshold:
                results.append(
                    {"label": "Unknown Person", "confidence": conf, "is_unknown": True}
                )
            else:
                results.append(
                    {
                        "label": class_names[idx],
                        "confidence": conf,
                        "is_unknown": False,
                    }
                )
        return results

    # ── Accuracy helper ───────────────────────────────────────────────────────

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return accuracy on (X, y).

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray

        Returns
        -------
        float
        """
        self._check_fitted()
        return self.mlp_.score(self.scaler_.transform(X), y)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path = config.ANN_MODEL_PATH) -> None:
        """Serialise the fitted classifier (scaler + MLP)."""
        save_object(self, path)

    @classmethod
    def load(cls, path: Path = config.ANN_MODEL_PATH) -> "ANNClassifier":
        """
        Load a serialised ANNClassifier from disk.

        Returns
        -------
        ANNClassifier
        """
        obj = load_object(path)
        logger.info("Loaded ANN model from %s", path)
        return obj

    # ── Internal ──────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("ANNClassifier has not been fitted yet. Call fit().")
