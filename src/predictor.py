"""
src/predictor.py
================
Single-image inference: load models → preprocess → project → classify.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.ann_classifier import ANNClassifier
from src.pca_eigenfaces import PCAEigenfaces
from src.preprocessing import ImagePreprocessor
from src.utils import get_logger, load_object

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

class Predictor:
    """
    Encapsulates model loading and single-image inference.

    Parameters
    ----------
    pca_path          : Path
    ann_path          : Path
    label_encoder_path: Path
    unknown_threshold : float
        Passed through to ANNClassifier; overrides the threshold stored in
        the saved model if provided.
    """

    def __init__(
        self,
        pca_path: Path = config.PCA_MODEL_PATH,
        ann_path: Path = config.ANN_MODEL_PATH,
        label_encoder_path: Path = config.LABEL_ENCODER_PATH,
        unknown_threshold: float = config.UNKNOWN_THRESHOLD,
    ) -> None:
        logger.info("Loading models …")
        self.pca: PCAEigenfaces = PCAEigenfaces.load(pca_path)
        self.ann: ANNClassifier = ANNClassifier.load(ann_path)
        self.ann.unknown_threshold = unknown_threshold
        self.label_encoder = load_object(label_encoder_path)
        self.class_names: list[str] = list(self.label_encoder.classes_)
        self.preprocessor = ImagePreprocessor()
        logger.info("Models loaded.  Classes: %s", self.class_names)

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, image_path: Path | str) -> dict:
        """
        Predict the identity of the face in *image_path*.

        Parameters
        ----------
        image_path : Path | str
            Path to the query image.

        Returns
        -------
        dict
            Keys: ``label``, ``confidence``, ``is_unknown``
        """
        path = Path(image_path)
        logger.info("Predicting: %s", path)

        # Preprocess
        x = self.preprocessor.load_and_preprocess_single(path)  # (mn,)

        # PCA projection
        sig = self.pca.project_one(x).reshape(1, -1)             # (1, k)

        # Classify with unknown detection
        results = self.ann.predict_with_unknown(sig, self.class_names)
        return results[0]

    # ── Pretty print ──────────────────────────────────────────────────────────

    @staticmethod
    def format_result(result: dict) -> str:
        """
        Format a prediction result for human-readable output.

        Parameters
        ----------
        result : dict

        Returns
        -------
        str
        """
        label = result["label"]
        conf = result["confidence"] * 100
        if result["is_unknown"]:
            return f"Unknown Person\nConfidence: {conf:.1f}%"
        return f"Predicted Person: {label}\nConfidence: {conf:.1f}%"
