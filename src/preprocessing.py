"""
src/preprocessing.py
====================
Image preprocessing pipeline.

Steps applied to every image
-----------------------------
1. Convert BGR → Grayscale
2. Resize to ``config.IMG_SIZE``
3. Normalise pixel values to [0, 1]  (float64)
4. Flatten to a 1-D column vector of length ``config.IMG_FLATTEN_DIM``

All parameters are driven by ``config.py`` so no magic numbers appear here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class ImagePreprocessor:
    """
    Stateless preprocessing transformer: raw images → flat normalised vectors.

    Parameters
    ----------
    img_size : tuple[int, int]
        Target (height, width) for resizing.  Defaults to ``config.IMG_SIZE``.
    """

    def __init__(self, img_size: Tuple[int, int] = config.IMG_SIZE) -> None:
        self.img_size = img_size

    # ── Single-image helpers ──────────────────────────────────────────────────

    def preprocess_one(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the full preprocessing pipeline to a single raw image.

        Parameters
        ----------
        image : np.ndarray
            Raw image array.  May be grayscale (H, W) or colour (H, W, C).

        Returns
        -------
        np.ndarray, shape (H*W,)
            Flattened, normalised float64 vector.
        """
        gray = self._to_grayscale(image)
        resized = self._resize(gray)
        normalised = self._normalise(resized)
        flat = normalised.flatten()
        return flat

    # ── Batch transform ───────────────────────────────────────────────────────

    def transform(self, images: np.ndarray) -> np.ndarray:
        """
        Apply the preprocessing pipeline to a batch of raw images.

        Parameters
        ----------
        images : np.ndarray, shape (N, H_src, W_src) or (N, H_src, W_src, C)
            Batch of raw images (uint8 or float).

        Returns
        -------
        np.ndarray, shape (N, H*W)
            Matrix where each row is a preprocessed face vector.
            This is the **Face_Db** matrix from the assignment (transposed).
        """
        N = len(images)
        dim = self.img_size[0] * self.img_size[1]
        X = np.empty((N, dim), dtype=np.float64)
        for i in range(N):
            # Support both regular ndarray batches and object-arrays
            img = images[i] if images.dtype == object else images[i]
            X[i] = self.preprocess_one(img)
        logger.info(
            "Preprocessed %d images → matrix shape %s", N, X.shape
        )
        return X

    def load_and_preprocess_single(self, image_path: Path) -> np.ndarray:
        """
        Load a single image from disk and run the full pipeline.

        Parameters
        ----------
        image_path : Path
            Path to the image file.

        Returns
        -------
        np.ndarray, shape (H*W,)
            Preprocessed flat vector.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If OpenCV cannot decode the file.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"OpenCV could not read image: {path}")

        return self.preprocess_one(img)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert an image to single-channel grayscale."""
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] == 1:
            return image[:, :, 0]
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize a grayscale image to (width, height) = config.IMG_SIZE[::-1]."""
        h, w = self.img_size
        if image.shape[:2] == (h, w):
            return image
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _normalise(image: np.ndarray) -> np.ndarray:
        """Scale pixel values from [0, 255] → [0, 1] (float64)."""
        return image.astype(np.float64) / 255.0
