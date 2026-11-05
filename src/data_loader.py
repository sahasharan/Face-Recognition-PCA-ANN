"""
src/data_loader.py
==================
Dataset discovery and raw image loading.

Responsibility
--------------
* Walk the ``dataset/faces`` directory.
* Collect image paths and string labels.
* Encode labels with sklearn's LabelEncoder.
* Print a human-readable dataset summary.
* Return raw pixel arrays ready for preprocessing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.utils import get_logger, save_object

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class DataLoader:
    """
    Discovers and loads the face image dataset from ``config.DATASET_DIR``.

    The expected directory layout is::

        dataset/faces/
        ├── PersonA/
        │   ├── image1.jpg
        │   └── ...
        ├── PersonB/
        │   └── ...
        └── ...

    Attributes
    ----------
    dataset_dir : Path
        Root directory for face images.
    label_encoder : LabelEncoder
        Fitted encoder mapping class names ↔ integer indices.
    class_names : list[str]
        Sorted list of discovered class names.
    image_paths : list[Path]
        Ordered list of every image path.
    labels_str : list[str]
        String label corresponding to each image path.
    """

    def __init__(self, dataset_dir: Path = config.DATASET_DIR) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.label_encoder = LabelEncoder()
        self.class_names: list[str] = []
        self.image_paths: list[Path] = []
        self.labels_str: list[str] = []

    # ── Discovery ─────────────────────────────────────────────────────────────

    def discover(self) -> "DataLoader":
        """
        Walk *dataset_dir* and collect image paths with their class labels.

        Returns
        -------
        DataLoader
            Self (for method chaining).

        Raises
        ------
        FileNotFoundError
            If *dataset_dir* does not exist.
        ValueError
            If no valid images are found.
        """
        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.dataset_dir}"
            )

        paths: list[Path] = []
        labels: list[str] = []

        for class_dir in sorted(self.dataset_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            found = [
                p
                for p in sorted(class_dir.iterdir())
                if p.suffix.lower() in config.SUPPORTED_EXTENSIONS
            ]
            if not found:
                logger.warning("No images in class folder '%s' — skipping.", class_name)
                continue
            paths.extend(found)
            labels.extend([class_name] * len(found))

        if not paths:
            raise ValueError(
                f"No images found in {self.dataset_dir}. "
                f"Supported extensions: {config.SUPPORTED_EXTENSIONS}"
            )

        self.image_paths = paths
        self.labels_str = labels
        self.class_names = sorted(set(labels))

        logger.info(
            "Discovered %d images across %d classes.", len(paths), len(self.class_names)
        )
        return self

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load every image as a raw pixel array and encode labels.

        Images are loaded as-is (colour); preprocessing (grayscale, resize,
        normalise, flatten) is handled by ``preprocessing.py``.

        Returns
        -------
        images : np.ndarray, shape (N, H, W, C)
            Raw BGR images as uint8 arrays.
        labels : np.ndarray, shape (N,)
            Integer-encoded class labels.

        Raises
        ------
        RuntimeError
            If ``discover()`` has not been called first.
        """
        if not self.image_paths:
            raise RuntimeError("Call discover() before load().")

        raw_images: list[np.ndarray] = []
        valid_paths: list[Path] = []
        valid_labels: list[str] = []

        for path, lbl in zip(self.image_paths, self.labels_str):
            img = cv2.imread(str(path))
            if img is None:
                logger.warning("Could not read image: %s — skipping.", path)
                continue
            raw_images.append(img)
            valid_paths.append(path)
            valid_labels.append(lbl)

        if not raw_images:
            raise ValueError("All images failed to load.")

        self.image_paths = valid_paths
        self.labels_str = valid_labels

        # Images may have different sizes; use object array as a container.
        # The ImagePreprocessor will resize every image to a uniform size.
        images = np.empty(len(raw_images), dtype=object)
        for i, img in enumerate(raw_images):
            images[i] = img
        labels = self.label_encoder.fit_transform(valid_labels)

        logger.info(
            "Loaded %d images.  Classes: %s",
            len(images),
            list(self.label_encoder.classes_),
        )
        return images, labels

    # ── Model Persistence ─────────────────────────────────────────────────────

    def save_label_encoder(self, path: Path = config.LABEL_ENCODER_PATH) -> None:
        """
        Persist the fitted LabelEncoder so inference does not need the dataset.

        Parameters
        ----------
        path : Path
            Destination pickle path.
        """
        save_object(self.label_encoder, path)

    # ── Summary ───────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """
        Print a formatted dataset summary to stdout / logger.
        """
        logger.info("=" * 60)
        logger.info("DATASET SUMMARY")
        logger.info("=" * 60)
        logger.info("Root directory : %s", self.dataset_dir)
        logger.info("Total images   : %d", len(self.image_paths))
        logger.info("Total classes  : %d", len(self.class_names))
        logger.info("")
        logger.info("Class distribution:")
        for cls in self.class_names:
            count = self.labels_str.count(cls)
            logger.info("  %-20s %d images", cls, count)
        logger.info("")
        logger.info("EXCLUDED DATASETS:")
        logger.info(
            "  Iris  — tabular CSV dataset (150 samples, 4 numeric features)."
            "  Irrelevant to face recognition; excluded from all processing."
        )
        logger.info("=" * 60)
