"""
train.py
========
Main training entry point.

Usage
-----
    python train.py

This script:
1. Loads and preprocesses the face dataset.
2. Performs a 60/40 stratified train/test split.
3. Runs the accuracy-vs-k sweep across ``config.K_VALUES``.
4. Trains the final PCA + ANN model with the best-k.
5. Evaluates the model and saves all artefacts.
"""

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.trainer import Trainer
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    logger.info("=" * 65)
    logger.info(" PCA + ANN Face Recognition — Training Pipeline")
    logger.info("=" * 65)
    logger.info("Dataset    : %s", config.DATASET_DIR)
    logger.info("Image size : %s", config.IMG_SIZE)
    logger.info("k values   : %s", config.K_VALUES)
    logger.info("ANN layers : %s", config.ANN_HIDDEN_LAYERS)
    logger.info("Threshold  : %.2f", config.UNKNOWN_THRESHOLD)
    logger.info("=" * 65)

    trainer = Trainer()
    trainer.run()


if __name__ == "__main__":
    main()
