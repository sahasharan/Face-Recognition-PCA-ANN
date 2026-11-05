"""
src/trainer.py
==============
Orchestrates the full training pipeline:

    DataLoader → Preprocessor → Train/Test Split
    → PCA sweep (accuracy vs k)
    → Best-k PCA fit
    → ANN training
    → Evaluation
    → Persistence
    → Visualisation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.ann_classifier import ANNClassifier
from src.data_loader import DataLoader
from src.evaluator import Evaluator
from src.pca_eigenfaces import PCAEigenfaces
from src.preprocessing import ImagePreprocessor
from src.visualization import Visualiser
from src.utils import Timer, ensure_dirs, get_logger, set_global_seed

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    End-to-end training orchestrator.

    Usage
    -----
    >>> trainer = Trainer()
    >>> trainer.run()
    """

    def __init__(self) -> None:
        set_global_seed()
        ensure_dirs(config.OUTPUTS_DIR, config.MODELS_DIR)
        self.loader = DataLoader()
        self.preprocessor = ImagePreprocessor()
        self.visualiser = Visualiser()

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self) -> None:
        """Execute the complete training pipeline."""

        # ── 1 · Load & preprocess ─────────────────────────────────────────────
        logger.info("Step 1/7 — Loading dataset …")
        self.loader.discover()
        self.loader.print_summary()

        with Timer("Data loading"):
            raw_images, y = self.loader.load()

        with Timer("Preprocessing"):
            X = self.preprocessor.transform(raw_images)       # (N, mn)

        # ── 2 · Train / test split ────────────────────────────────────────────
        logger.info("Step 2/7 — Splitting 60/40 …")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.TEST_RATIO,
            stratify=y,
            random_state=config.RANDOM_STATE,
        )
        logger.info(
            "Train: %d samples | Test: %d samples", len(X_train), len(X_test)
        )

        # ── 3 · Visualise mean face ───────────────────────────────────────────
        logger.info("Step 3/7 — Computing and saving mean face …")
        mean_face = X_train.mean(axis=0)
        self.visualiser.save_mean_face(mean_face)

        # ── 4 · Accuracy vs k sweep ───────────────────────────────────────────
        logger.info("Step 4/7 — Accuracy vs k sweep …")
        k_values, accs, precs, recs, f1s = self._sweep_k(
            X_train, X_test, y_train, y_test
        )
        best_idx = int(np.argmax(accs))
        best_k = k_values[best_idx]
        best_acc = accs[best_idx]
        logger.info("Best k=%d  →  accuracy=%.4f", best_k, best_acc)

        self.visualiser.save_accuracy_vs_k(k_values, accs, best_k, best_acc)

        evaluator = Evaluator(class_names=list(self.loader.label_encoder.classes_))
        evaluator.save_results_csv(k_values, accs, precs, recs, f1s)

        # ── 5 · Fit final PCA + ANN with best k ──────────────────────────────
        logger.info("Step 5/7 — Final PCA + ANN (k=%d) …", best_k)
        with Timer(f"PCA fit k={best_k}"):
            pca = PCAEigenfaces(n_components=best_k)
            X_train_proj = pca.fit_transform(X_train)
            X_test_proj = pca.transform(X_test)

        self.visualiser.save_eigenfaces(pca.eigenfaces_)

        with Timer("ANN training"):
            ann = ANNClassifier()
            ann.fit(X_train_proj, y_train)

        # ── 6 · Evaluate ──────────────────────────────────────────────────────
        logger.info("Step 6/7 — Evaluating …")
        y_pred = ann.predict(X_test_proj)
        metrics = evaluator.compute_metrics(y_test, y_pred)
        evaluator.print_banner(metrics, best_k)
        evaluator.save_classification_report(y_test, y_pred, k=best_k)
        self.visualiser.save_confusion_matrix(
            y_test, y_pred,
            class_names=list(self.loader.label_encoder.classes_),
        )

        # ── 7 · Persist models ────────────────────────────────────────────────
        logger.info("Step 7/7 — Saving models …")
        pca.save()
        ann.save()
        self.loader.save_label_encoder()
        logger.info("✅  Training complete.  Outputs in '%s'.", config.OUTPUTS_DIR)

    # ── k-sweep ───────────────────────────────────────────────────────────────

    def _sweep_k(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[list, list, list, list, list]:
        """
        Train PCA + ANN for each k in ``config.K_VALUES`` and collect metrics.

        Returns
        -------
        Tuple of five equal-length lists:
            k_values, accuracies, precisions, recalls, f1_scores
        """
        evaluator = Evaluator(class_names=list(self.loader.label_encoder.classes_))
        k_values, accs, precs, recs, f1s = [], [], [], [], []

        max_k = min(X_train.shape[0] - 1, X_train.shape[1])

        for k in config.K_VALUES:
            if k > max_k:
                logger.warning("k=%d exceeds max allowable (%d) — skipping.", k, max_k)
                continue

            pca = PCAEigenfaces(n_components=k)
            X_tr_proj = pca.fit_transform(X_train)
            X_te_proj = pca.transform(X_test)

            ann = ANNClassifier()
            ann.fit(X_tr_proj, y_train)

            y_pred = ann.predict(X_te_proj)
            m = evaluator.compute_metrics(y_test, y_pred)

            k_values.append(k)
            accs.append(m["accuracy"])
            precs.append(m["precision"])
            recs.append(m["recall"])
            f1s.append(m["f1"])

            logger.info("k=%3d | acc=%.4f | f1=%.4f", k, m["accuracy"], m["f1"])

        return k_values, accs, precs, recs, f1s
