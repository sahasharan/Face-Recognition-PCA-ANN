"""
src/evaluator.py
================
Model evaluation: metrics, classification report, results CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.utils import ensure_dirs, get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Computes and persists evaluation metrics.

    Parameters
    ----------
    class_names : list[str]
        Ordered list of class labels.
    """

    def __init__(self, class_names: list[str]) -> None:
        self.class_names = class_names

    # ── Core metrics ──────────────────────────────────────────────────────────

    def compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict:
        """
        Compute accuracy, precision, recall, and F1 score.

        Parameters
        ----------
        y_true : np.ndarray
        y_pred : np.ndarray

        Returns
        -------
        dict with keys: accuracy, precision, recall, f1
        """
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
        logger.info(
            "Metrics — Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f",
            acc, prec, rec, f1,
        )
        return metrics

    # ── Classification report ─────────────────────────────────────────────────

    def save_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        path: Path = config.CLASSIFICATION_REPORT_PATH,
        k: Optional[int] = None,
    ) -> str:
        """
        Generate sklearn classification report and save to disk.

        Parameters
        ----------
        y_true : np.ndarray
        y_pred : np.ndarray
        path   : Path
        k      : Optional[int]
            Number of PCA components used (for labelling).

        Returns
        -------
        str
            The report text.
        """
        ensure_dirs(path.parent)
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            zero_division=0,
        )
        header = f"PCA + ANN Face Recognition — Classification Report"
        if k is not None:
            header += f"  (k={k} components)"
        header += "\n" + "=" * 60 + "\n"

        full_text = header + report
        path.write_text(full_text)
        logger.info("Saved classification report → %s", path)
        return full_text

    # ── Results CSV ───────────────────────────────────────────────────────────

    def save_results_csv(
        self,
        k_values: list[int],
        accuracies: list[float],
        precisions: list[float],
        recalls: list[float],
        f1s: list[float],
        path: Path = config.RESULTS_CSV_PATH,
    ) -> None:
        """
        Save per-k experiment results to a CSV file.

        Parameters
        ----------
        k_values    : list[int]
        accuracies  : list[float]
        precisions  : list[float]
        recalls     : list[float]
        f1s         : list[float]
        path        : Path
        """
        ensure_dirs(path.parent)
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["k", "accuracy", "precision", "recall", "f1"],
            )
            writer.writeheader()
            for k, acc, prec, rec, f1 in zip(
                k_values, accuracies, precisions, recalls, f1s
            ):
                writer.writerow(
                    {
                        "k": k,
                        "accuracy": f"{acc:.4f}",
                        "precision": f"{prec:.4f}",
                        "recall": f"{rec:.4f}",
                        "f1": f"{f1:.4f}",
                    }
                )
        logger.info("Saved results CSV → %s", path)

    # ── Print banner ──────────────────────────────────────────────────────────

    @staticmethod
    def print_banner(metrics: dict, k: int) -> None:
        """Pretty-print the evaluation summary."""
        logger.info("=" * 50)
        logger.info("EVALUATION RESULTS  (k=%d principal components)", k)
        logger.info("=" * 50)
        logger.info("  Accuracy  : %.4f  (%.2f%%)", metrics["accuracy"], metrics["accuracy"] * 100)
        logger.info("  Precision : %.4f", metrics["precision"])
        logger.info("  Recall    : %.4f", metrics["recall"])
        logger.info("  F1 Score  : %.4f", metrics["f1"])
        logger.info("=" * 50)
