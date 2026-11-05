"""
src/visualization.py
====================
Matplotlib-based visualisation routines.

Generates
---------
* Mean face image
* Top-k Eigenface images
* Accuracy vs k plot
* Confusion matrix heatmap
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.utils import ensure_dirs, get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

class Visualiser:
    """Collection of static plot-generation methods."""

    # ── Mean Face ─────────────────────────────────────────────────────────────

    @staticmethod
    def save_mean_face(
        mean_face: np.ndarray,
        img_size: tuple = config.IMG_SIZE,
        path: Path = config.MEAN_FACE_PATH,
    ) -> None:
        """
        Render and save the mean face.

        Parameters
        ----------
        mean_face : np.ndarray, shape (H*W,)
        img_size  : tuple[int, int]
        path      : Path
        """
        ensure_dirs(path.parent)
        face_2d = mean_face.reshape(img_size)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(face_2d, cmap="gray", vmin=0, vmax=1)
        ax.set_title("Mean Face", fontsize=14, fontweight="bold")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved mean face → %s", path)

    # ── Eigenfaces ────────────────────────────────────────────────────────────

    @staticmethod
    def save_eigenfaces(
        eigenfaces: np.ndarray,
        img_size: tuple = config.IMG_SIZE,
        n: int = config.N_EIGENFACES_VISUALISE,
        output_dir: Path = config.OUTPUTS_DIR,
    ) -> None:
        """
        Save each of the top *n* eigenfaces as individual PNG files,
        and also a composite grid figure.

        Parameters
        ----------
        eigenfaces : np.ndarray, shape (H*W, k)
        img_size   : tuple[int, int]
        n          : int
            Number of eigenfaces to visualise.
        output_dir : Path
        """
        ensure_dirs(output_dir)
        k = eigenfaces.shape[1]
        n = min(n, k)

        # Individual eigenface files
        for i in range(n):
            ef = eigenfaces[:, i].reshape(img_size)
            # Normalise to [0, 1] for display
            ef_norm = (ef - ef.min()) / (ef.max() - ef.min() + 1e-8)
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(ef_norm, cmap="gray")
            ax.set_title(f"Eigenface #{i + 1}", fontsize=10)
            ax.axis("off")
            fig.tight_layout()
            path = output_dir / f"eigenface_{i + 1}.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)

        # Composite grid
        cols = min(5, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
        axes = np.array(axes).flatten()

        for i in range(n):
            ef = eigenfaces[:, i].reshape(img_size)
            ef_norm = (ef - ef.min()) / (ef.max() - ef.min() + 1e-8)
            axes[i].imshow(ef_norm, cmap="gray")
            axes[i].set_title(f"#{i + 1}", fontsize=9)
            axes[i].axis("off")

        for j in range(n, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"Top-{n} Eigenfaces", fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout()
        grid_path = output_dir / "eigenfaces_grid.png"
        fig.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %d eigenface images + grid → %s", n, output_dir)

    # ── Accuracy vs k ─────────────────────────────────────────────────────────

    @staticmethod
    def save_accuracy_vs_k(
        k_values: list[int],
        accuracies: list[float],
        best_k: int,
        best_accuracy: float,
        path: Path = config.ACCURACY_VS_K_PATH,
    ) -> None:
        """
        Plot classification accuracy as a function of the number of
        principal components k.

        Parameters
        ----------
        k_values     : list[int]   — x-axis values
        accuracies   : list[float] — y-axis values (0–1)
        best_k       : int         — highlighted point
        best_accuracy: float
        path         : Path
        """
        ensure_dirs(path.parent)
        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(
            k_values, [a * 100 for a in accuracies],
            "o-", color="#2c7bb6", linewidth=2, markersize=7,
            label="Test Accuracy"
        )

        # Highlight best k
        ax.axvline(best_k, color="#d7191c", linestyle="--", linewidth=1.4,
                   label=f"Best k = {best_k}")
        ax.scatter([best_k], [best_accuracy * 100], s=150, zorder=5,
                   color="#d7191c", edgecolors="white", linewidths=1.5)
        ax.annotate(
            f"k={best_k}\n{best_accuracy*100:.1f}%",
            xy=(best_k, best_accuracy * 100),
            xytext=(best_k + 1, best_accuracy * 100 - 5),
            fontsize=9, color="#d7191c",
            arrowprops=dict(arrowstyle="->", color="#d7191c"),
        )

        ax.set_xlabel("Number of Principal Components (k)", fontsize=12)
        ax.set_ylabel("Test Accuracy (%)", fontsize=12)
        ax.set_title(
            "PCA + ANN Face Recognition: Accuracy vs k", fontsize=13, fontweight="bold"
        )
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved accuracy-vs-k plot → %s", path)

    # ── Confusion Matrix ──────────────────────────────────────────────────────

    @staticmethod
    def save_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: list[str],
        path: Path = config.CONFUSION_MATRIX_PATH,
    ) -> None:
        """
        Render and save a normalised confusion matrix heatmap.

        Parameters
        ----------
        y_true       : np.ndarray, shape (N,)
        y_pred       : np.ndarray, shape (N,)
        class_names  : list[str]
        path         : Path
        """
        ensure_dirs(path.parent)
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        n = len(class_names)
        fig_size = max(8, n * 1.1)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

        sns.heatmap(
            cm_norm,
            annot=cm,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "Normalised Count"},
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved confusion matrix → %s", path)
