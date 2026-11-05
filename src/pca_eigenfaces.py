"""
src/pca_eigenfaces.py
=====================
Manual PCA / Eigenfaces implementation.

Mathematical pipeline (following the assignment exactly)
---------------------------------------------------------

Face_Db  (mn × p)  — each column is a flattened face image.

1.  Mean Face
        M  =  (1/p) * sum_i Face_Db[:, i]          [mn × 1]

2.  Mean Centering
        Δ_i  =  Face_Db[:, i] - M                  [mn × 1]
        Δ    assembled into matrix A  (mn × p)

3.  Surrogate Covariance  (Turk & Pentland, 1991)
        C_surrogate  =  Δᵀ Δ                        [p × p]
        (avoids the intractable mn×mn covariance matrix)

4.  Eigenvalue Decomposition of C_surrogate
        C_surrogate · v  =  λ · v

5.  Feature vector (basis) Ψ  (p × k)
        Sort eigenvectors by descending eigenvalue.
        Select top-k columns.

6.  Eigenfaces  Φ  (mn × k)
        Φ  =  Ψᵀ · Δᵀ   →   Φ  =  A · Ψ           [mn × k]
        L2-normalise each column.

7.  Signatures (projections)  ω  (k × p)
        ω_i  =  Φᵀ · Δ_i                            [k × 1]

Notation
--------
  mn  = flattened image dimension  (H × W)
  p   = number of training images
  k   = chosen number of principal components  (k ≤ p)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import eigh

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.utils import get_logger, save_object, load_object

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PCA Eigenfaces
# ─────────────────────────────────────────────────────────────────────────────

class PCAEigenfaces:
    """
    Manual PCA-based Eigenfaces for face recognition.

    Parameters
    ----------
    n_components : int
        Number of principal components (k) to retain.

    Attributes
    ----------
    mean_face_ : np.ndarray, shape (mn,)
        The computed mean face vector M.
    eigenfaces_ : np.ndarray, shape (mn, k)
        The Eigenface basis matrix Φ  (column-wise unit-norm eigenvectors).
    eigenvalues_ : np.ndarray, shape (k,)
        The corresponding eigenvalues, descending order.
    explained_variance_ratio_ : np.ndarray, shape (k,)
        Fraction of total variance captured by each component.
    n_components_ : int
        Actual k used (may be ≤ requested if training set is small).
    """

    def __init__(self, n_components: int = config.DEFAULT_K) -> None:
        self.n_components = n_components

        # Fitted attributes (set after fit())
        self.mean_face_: Optional[np.ndarray] = None
        self.eigenfaces_: Optional[np.ndarray] = None
        self.eigenvalues_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.n_components_: int = 0

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "PCAEigenfaces":
        """
        Compute PCA Eigenfaces from training data.

        Parameters
        ----------
        X : np.ndarray, shape (p, mn)
            Training matrix — each row is a flattened face image.

        Returns
        -------
        PCAEigenfaces
            Self (fitted).
        """
        p, mn = X.shape
        logger.info("Fitting PCA on %d training images, dim=%d.", p, mn)

        # ── Step 1 · Mean face ────────────────────────────────────────────────
        #   M  =  (1/p) * Σ X_i
        M: np.ndarray = X.mean(axis=0)                         # (mn,)
        self.mean_face_ = M

        # ── Step 2 · Mean-centre (Δ) ─────────────────────────────────────────
        #   Δ_i  =  X_i - M
        Delta: np.ndarray = X - M                              # (p, mn)

        # ── Step 3 · Surrogate covariance  C = ΔᵀΔ / (p-1) ──────────────────
        #   Shape:  (p, p)  — far cheaper than the (mn × mn) full covariance.
        C_surrogate: np.ndarray = (Delta @ Delta.T) / (p - 1) # (p, p)
        logger.info("Surrogate covariance matrix: %s", C_surrogate.shape)

        # ── Step 4 · Eigenvalue decomposition ────────────────────────────────
        #   scipy.linalg.eigh solves the symmetric eigenproblem in O(p³).
        #   Returns eigenvalues in ASCENDING order → reverse for descending.
        k = min(self.n_components, p - 1)
        eigenvalues, V = eigh(C_surrogate, subset_by_index=[p - k, p - 1])
        eigenvalues = eigenvalues[::-1].copy()                 # descending
        V = V[:, ::-1].copy()                                  # (p, k)

        logger.info(
            "Eigenvalue decomposition done. "
            "Requested k=%d, effective k=%d.", self.n_components, k
        )

        # ── Step 5 · Eigenfaces  Φ = Δᵀ · V ─────────────────────────────────
        #   Map surrogate eigenvectors back to image space.
        #   Δᵀ : (mn, p),  V : (p, k)  →  Φ : (mn, k)
        Phi: np.ndarray = Delta.T @ V                          # (mn, k)

        # L2-normalise each eigenface column
        norms = np.linalg.norm(Phi, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        Phi /= norms                                           # (mn, k)

        self.eigenfaces_ = Phi
        self.eigenvalues_ = eigenvalues
        self.n_components_ = k

        # Explained variance ratio
        total_var = eigenvalues.sum()
        self.explained_variance_ratio_ = (
            eigenvalues / total_var if total_var > 0 else np.zeros(k)
        )

        logger.info(
            "Top-%d components explain %.2f%% of variance.",
            k,
            100.0 * self.explained_variance_ratio_.sum(),
        )
        return self

    # ── Transform ─────────────────────────────────────────────────────────────

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project images into the Eigenface space (face signatures).

        Parameters
        ----------
        X : np.ndarray, shape (N, mn)
            Batch of flattened face vectors.

        Returns
        -------
        np.ndarray, shape (N, k)
            Projected signatures  ω  =  Φᵀ (X - M).

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        self._check_fitted()
        Delta = X - self.mean_face_                            # (N, mn)
        signatures = Delta @ self.eigenfaces_                  # (N, k)
        return signatures

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on *X* and return projections immediately."""
        return self.fit(X).transform(X)

    # ── Single-image inference ────────────────────────────────────────────────

    def project_one(self, x: np.ndarray) -> np.ndarray:
        """
        Project a single face vector into Eigenface space.

        Parameters
        ----------
        x : np.ndarray, shape (mn,)
            A single preprocessed face vector.

        Returns
        -------
        np.ndarray, shape (k,)
            Face signature.
        """
        self._check_fitted()
        delta = x - self.mean_face_                            # (mn,)
        return delta @ self.eigenfaces_                        # (k,)

    # ── Reconstruction ────────────────────────────────────────────────────────

    def reconstruct(self, signatures: np.ndarray) -> np.ndarray:
        """
        Reconstruct face images from their Eigenface signatures.

        Parameters
        ----------
        signatures : np.ndarray, shape (N, k)

        Returns
        -------
        np.ndarray, shape (N, mn)
            Approximate reconstructed face vectors.
        """
        self._check_fitted()
        return signatures @ self.eigenfaces_.T + self.mean_face_

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path = config.PCA_MODEL_PATH) -> None:
        """Serialise the fitted PCA model."""
        save_object(self, path)

    @classmethod
    def load(cls, path: Path = config.PCA_MODEL_PATH) -> "PCAEigenfaces":
        """
        Load a serialised PCA model from disk.

        Parameters
        ----------
        path : Path

        Returns
        -------
        PCAEigenfaces
        """
        obj = load_object(path)
        logger.info("Loaded PCA model from %s", path)
        return obj

    # ── Internal ──────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self.mean_face_ is None:
            raise RuntimeError("PCAEigenfaces has not been fitted yet. Call fit().")
