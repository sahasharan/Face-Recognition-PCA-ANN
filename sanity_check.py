"""
sanity_check.py
===============
Rapid sanity-check script that verifies every major component without
running a full training cycle.

Checks performed
----------------
1.  Dataset loading              — discovers all classes and images
2.  Image preprocessing          — shape and value range
3.  PCA dimensions               — fit on small subset; verify k
4.  ANN training                 — fit on toy data; verify predict()
5.  Model saving / loading       — round-trip pickle serialisation

Run
---
    python sanity_check.py

Expected output
---------------
All checks print PASS.  Any failure prints FAIL + traceback.
"""

import sys
import traceback
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

import config
from src.data_loader import DataLoader
from src.preprocessing import ImagePreprocessor
from src.pca_eigenfaces import PCAEigenfaces
from src.ann_classifier import ANNClassifier
from src.utils import get_logger

logger = get_logger("sanity_check")
PASS = "  ✅  PASS"
FAIL = "  ❌  FAIL"


def check(name: str, fn) -> bool:
    print(f"\n{'─'*55}")
    print(f"  Checking: {name}")
    print(f"{'─'*55}")
    try:
        fn()
        print(PASS)
        return True
    except Exception:
        print(FAIL)
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Check 1 – Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def test_dataset_loading():
    loader = DataLoader()
    loader.discover()
    assert len(loader.image_paths) > 0, "No images found"
    assert len(loader.class_names) > 0, "No classes found"
    images, labels = loader.load()
    assert len(images) > 0, "No images loaded"
    assert len(images) == len(labels), "Images / labels length mismatch"
    print(f"    Classes : {loader.class_names}")
    print(f"    Images  : {len(images)}")
    print(f"    Labels  : {labels[:5]} …")


# ─────────────────────────────────────────────────────────────────────────────
# Check 2 – Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def test_preprocessing():
    loader = DataLoader()
    loader.discover()
    raw, _ = loader.load()

    pre = ImagePreprocessor()
    X = pre.transform(raw[:10])

    expected_dim = config.IMG_HEIGHT * config.IMG_WIDTH
    assert X.shape == (10, expected_dim), f"Shape mismatch: {X.shape}"
    assert X.min() >= 0.0, "Values below 0"
    assert X.max() <= 1.0, "Values above 1"
    print(f"    Preprocessed shape : {X.shape}")
    print(f"    Value range        : [{X.min():.3f}, {X.max():.3f}]")


# ─────────────────────────────────────────────────────────────────────────────
# Check 3 – PCA dimensions
# ─────────────────────────────────────────────────────────────────────────────

def test_pca():
    rng = np.random.default_rng(42)
    # Synthetic data: 50 samples, 100 features
    X = rng.random((50, 100))
    k = 10

    pca = PCAEigenfaces(n_components=k)
    X_proj = pca.fit_transform(X)

    assert pca.mean_face_ is not None
    assert pca.eigenfaces_.shape[1] <= k, "Too many eigenfaces"
    assert X_proj.shape == (50, pca.n_components_), f"Proj shape: {X_proj.shape}"

    # Single-image projection
    sig = pca.project_one(X[0])
    assert sig.shape == (pca.n_components_,)

    print(f"    Eigenfaces shape      : {pca.eigenfaces_.shape}")
    print(f"    Projections shape     : {X_proj.shape}")
    print(f"    Explained variance    : {pca.explained_variance_ratio_.sum()*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Check 4 – ANN training
# ─────────────────────────────────────────────────────────────────────────────

def test_ann():
    rng = np.random.default_rng(42)
    X_tr = rng.random((80, 20))
    y_tr = rng.integers(0, 4, size=80)
    X_te = rng.random((20, 20))
    y_te = rng.integers(0, 4, size=20)

    ann = ANNClassifier(hidden_layers=(32,))
    ann.fit(X_tr, y_tr)
    preds = ann.predict(X_te)
    assert preds.shape == (20,), f"Predictions shape: {preds.shape}"
    proba = ann.predict_proba(X_te)
    assert proba.shape[1] == 4

    class_names = ["A", "B", "C", "D"]
    results = ann.predict_with_unknown(X_te[:3], class_names)
    assert len(results) == 3
    print(f"    Predictions sample  : {preds[:5]}")
    print(f"    Unknown example     : {results[0]}")


# ─────────────────────────────────────────────────────────────────────────────
# Check 5 – Model save / load (round-trip)
# ─────────────────────────────────────────────────────────────────────────────

def test_model_persistence():
    rng = np.random.default_rng(0)
    X = rng.random((40, 50))

    pca = PCAEigenfaces(n_components=5)
    pca.fit(X)

    tmp_dir = Path(tempfile.mkdtemp())
    pca_path = tmp_dir / "pca_test.pkl"

    try:
        pca.save(pca_path)
        pca2 = PCAEigenfaces.load(pca_path)

        sig1 = pca.project_one(X[0])
        sig2 = pca2.project_one(X[0])
        np.testing.assert_allclose(sig1, sig2, rtol=1e-6)
        print(f"    Round-trip signature match: ✅  max diff={np.abs(sig1-sig2).max():.2e}")
    finally:
        shutil.rmtree(tmp_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  PCA + ANN Face Recognition — Sanity Check")
    print("=" * 55)

    results = [
        check("1 — Dataset Loading",    test_dataset_loading),
        check("2 — Image Preprocessing",test_preprocessing),
        check("3 — PCA Dimensions",     test_pca),
        check("4 — ANN Training",       test_ann),
        check("5 — Model Save / Load",  test_model_persistence),
    ]

    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 55)
    print(f"  Results: {passed}/{total} checks passed.")
    print("=" * 55 + "\n")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
