"""
config.py
=========
Central configuration for PCA + ANN Face Recognition System.
All hyperparameters, paths, and settings are defined here.
"""

import os
from pathlib import Path

# ── Project Root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.resolve()

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_DIR: Path = ROOT_DIR / "dataset" / "faces"
SUPPORTED_EXTENSIONS: tuple = (".jpg", ".jpeg", ".png", ".pgm", ".bmp")

# ── Image Preprocessing ───────────────────────────────────────────────────────
IMG_HEIGHT: int = 64          # Resize height (pixels)
IMG_WIDTH: int = 64           # Resize width  (pixels)
IMG_SIZE: tuple = (IMG_HEIGHT, IMG_WIDTH)
IMG_FLATTEN_DIM: int = IMG_HEIGHT * IMG_WIDTH  # 4096

# ── Train / Test Split ────────────────────────────────────────────────────────
TRAIN_RATIO: float = 0.60     # 60% training
TEST_RATIO: float = 0.40      # 40% testing
RANDOM_STATE: int = 42

# ── PCA ───────────────────────────────────────────────────────────────────────
# k values to sweep for accuracy-vs-k experiment
K_VALUES: list = [5, 10, 20, 30, 40, 50, 75, 100]
DEFAULT_K: int = 50           # Default number of principal components

# Number of eigenfaces to visualise
N_EIGENFACES_VISUALISE: int = 10

# ── ANN (MLPClassifier) ───────────────────────────────────────────────────────
ANN_HIDDEN_LAYERS: tuple = (256, 128)
ANN_ACTIVATION: str = "relu"
ANN_SOLVER: str = "adam"
ANN_ALPHA: float = 1e-4       # L2 regularisation
ANN_MAX_ITER: int = 500
ANN_EARLY_STOPPING: bool = True
ANN_VALIDATION_FRACTION: float = 0.1
ANN_N_ITER_NO_CHANGE: int = 20
ANN_RANDOM_STATE: int = 42

# ── Unknown Person Detection ──────────────────────────────────────────────────
UNKNOWN_THRESHOLD: float = 0.55   # If max class probability < threshold → Unknown

# ── Output Paths ──────────────────────────────────────────────────────────────
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"
MODELS_DIR: Path = ROOT_DIR / "models"

MEAN_FACE_PATH: Path = OUTPUTS_DIR / "mean_face.png"
ACCURACY_VS_K_PATH: Path = OUTPUTS_DIR / "accuracy_vs_k.png"
CONFUSION_MATRIX_PATH: Path = OUTPUTS_DIR / "confusion_matrix.png"
CLASSIFICATION_REPORT_PATH: Path = OUTPUTS_DIR / "classification_report.txt"
RESULTS_CSV_PATH: Path = OUTPUTS_DIR / "results.csv"

PCA_MODEL_PATH: Path = MODELS_DIR / "pca_model.pkl"
ANN_MODEL_PATH: Path = MODELS_DIR / "ann_model.pkl"
LABEL_ENCODER_PATH: Path = MODELS_DIR / "label_encoder.pkl"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# ── Reproducibility ───────────────────────────────────────────────────────────
NUMPY_SEED: int = 42
