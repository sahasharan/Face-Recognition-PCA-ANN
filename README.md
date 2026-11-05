# PCA + ANN Face Recognition System

**Assignment:** Implementation of PCA with ANN algorithm for Face Recognition  
**Dataset:** Bollywood Celebrity Face Dataset (9 classes, 450 images)  
**Approach:** Eigenfaces (manual NumPy/SciPy PCA) + MLPClassifier

---

## 1 · Project Overview

This project implements a complete, production-ready face recognition system that:

1. Computes **Eigenfaces** via manual PCA (Turk & Pentland, 1991)
2. Projects each face into a low-dimensional Eigenface subspace
3. Trains a **multi-layer perceptron (ANN)** on the projected signatures
4. Detects **unknown persons** via confidence thresholding
5. Analyses the effect of varying **k** (principal components) on accuracy

---

## 2 · Assignment Objective

- Implement the Eigenfaces algorithm from scratch (NumPy / SciPy only)
- Use a 60 % training / 40 % test split
- Plot accuracy vs number of principal components (k)
- Handle impostors (unknown persons) not present in training

---

## 3 · Dataset Analysis

| Property       | Value                                                        |
|---------------|--------------------------------------------------------------|
| Root path      | `dataset/faces/`                                             |
| Classes        | Aamir, Ajay, Akshay, Alia, Amitabh, Deepika, Disha, Farhan, Ileana |
| Images / class | 50                                                           |
| Total images   | 450                                                          |
| Raw image size | 97 × 97 px, RGB                                              |
| Preprocessed   | 64 × 64 px, grayscale, normalised [0, 1]                     |

**Excluded dataset:** `dataset/Iris/` — tabular CSV data (iris flowers, 4 numeric
features, 3 classes).  Not related to face recognition and therefore excluded
from all processing.

---

## 4 · PCA Theory

Principal Component Analysis (PCA) finds the orthogonal directions of maximum
variance in a high-dimensional dataset.  Given a matrix of centred observations,
the principal components are the **eigenvectors of the covariance matrix**
ordered by their eigenvalues (descending).

Projecting onto the top-k components yields a compact representation that
preserves most information while discarding noise.

---

## 5 · Eigenfaces Theory

Eigenfaces (Turk & Pentland, 1991) apply PCA to face images:

- Each face is a point in an **mn-dimensional space** (m × n pixels flattened)
- The principal components of the face space are called **Eigenfaces**
- Any face can be approximately reconstructed as a linear combination of Eigenfaces
- The **weights** of this combination form a compact **face signature** used for recognition

Key insight: instead of computing the huge mn × mn covariance matrix, the
**surrogate covariance** (p × p, where p = number of training images) is used.
Its eigenvectors are then mapped back to image space.

---

## 6 · ANN Theory

A Multi-Layer Perceptron (MLP) is trained on the PCA-projected face signatures:

- **Input layer:** k neurons (one per principal component)
- **Hidden layers:** configurable (default: 256 → 128 neurons, ReLU activation)
- **Output layer:** 9 neurons (softmax probabilities over classes)
- **Training:** Adam optimiser, L2 regularisation, early stopping

---

## 7 · Mathematical Formulation

### Training

```
Face_Db (mn × p)       — each column is a flattened face image

Mean face:
    M = (1/p) Σ Face_Db[:, i]                     (mn × 1)

Mean-centred matrix:
    Δ_i = Face_Db[:, i] − M
    A   = [Δ_1 | Δ_2 | … | Δ_p]                  (mn × p)

Surrogate covariance (Turk & Pentland):
    C = Aᵀ A / (p−1)                              (p × p)

Eigendecomposition:
    C · v = λ · v    →    V (p × k), Λ (k,)

Eigenfaces (mapped back to image space):
    Φ = A · V                                      (mn × k)
    (each column L2-normalised)

Face signature (projection):
    ω_i = Φᵀ · Δ_i                                (k × 1)
```

### Testing

```
Given test image I (mn × 1):

    Δ = I − M                          (mean centering)
    Ω = Φᵀ · Δ                         (projection onto Eigenface space)

Feed Ω into trained ANN → softmax probabilities
If max probability < threshold → "Unknown Person"
```

---

## 8 · Project Structure

```
FaceRecognitionPCAANN/
│
├── dataset/
│   └── faces/
│       ├── Aamir/      (50 images)
│       ├── Ajay/       (50 images)
│       └── …
│
├── src/
│   ├── data_loader.py      Dataset discovery & loading
│   ├── preprocessing.py    Grayscale, resize, normalise, flatten
│   ├── pca_eigenfaces.py   Manual PCA (NumPy / SciPy)
│   ├── ann_classifier.py   MLPClassifier wrapper
│   ├── trainer.py          Full training orchestrator
│   ├── evaluator.py        Metrics, reports, CSV
│   ├── predictor.py        Single-image inference
│   ├── visualization.py    All matplotlib plots
│   └── utils.py            Logging, serialisation, timer
│
├── models/
│   ├── pca_model.pkl
│   ├── ann_model.pkl
│   └── label_encoder.pkl
│
├── outputs/
│   ├── mean_face.png
│   ├── eigenface_1.png … eigenface_10.png
│   ├── eigenfaces_grid.png
│   ├── accuracy_vs_k.png
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── results.csv
│
├── train.py            Main training entry point
├── predict.py          CLI inference
├── sanity_check.py     Component verification
├── config.py           All hyperparameters & paths
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 9 · Installation

```bash
# Clone / download the project, then:
cd FaceRecognitionPCAANN
pip install -r requirements.txt
```

---

## 10 · Training Instructions

```bash
python train.py
```

This will:
1. Discover and load all face images
2. Preprocess: grayscale → 64×64 → normalise → flatten
3. 60/40 stratified split
4. Sweep k ∈ {5, 10, 20, 30, 40, 50, 75, 100}
5. Select best k, fit final PCA + ANN
6. Save models, plots, and reports

---

## 11 · Prediction Instructions

```bash
python predict.py dataset/faces/Alia/face_1.jpg
```

Output (known person):
```
────────────────────────────────────────
Predicted Person: Alia
Confidence: 97.2%
────────────────────────────────────────
```

Output (unknown person):
```
────────────────────────────────────────
Unknown Person
Confidence: 25.1%
────────────────────────────────────────
```

---

## 12 · Output Examples

| File                        | Description                                |
|-----------------------------|--------------------------------------------|
| `mean_face.png`             | Average face across all training images    |
| `eigenface_1.png` … `10.png`| Most significant principal directions      |
| `eigenfaces_grid.png`       | All eigenfaces in a single grid            |
| `accuracy_vs_k.png`         | Accuracy curve with best-k highlighted     |
| `confusion_matrix.png`      | Normalised confusion matrix heatmap        |
| `classification_report.txt` | Per-class precision / recall / F1          |
| `results.csv`               | Tabular k → accuracy / precision / recall / F1 |

---

## 13 · Accuracy vs k Analysis

| k   | Expected Behaviour                          |
|-----|---------------------------------------------|
| 5   | Very low accuracy — too few components       |
| 20–50 | Sweet spot — best accuracy / complexity   |
| 100 | Potential overfitting / diminishing returns  |

The system automatically identifies and reports the best k.

---

## 14 · Unknown Person Detection

The unknown-person detector uses the ANN's softmax output probabilities:

```python
if max(class_probabilities) < UNKNOWN_THRESHOLD:
    prediction = "Unknown Person"
```

**Threshold selection:** 0.55 (configurable in `config.py`).  
A threshold of 0.55 means the network must be at least 55 % confident
to make a positive identification.  Lower values increase recall at the
cost of more false positives; higher values do the opposite.

---

## 15 · Sanity Check

```bash
python sanity_check.py
```

Verifies all five major components before a full training run.

---

## 16 · Future Improvements

- **LFW / aligned face datasets** — larger, more challenging benchmarks
- **CNN features** — replace PCA with deep feature extraction
- **Fisher Faces (LDA)** — class-discriminative subspace
- **Data augmentation** — horizontal flip, brightness jitter
- **ONNX export** — deploy the trained model cross-platform
- **REST API** — serve predictions via FastAPI
