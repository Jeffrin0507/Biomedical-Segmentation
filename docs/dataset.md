# Dataset Guide: Kvasir-SEG & Data Pipeline

This document provides a complete guide to the **Kvasir-SEG** gastrointestinal polyp dataset used in this project, including clinical background, directory structure, data preprocessing, augmentation pipelines, and diagnostic utilities.

---

## Table of Contents
- [1. Clinical Context & Dataset Overview](#1-clinical-context--dataset-overview)
- [2. Dataset Filesystem Hierarchy](#2-dataset-filesystem-hierarchy)
- [3. Train / Validation / Test Splitting Strategy](#3-train--validation--test-splitting-strategy)
- [4. Data Preprocessing & Augmentation Pipeline](#4-data-preprocessing--augmentation-pipeline)
  - [4.1 Image Normalization](#41-image-normalization)
  - [4.2 Mask Binarization & Interpolation Precision](#42-mask-binarization--interpolation-precision)
  - [4.3 Online Data Augmentations](#43-online-data-augmentations)
- [5. Dataset Diagnostics & Validation Scripts](#5-dataset-diagnostics--validation-scripts)
  - [5.1 Sanity Checking (`check_dataset.py`)](#51-sanity-checking-check_datasetpy)
  - [5.2 Dimension & Integrity Analysis (`analyze_dataset.py`)](#52-dimension--integrity-analysis-analyze_datasetpy)
  - [5.3 Visual Ground Truth Inspection (`visualize_dataset.py`)](#53-visual-ground-truth-inspection-visualize_datasetpy)

---

## 1. Clinical Context & Dataset Overview

**Colorectal Cancer (CRC)** is the third most common cancer globally and one of the leading causes of cancer-related mortality. Most colorectal cancers originate as benign adenomatous polyps in the mucosa lining the colon and rectum. 

During colonoscopy procedures, early detection and full mucosal resection of polyps drastically reduces colorectal cancer incidence. However, colonoscopic inspection is prone to polyp miss-rates as high as 20–25% due to:
- Visual similarity between subtle flat lesions and surrounding healthy mucosa.
- Visual artifacts such as glare from liquid, camera motion blur, and blood vessels.
- Fatigue and cognitive load on gastroenterologists.

The **Kvasir-SEG** dataset is an open-access benchmark released by Simula Research Laboratory (Norway), containing **1,000 high-resolution endoscopy frames** and their corresponding pixel-accurate ground-truth masks verified by expert gastroenterologists.

---

## 2. Dataset Filesystem Hierarchy

The dataset files are excluded from git via [`.gitignore`](../.gitignore) to keep the repository lightweight. To use the codebase, download the Kvasir-SEG dataset from the official repository and organize it as follows:

```text
Biomedical-Segmentation/
└── dataset/
    └── Kvasir-SEG/
        ├── train.txt                           # Official training list (880 IDs)
        ├── val.txt                             # Official validation list (120 IDs)
        └── Kvasir-SEG/
            └── Kvasir-SEG/
                ├── images/
                │   ├── cju0qoxqj9q6s0835b43399p4.jpg
                │   ├── cju0qx73c9qjx0835aguntajh.jpg
                │   └── ... (1,000 images)
                └── masks/
                    ├── cju0qoxqj9q6s0835b43399p4.jpg
                    ├── cju0qx73c9qjx0835aguntajh.jpg
                    └── ... (1,000 masks)
```

> **Note on path structure**: The nested `dataset/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG` path matches the standard uncompressed folder structure emitted by the default Kvasir-SEG archive. This configuration is centralized in [`src/dataset.py`](../src/dataset.py#L15-L25).

---

## 3. Train / Validation / Test Splitting Strategy

The original Kvasir-SEG split defines 880 training and 120 validation samples. To maintain strict evaluation integrity and evaluate true out-of-distribution generalization, [`src/dataset.py`](../src/dataset.py#L58-L84) splits the 880 training IDs into a 80/20 train/test partition using a deterministic random seed:

$$\text{Total: 1,000 Samples}$$
- **Training Set**: 704 samples (~70.4%)
- **Validation Set**: 120 samples (12.0%, official split)
- **Held-Out Test Set**: 176 samples (17.6%)

```python
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
random.shuffle(train_ids)

test_ids = train_ids[:176]
final_train_ids = train_ids[176:]
```

Because the seed is fixed to `42`, every model evaluation and test comparison is guaranteed to run against the identical sample partition.

---

## 4. Data Preprocessing & Augmentation Pipeline

The PyTorch dataset loader is implemented as `KvasirDataset` in [`src/dataset.py`](../src/dataset.py#L90-L208).

### 4.1 Image Normalization
Input images are transformed from uint8 RGB arrays $[0, 255]$ into floating point tensors $\mathbf{X} \in [-1.0, 1.0]$:

$$\mathbf{X}_{\text{norm}} = \frac{\mathbf{X} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$$
Where $\boldsymbol{\mu} = [0.5, 0.5, 0.5]$ and $\boldsymbol{\sigma} = [0.5, 0.5, 0.5]$.

### 4.2 Mask Binarization & Interpolation Precision
To prevent blurred or non-binary values along boundary pixels during resizing:
1. **Images** are resized to $256 \times 256$ using `InterpolationMode.BILINEAR` for smooth feature interpolation.
2. **Masks** are resized using `InterpolationMode.NEAREST` to preserve crisp categorical edges.
3. Masks are thresholded at intensity 128:
   $$M(x, y) = \begin{cases} 1.0 & \text{if } I_{\text{mask}}(x, y) \ge 128 \\ 0.0 & \text{otherwise} \end{cases}$$
4. An explicit channel dimension is appended, yielding a target shape of `(1, 256, 256)`.

### 4.3 Online Data Augmentations
To prevent overfitting on the 704 training images, synchronized geometric augmentations are applied on-the-fly:
- **Random Horizontal Flip**: $p = 0.5$, mirroring both image and mask horizontally.
- **Random Vertical Flip**: $p = 0.5$, mirroring both image and mask vertically.

---

## 5. Dataset Diagnostics & Validation Scripts

Before launching training runs, verify dataset integrity using the built-in diagnostic tools.

### 5.1 Sanity Checking (`check_dataset.py`)
Run:
```bash
python src/check_dataset.py
```
This script confirms:
- The presence of the `images` and `masks` directories.
- An equal count of 1,000 images and 1,000 masks.
- 1:1 filename matching between pairs.

### 5.2 Dimension & Integrity Analysis (`analyze_dataset.py`)
Run:
```bash
python src/analyze_dataset.py
```
Outputs statistical distributions of raw image resolutions (varying from $332 \times 487$ to $1920 \times 1072$), confirms that raw image and mask dimensions match per instance, and checks for corrupted files.

### 5.3 Visual Ground Truth Inspection (`visualize_dataset.py`)
Run:
```bash
python src/visualize_dataset.py
```
Renders a Matplotlib window showing a representative endoscopy image side-by-side with its ground-truth polyp boundary mask.

---

Continue reading:
- [Architectural Deep Dive](architectures.md)
- [Training & Loss Formulations](training_and_losses.md)
- [Evaluation & Metrics Breakdown](metrics_and_evaluation.md)
