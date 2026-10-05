# Evaluation Metrics & Comparative Analysis

This document provides a thorough explanation of the quantitative evaluation metrics used in this project, the difference between sample-wise and dataset-level aggregation, and instructions for running comparative evaluations across all models.

---

## Table of Contents
- [1. Quantitative Evaluation Metrics](#1-quantitative-evaluation-metrics)
  - [1.1 Confusion Matrix Formulation](#11-confusion-matrix-formulation)
  - [1.2 Dice Similarity Coefficient (DSC) / F1-Score](#12-dice-similarity-coefficient-dsc--f1-score)
  - [1.3 Intersection over Union (IoU) / Jaccard Index](#13-intersection-over-union-iou--jaccard-index)
  - [1.4 Precision (Positive Predictive Value)](#14-precision-positive-predictive-value)
  - [1.5 Recall (Sensitivity / True Positive Rate)](#15-recall-sensitivity--true-positive-rate)
- [2. Sample-Wise vs. Batch-Averaged Evaluation](#2-sample-wise-vs-batch-averaged-evaluation)
- [3. Running Evaluation Scripts](#3-running-evaluation-scripts)
  - [3.1 Baseline U-Net Evaluation](#31-baseline-u-net-evaluation)
  - [3.2 SE U-Net Evaluation](#32-se-u-net-evaluation)
  - [3.3 CBAM U-Net Evaluation](#33-cbam-u-net-evaluation)
- [4. Multi-Model Visual Comparison (`compare_models.py`)](#4-multi-model-visual-comparison-compare_modelspy)
  - [4.1 Deterministic Test Sampling](#41-deterministic-test-sampling)
  - [4.2 5-Panel Visual Display Layout](#42-5-panel-visual-display-layout)
- [5. Benchmark Results Reference Template](#5-benchmark-results-reference-template)

---

## 1. Quantitative Evaluation Metrics

Implemented in: [`src/evaluate.py`](../src/evaluate.py#L75-L111)

### 1.1 Confusion Matrix Formulation
For each test sample, the predicted continuous logit output is transformed into a binary mask via Sigmoid activation followed by a decision threshold of $\tau = 0.5$:

$$\hat{y}_i = \begin{cases} 1 & \text{if } \sigma(z_i) \ge 0.5 \\ 0 & \text{otherwise} \end{cases}$$

Pixel-wise comparisons with the ground-truth binary mask $y_i$ yield four fundamental confusion matrix counts:
- **True Positives (TP)**: Pixels correctly identified as polyp: $(\hat{y}_i = 1) \land (y_i = 1)$
- **False Positives (FP)**: Healthy mucosal pixels incorrectly segmented as polyp: $(\hat{y}_i = 1) \land (y_i = 0)$
- **False Negatives (FN)**: Actual polyp pixels missed by the model: $(\hat{y}_i = 0) \land (y_i = 1)$
- **True Negatives (TN)**: Background mucosal pixels correctly identified as non-polyp: $(\hat{y}_i = 0) \land (y_i = 0)$

An infinitesimal epsilon $\epsilon = 1 \times 10^{-7}$ is included across all denominators to safeguard against zero-division.

---

### 1.2 Dice Similarity Coefficient (DSC) / F1-Score
The Dice Similarity Coefficient is the standard primary metric in biomedical image segmentation benchmarks. It evaluates the harmonic mean of precision and recall:

$$\text{Dice} = \frac{2 \times \text{TP}}{2 \times \text{TP} + \text{FP} + \text{FN} + \epsilon} = \frac{2 |\hat{\mathbf{Y}} \cap \mathbf{Y}|}{|\hat{\mathbf{Y}}| + |\mathbf{Y}| + \epsilon}$$

- **Range**: $[0, 1]$, where $1.0$ indicates perfect spatial overlap.
- **Clinical Significance**: Heavily penalizes false positives and false negatives while being insensitive to large background areas (True Negatives).

---

### 1.3 Intersection over Union (IoU) / Jaccard Index
The Jaccard Index measures the ratio of the overlap area to the total union area:

$$\text{IoU} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN} + \epsilon} = \frac{|\hat{\mathbf{Y}} \cap \mathbf{Y}|}{|\hat{\mathbf{Y}} \cup \mathbf{Y}| + \epsilon}$$

Relation to Dice:
$$\text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}}, \quad \text{Dice} = \frac{2 \times \text{IoU}}{1 + \text{IoU}}$$
Because $\text{IoU} \le \text{Dice}$, it imposes a stricter penalty on boundary discrepancies.

---

### 1.4 Precision (Positive Predictive Value)
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP} + \epsilon}$$
- **Interpretation**: Out of all pixels the model classified as a polyp, what percentage actually belongs to a polyp? High precision minimizes spurious false alarms.

---

### 1.5 Recall (Sensitivity / True Positive Rate)
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN} + \epsilon}$$
- **Interpretation**: Out of all true polyp pixels present in the patient's endoscopy image, what percentage did the model capture? In clinical screening, **high recall is critical** to prevent missed lesions.

---

## 2. Sample-Wise vs. Batch-Averaged Evaluation

A critical design choice in [`src/evaluate.py`](../src/evaluate.py#L157-L177) is calculating metrics **sample-by-sample** rather than accumulating TP, FP, and FN across the entire test dataset:

$$\text{Dice}_{\text{dataset}} = \frac{1}{M} \sum_{k=1}^M \text{Dice}(\hat{\mathbf{Y}}_k, \mathbf{Y}_k)$$

### Why Sample-Wise Averaging is Superior:
1. **Unbiased by Polyp Size**: Global accumulation allows large polyps (occupying hundreds of thousands of pixels) to dominate the metric, masking complete failures on small, early-stage polyps. Sample-wise averaging treats every patient image with equal weight.
2. **Standard Benchmark Compliance**: MICCAI and leading biomedical journals mandate sample-wise mean and standard deviation reporting.

---

## 3. Running Evaluation Scripts

All evaluation scripts evaluate the test partition (176 images) using the best saved weights.

### 3.1 Baseline U-Net Evaluation
```bash
python src/evaluate.py
```

### 3.2 SE U-Net Evaluation
```bash
python src/evaluate_se_unet.py
```

### 3.3 CBAM U-Net Evaluation
```bash
python src/evaluate_cbam_unet.py
```

Example terminal output:
```text
================================
      CBAM U-NET TEST RESULTS
     SAMPLE-WISE AVERAGE
================================
Test Samples : 176
Dice Score   : 0.8354
IoU          : 0.7482
Precision    : 0.8621
Recall       : 0.8415
================================
```

---

## 4. Multi-Model Visual Comparison (`compare_models.py`)

Implemented in: [`src/compare_models.py`](../src/compare_models.py)

To qualitatively inspect how attention mechanisms alter boundary delineations, the comparison script executes side-by-side inference across all three architectures.

```bash
python src/compare_models.py
```

### 4.1 Deterministic Test Sampling
To ensure reproducibility across research presentations, fixed test sample indices are evaluated:
```python
TEST_INDICES = [35, 32, 68, 107, 160]
```
These indices encompass varied clinical scenarios:
- Small isolated lesions
- Large irregularly shaped adenomas
- High-glare reflections
- Low-contrast mucosal surfaces

### 4.2 5-Panel Visual Display Layout
For each test sample, a 5-panel figure is generated:
1. **Original Endoscopy Image**: Normalized RGB frame.
2. **Ground Truth Mask**: Expert gastroenterologist annotation.
3. **U-Net Prediction**: Baseline segmentation output with sample Dice score.
4. **SE U-Net Prediction**: Channel-attention segmentation output with sample Dice score.
5. **CBAM U-Net Prediction**: Channel-and-spatial attention segmentation output with sample Dice score.

---

## 5. Benchmark Results Reference Template

| Architecture | Dice Score | IoU (Jaccard) | Precision | Recall | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **U-Net (Baseline)** | `0.792` | `0.704` | `0.812` | `0.810` | 31.04 M |
| **SE U-Net** | `0.818` | `0.728` | `0.839` | `0.829` | 31.09 M |
| **CBAM U-Net** | `0.836` | `0.749` | `0.862` | `0.842` | 31.11 M |

*(Note: Exact values reflect weights trained under default hyperparameters; evaluate your local checkpoints for exact figures.)*

---

Continue reading:
- [Architectural Deep Dive](architectures.md)
- [Dataset Pipeline Documentation](dataset.md)
- [Training & Loss Formulations](training_and_losses.md)
- [Roadmap & Contributing Guidelines](roadmap_and_contributing.md)
