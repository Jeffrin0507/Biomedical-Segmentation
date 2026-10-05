# Training Pipeline, Optimization & Loss Formulations

This document describes the mathematical formulation of the loss functions, the optimization configuration, checkpointing mechanisms, and training procedures across all three models.

---

## Table of Contents
- [1. Loss Functions & Objective Formulations](#1-loss-functions--objective-formulations)
  - [1.1 The Class Imbalance Challenge in Polyp Segmentation](#11-the-class-imbalance-challenge-in-polyp-segmentation)
  - [1.2 Binary Cross-Entropy with Logits (BCE)](#12-binary-cross-entropy-with-logits-bce)
  - [1.3 Soft Dice Loss](#13-soft-dice-loss)
  - [1.4 Combined BCE + Dice Loss](#14-combined-bce--dice-loss)
- [2. Optimization Strategy & Hyperparameters](#2-optimization-strategy--hyperparameters)
- [3. Execution Procedures](#3-execution-procedures)
  - [3.1 Baseline U-Net Training](#31-baseline-u-net-training)
  - [3.2 SE U-Net Training](#32-se-u-net-training)
  - [3.3 CBAM U-Net Training](#33-cbam-u-net-training)
- [4. Model Checkpointing & History Artifacts](#4-model-checkpointing--history-artifacts)
- [5. Recommended Hyperparameter Tuning & Enhancements](#5-recommended-hyperparameter-tuning--enhancements)

---

## 1. Loss Functions & Objective Formulations

Implemented in: [`src/losses.py`](../src/losses.py)

### 1.1 The Class Imbalance Challenge in Polyp Segmentation
In endoscopic polyp segmentation, polyps frequently occupy only a small fraction of the image area (often 5–15% of the total pixel count). Training with standard pixel-level Cross-Entropy alone tends to bias the model heavily toward background predictions, resulting in high overall pixel accuracy but poor lesion recall.

To address this, our training objective combines a **distribution-based loss** (Binary Cross-Entropy) with a **region-based overlap loss** (Soft Dice Loss).

### 1.2 Binary Cross-Entropy with Logits (BCE)
Binary Cross-Entropy penalizes pixel-level misclassifications. Rather than calculating $\sigma(\mathbf{z})$ followed by $\log(\cdot)$, PyTorch's `binary_cross_entropy_with_logits` combines the operations using the log-sum-exp trick for numerical stability:

$$\mathcal{L}_{\text{BCE}}(\mathbf{z}, \mathbf{y}) = -\frac{1}{N} \sum_{i=1}^N \Big[ y_i \cdot \log(\sigma(z_i)) + (1 - y_i) \cdot \log(1 - \sigma(z_i)) \Big]$$

Where:
- $z_i$ is the unnormalized logit output for pixel $i$.
- $y_i \in \{0, 1\}$ is the ground-truth binary label.
- $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid activation.

### 1.3 Soft Dice Loss
The Soft Dice Loss directly maximizes the spatial overlap between the predicted probability map $\mathbf{p} = \sigma(\mathbf{z})$ and the ground-truth target $\mathbf{y}$:

$$\text{Dice}(\mathbf{p}, \mathbf{y}) = \frac{2 \sum_{i=1}^N p_i y_i + \epsilon}{\sum_{i=1}^N p_i + \sum_{i=1}^N y_i + \epsilon}$$

$$\mathcal{L}_{\text{Dice}} = 1 - \text{Dice}(\mathbf{p}, \mathbf{y})$$

Where:
- $\epsilon = 1 \times 10^{-6}$ is a smoothing coefficient that avoids zero-division when an image contains no lesion.
- Unlike hard thresholding, treating $p_i \in [0, 1]$ as continuous probabilities produces smooth gradients throughout backpropagation.

### 1.4 Combined BCE + Dice Loss
The composite loss is defined as:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(\mathbf{z}, \mathbf{y}) + \mathcal{L}_{\text{Dice}}(\sigma(\mathbf{z}), \mathbf{y})$$

- $\mathcal{L}_{\text{BCE}}$ provides smooth gradient descent at early training stages and ensures calibrated pixel probabilities.
- $\mathcal{L}_{\text{Dice}}$ balances foreground and background penalization, guaranteeing robust lesion segmentation regardless of polyp scale.

---

## 2. Optimization Strategy & Hyperparameters

| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| **Optimizer** | Adam | Adaptive moment estimation stabilizes training on attention layers |
| **Learning Rate** | $1 \times 10^{-4}$ | Prevents gradient explosion in transposed convolution layers |
| **Batch Size** | 8 | Fits comfortably in 6GB/8GB VRAM (e.g. RTX 4050) without OOM |
| **Epochs** | 20 | Empirically reaches convergence on Kvasir-SEG (704 train samples) |
| **Input Resolution** | $256 \times 256 \times 3$ | Balanced speed and spatial detail |
| **Device Execution** | CUDA / GPU | Automatic fallback to CPU if CUDA is unavailable |
| **Data Transfer** | `pin_memory=True`, `non_blocking=True` | Asynchronous host-to-device memory copies |

---

## 3. Execution Procedures

Ensure the dataset is placed under `dataset/` (see [Dataset Guide](dataset.md)).

### 3.1 Baseline U-Net Training
To train the baseline encoder-decoder model:
```bash
python src/train.py
```

### 3.2 SE U-Net Training
To train the channel-attention-enhanced model:
```bash
python src/train_se_unet.py
```

### 3.3 CBAM U-Net Training
To train the combined channel-and-spatial attention model:
```bash
python src/train_cbam_unet.py
```

During execution, training and validation loss are logged at every epoch, and the best-performing checkpoint on the validation split is automatically persisted.

---

## 4. Model Checkpointing & History Artifacts

Checkpoints are stored in the local `results/` directory:

```text
results/
├── unet_best.pth            # Weights yielding lowest validation loss (U-Net)
├── unet_final.pth           # Final epoch weights (U-Net)
├── unet_history.pth         # Dictionary containing train_loss and val_loss curves
├── se_unet_best.pth         # Best checkpoint for SE U-Net
├── se_unet_final.pth        # Final checkpoint for SE U-Net
├── se_unet_history.pth      # Loss history for SE U-Net
├── cbam_unet_best.pth       # Best checkpoint for CBAM U-Net
├── cbam_unet_final.pth      # Final checkpoint for CBAM U-Net
└── cbam_unet_history.pth    # Loss history for CBAM U-Net
```

To plot training convergence histories:
```python
import torch
import matplotlib.pyplot as plt

history = torch.load("results/cbam_unet_history.pth")
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("CBAM U-Net Loss Curve")
plt.show()
```

---

## 5. Recommended Hyperparameter Tuning & Enhancements

When scaling experiments, consider the following enhancements:
1. **Learning Rate Scheduler**:
   Add a `ReduceLROnPlateau` or `CosineAnnealingLR` scheduler to smoothly anneal the learning rate when validation loss plateaus:
   ```python
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
   ```
2. **Mixed Precision Training (AMP)**:
   Leverage `torch.cuda.amp.autocast()` and `GradScaler` to double training throughput and decrease VRAM usage by ~40% on modern RTX GPUs.
3. **Focal Loss Weighting**:
   For extremely small lesions (<2% of image area), explore adding $\alpha$-balanced focal loss to further penalize hard boundary errors.

---

Continue reading:
- [Architectural Deep Dive](architectures.md)
- [Dataset Pipeline Documentation](dataset.md)
- [Metrics & Evaluation Guide](metrics_and_evaluation.md)
