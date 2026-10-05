<div align="center">

# Deep Learning-Based Biomedical Image Segmentation
### Comparative Benchmarking of Baseline U-Net, SE U-Net, and CBAM U-Net on Endoscopic Polyp Datasets

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.x-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](docs/roadmap_and_contributing.md)

<p align="center">
  <b>A research-focused framework for pixel-accurate lesion boundary delineation in gastrointestinal colonoscopy imaging using channel and spatial attention mechanisms.</b>
</p>

[📚 Architecture Theory](docs/architectures.md) •
[📁 Dataset Pipeline](docs/dataset.md) •
[🏋️ Training & Losses](docs/training_and_losses.md) •
[📊 Metrics & Evaluation](docs/metrics_and_evaluation.md) •
[🔮 Roadmap & Contributing](docs/roadmap_and_contributing.md)

</div>

---

## 📌 Executive Summary

Accurate colorectal polyp segmentation from endoscopic video feeds is a cornerstone of Computer-Aided Diagnosis (CADx) systems for colorectal cancer prevention. However, lesions exhibit significant variations in morphology, flat mucosal transitions, and endoscopic illumination glare.

This repository implements, trains, and empirically benchmarks three deep learning architectures on the **Kvasir-SEG** dataset:
1. **Baseline U-Net** — Standard encoder-decoder architecture with symmetric skip connections.
2. **SE U-Net** — U-Net augmented with **Squeeze-and-Excitation (SE)** channel attention to dynamically weight informative feature channels.
3. **CBAM U-Net** — U-Net augmented with **Convolutional Block Attention Modules (CBAM)** combining sequential channel and spatial attention to pinpoint *what* features and *where* lesions are located.

---

## 📖 Complete Documentation Index

For exhaustive theoretical derivations, loss formulation mathematics, dataset statistics, and guidelines, refer to the dedicated sub-documents:

| Document | Primary Topics |
| :--- | :--- |
| 🧠 [**docs/architectures.md**](docs/architectures.md) | DoubleConv block structure, SE channel recalibration derivations, CBAM channel+spatial formulation, parameter comparisons. |
| 📁 [**docs/dataset.md**](docs/dataset.md) | Clinical background on polyp screening, Kvasir-SEG filesystem structure, deterministic 704/120/176 splitting, bilinear/nearest-neighbor transformations, online data augmentations. |
| 🏋️ [**docs/training_and_losses.md**](docs/training_and_losses.md) | Class imbalance mechanics, Binary Cross-Entropy with Logits, Soft Dice Loss derivation, composite loss objective, Adam optimizer settings, and checkpointing. |
| 📊 [**docs/metrics_and_evaluation.md**](docs/metrics_and_evaluation.md) | Sample-wise vs dataset-wide aggregation, Confusion Matrix derivation, Dice score (F1), IoU (Jaccard), Precision, Recall, and the 5-panel comparative visualization tool. |
| 🔮 [**docs/roadmap_and_contributing.md**](docs/roadmap_and_contributing.md) | Architectural roadmap (Attention U-Net, UNet++, TransUNet, ONNX export), GitHub PR workflow, coding guidelines, and academic BibTeX citations. |

---

## 🧠 Architectural Overview

```text
                                END-TO-END PIPELINE

    ┌─────────────────┐       ┌────────────────────────┐       ┌──────────────────────┐
    │  Kvasir-SEG     │       │  Preprocessing &       │       │  Architectures       │
    │  Endoscopy RGB  │ ────► │  Online Augmentation   │ ────► │  • Baseline U-Net    │
    │  (256 x 256)    │       │  (H-Flip / V-Flip)     │       │  • SE U-Net          │
    └─────────────────┘       └────────────────────────┘       │  • CBAM U-Net        │
                                                               └──────────┬───────────┘
                                                                          │
                                                                          ▼
    ┌─────────────────┐       ┌────────────────────────┐       ┌──────────────────────┐
    │ Evaluation &    │       │ Metric Computation     │       │ Combined Objective   │
    │ Multi-Model     │ ◄──── │ • Dice / F1-Score      │ ◄──── │ • BCE With Logits    │
    │ 5-Panel Display │       │ • IoU / Precision / Rec│       │ • Soft Dice Loss     │
    └─────────────────┘       └────────────────────────┘       └──────────────────────┘
```

### Model Comparison Matrix

| Architecture | Attention Type | Focus Mechanism | Key Advantage |
| :--- | :--- | :--- | :--- |
| **U-Net** | None | Uniform spatial & channel convolutions | Low computational overhead, solid baseline |
| **SE U-Net** | Channel Attention | Global Avg Pool $\rightarrow$ 2-layer MLP $\rightarrow$ Channel Recalibration | Suppresses non-informative background channels |
| **CBAM U-Net** | Channel + Spatial | Dual Pooling (Avg+Max) MLP + $7\times 7$ Convolution | Pinpoints both lesion features and exact spatial coordinates |

👉 *Full mathematical formulations and block diagrams are detailed in [**docs/architectures.md**](docs/architectures.md).*

---

## 📂 Project Structure

```text
Biomedical-Segmentation/
├── docs/                                  # In-depth technical documentation
│   ├── architectures.md                   # Mathematical derivations of U-Net, SE, CBAM
│   ├── dataset.md                         # Kvasir-SEG layout, preprocessing & augmentation
│   ├── training_and_losses.md             # Combined BCE + Dice loss and training mechanics
│   ├── metrics_and_evaluation.md          # Quantitative metrics and comparative visual evaluation
│   └── roadmap_and_contributing.md        # Development roadmap, PR guide, BibTeX citations
│
├── models/                                # PyTorch model definitions
│   ├── unet.py                            # Baseline U-Net architecture
│   ├── se_unet.py                         # Squeeze-and-Excitation U-Net
│   └── cbam_unet.py                       # Convolutional Block Attention Module U-Net
│
├── src/                                   # Data pipeline, training & evaluation scripts
│   ├── dataset.py                         # KvasirDataset loader, splits (704/120/176), transforms
│   ├── losses.py                          # Soft Dice loss and combined BCE + Dice loss
│   ├── check_dataset.py                   # Integrity check for image & mask pair counts
│   ├── analyze_dataset.py                 # Dimension distribution and pixel integrity analysis
│   ├── visualize_dataset.py               # Ground-truth sample visualizer
│   ├── train.py                           # Training pipeline for baseline U-Net
│   ├── train_se_unet.py                   # Training pipeline for SE U-Net
│   ├── train_cbam_unet.py                 # Training pipeline for CBAM U-Net
│   ├── evaluate.py                        # Quantitative evaluation for baseline U-Net
│   ├── evaluate_se_unet.py                # Quantitative evaluation for SE U-Net
│   ├── evaluate_cbam_unet.py              # Quantitative evaluation for CBAM U-Net
│   ├── compare_models.py                  # Side-by-side 5-panel inference visualizer
│   ├── visualize_prediction.py            # Prediction visualization for U-Net
│   ├── visualize_se_unet.py               # Prediction visualization for SE U-Net
│   └── visualize_cbam_unet.py             # Prediction visualization for CBAM U-Net
│
├── results/                               # Saved checkpoints & loss history (local, git-ignored)
├── dataset/                               # Kvasir-SEG dataset root (local, git-ignored)
├── .gitignore                             # Ignores .venv, dataset/, checkpoints (*.pth)
└── README.md                              # Main project documentation hub
```

---

## ⚡ Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/Jeffrin0507/Biomedical-Segmentation.git
cd Biomedical-Segmentation
```

### 2. Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install PyTorch with CUDA support (adjust for your CUDA runtime)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install numpy matplotlib opencv-python pillow scikit-learn tqdm
```

### 3. Dataset Placement
Download the [Kvasir-SEG dataset](https://datasets.simula.no/kvasir-seg/) and unpack it under `dataset/`:
```text
dataset/
└── Kvasir-SEG/
    ├── train.txt
    ├── val.txt
    └── Kvasir-SEG/Kvasir-SEG/
        ├── images/
        └── masks/
```

Verify your dataset configuration:
```bash
python src/check_dataset.py
python src/analyze_dataset.py
```
👉 *See [**docs/dataset.md**](docs/dataset.md) for full dataset specifications.*

---

## 🏋️ Model Training

Train each architecture using the optimized default hyperparameters (Batch Size: 8, Adam Optimizer, Learning Rate: $1\times 10^{-4}$, 20 Epochs, Input Size: $256 \times 256$):

```bash
# 1. Baseline U-Net
python src/train.py

# 2. SE U-Net (Channel Attention)
python src/train_se_unet.py

# 3. CBAM U-Net (Channel + Spatial Attention)
python src/train_cbam_unet.py
```

Model checkpoints and loss histories are automatically saved into `results/`:
- `results/<model>_best.pth` (minimum validation loss weights)
- `results/<model>_final.pth` (epoch 20 weights)
- `results/<model>_history.pth` (loss trajectories)

👉 *Detailed optimization mechanics are covered in [**docs/training_and_losses.md**](docs/training_and_losses.md).*

---

## 📊 Evaluation & Benchmarking

Evaluate the held-out test set (176 images) with sample-wise metric calculation:

```bash
# Evaluate Baseline U-Net
python src/evaluate.py

# Evaluate SE U-Net
python src/evaluate_se_unet.py

# Evaluate CBAM U-Net
python src/evaluate_cbam_unet.py
```

### Quantitative Results Reference

| Model Architecture | Dice Score (F1) ↑ | IoU (Jaccard) ↑ | Precision ↑ | Recall (Sensitivity) ↑ | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline U-Net** | `0.792` | `0.704` | `0.812` | `0.810` | 31.04 M |
| **SE U-Net** | `0.818` | `0.728` | `0.839` | `0.829` | 31.09 M |
| **CBAM U-Net** | `0.836` | `0.749` | `0.862` | `0.842` | 31.11 M |

*Sample-wise averaging guarantees that evaluation is not disproportionately biased toward larger lesions.*

👉 *Metric derivations and analysis are documented in [**docs/metrics_and_evaluation.md**](docs/metrics_and_evaluation.md).*

---

## 🖼️ Multi-Model Visual Comparison

To qualitatively inspect segmentation fidelity across all three models on fixed reproducible test samples:

```bash
python src/compare_models.py
```

This generates a side-by-side 5-panel figure for clinical samples:
1. **Original Endoscopy Image** (Normalized RGB)
2. **Ground Truth Annotation** (Expert Gastroenterologist Mask)
3. **U-Net Prediction** (+ Sample Dice score)
4. **SE U-Net Prediction** (+ Sample Dice score)
5. **CBAM U-Net Prediction** (+ Sample Dice score)

To visualize single model inference masks:
```bash
python src/visualize_prediction.py    # U-Net
python src/visualize_se_unet.py       # SE U-Net
python src/visualize_cbam_unet.py     # CBAM U-Net
```

---

## 🔮 Roadmap & Contributing

We welcome community contributions, architectural improvements, and bug fixes!
- [x] Baseline U-Net with skip connections
- [x] Squeeze-and-Excitation channel attention integration
- [x] CBAM dual channel-and-spatial attention integration
- [ ] Attention U-Net with additive gating
- [ ] UNet++ nested dense skip pathways
- [ ] Vision Transformer (TransUNet / Swin-Unet) backbones
- [ ] ONNX / TensorRT export for real-time video stream inference

👉 *Check out [**docs/roadmap_and_contributing.md**](docs/roadmap_and_contributing.md) for PR guidelines, coding conventions, and BibTeX citations.*

---

## 👨‍💻 Authors & Acknowledgments

- **Jeffrin Jose** — Original Author & Architecture Implementations ([@Jeffrin0507](https://github.com/Jeffrin0507))
- **Jerome Antony Robin** — Documentation Architecture & Benchmarking Enhancements ([@JeromeAntonyRobin](https://github.com/JeromeAntonyRobin))

Special thanks to the **Simula Research Laboratory** for curating and publishing the open-access [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) dataset.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
