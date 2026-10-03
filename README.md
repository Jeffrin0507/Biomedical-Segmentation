# Deep Learning-Based Biomedical Image Segmentation Using U-Net and Attention Mechanisms

A deep learning project for biomedical image segmentation using a baseline **U-Net** architecture and attention-enhanced variants based on **Squeeze-and-Excitation (SE)** and **Convolutional Block Attention Module (CBAM)**.

## 📌 Project Overview

Biomedical image segmentation is the process of identifying and separating regions of interest from medical images at the pixel level. Accurate segmentation can support downstream tasks such as disease analysis, measurement, and computer-aided diagnosis.

This project implements and compares three segmentation architectures:

- **U-Net** — baseline encoder-decoder segmentation network
- **SE U-Net** — U-Net enhanced with Squeeze-and-Excitation channel attention
- **CBAM U-Net** — U-Net enhanced with Convolutional Block Attention Module

The objective is to investigate whether attention mechanisms can improve segmentation performance compared with the baseline U-Net.

---

## 🧠 Architectures

### 1. U-Net

The baseline U-Net follows the standard encoder-decoder structure:

```text
Input Image
     │
     ▼
┌─────────────┐
│   Encoder   │
└─────────────┘
     │
     │ Skip Connections
     ▼
┌─────────────┐
│   Decoder   │
└─────────────┘
     │
     ▼
Segmentation Mask
```

The encoder extracts hierarchical image features while the decoder progressively reconstructs the spatial resolution. Skip connections preserve fine-grained spatial information.

### 2. SE U-Net

SE U-Net incorporates **Squeeze-and-Excitation blocks** to recalibrate channel-wise feature responses.

The SE mechanism:

1. Squeezes spatial information using global pooling
2. Learns channel dependencies
3. Excites informative channels using learned weights
4. Reweights feature maps

This allows the network to emphasize channels that contribute more strongly to segmentation.

### 3. CBAM U-Net

CBAM introduces two complementary attention mechanisms:

- **Channel Attention**
- **Spatial Attention**

The module learns both:

```text
Which features are important?
        +
Where are the important features?
```

This provides a more spatially aware attention mechanism compared with channel-only attention.

---

## 📂 Project Structure

```text
Biomedical-Segmentation/
│
├── models/
│   ├── unet.py
│   ├── se_unet.py
│   └── cbam_unet.py
│
├── src/
│   ├── analyze_dataset.py
│   ├── check_dataset.py
│   ├── compare_models.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── evaluate_cbam_unet.py
│   ├── evaluate_se_unet.py
│   ├── losses.py
│   ├── train.py
│   ├── train_cbam_unet.py
│   ├── train_se_unet.py
│   ├── visualize_cbam_unet.py
│   ├── visualize_dataset.py
│   ├── visualize_prediction.py
│   └── visualize_se_unet.py
│
├── results/          # Local model checkpoints/results
├── dataset/          # Local dataset (not tracked by Git)
├── .venv/            # Local Python virtual environment
├── .gitignore
└── README.md
```

> Large datasets, virtual-environment files, Python cache files, and trained model checkpoints (`.pth`) are excluded from the Git repository.

---

## 🛠️ Technologies Used

- Python
- PyTorch
- NumPy
- OpenCV / PIL
- Matplotlib
- scikit-learn
- Jupyter Notebook
- Git & GitHub
- NVIDIA CUDA GPU acceleration

### Hardware

Development and model training are performed using an:

**NVIDIA RTX 4050 GPU**

---

## 📊 Evaluation Metrics

The segmentation models can be evaluated using commonly used segmentation metrics:

### Dice Similarity Coefficient

Measures the overlap between the predicted segmentation and the ground-truth mask.

```text
Dice = 2 × |Prediction ∩ Ground Truth|
       ────────────────────────────────
       |Prediction| + |Ground Truth|
```

### Intersection over Union (IoU)

```text
IoU = |Prediction ∩ Ground Truth|
      ────────────────────────────
      |Prediction ∪ Ground Truth|
```

Other evaluation measures can include:

- Precision
- Recall
- Accuracy
- Loss

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Jeffrin0507/Biomedical-Segmentation.git
cd Biomedical-Segmentation
```

Create a virtual environment:

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib opencv-python pillow scikit-learn
```

If using an NVIDIA GPU, install the PyTorch version compatible with your CUDA environment from the official PyTorch installation instructions.

---

## 📁 Dataset

The dataset is intentionally not included in this repository because biomedical datasets can be large and may have specific access/licensing requirements.

Place the dataset inside:

```text
dataset/
```

The expected directory structure depends on the dataset and the configuration implemented in `src/dataset.py`.

Before training, use:

```bash
python src/check_dataset.py
```

to verify the dataset structure.

You can also analyze the dataset using:

```bash
python src/analyze_dataset.py
```

---

## 🏋️ Training

### Train the baseline U-Net

```bash
python src/train.py
```

### Train SE U-Net

```bash
python src/train_se_unet.py
```

### Train CBAM U-Net

```bash
python src/train_cbam_unet.py
```

Trained model checkpoints are saved locally in the `results/` directory.

Because the trained `.pth` files are large, they are excluded from this GitHub repository.

---

## 🔬 Evaluation

Evaluate the baseline U-Net:

```bash
python src/evaluate.py
```

Evaluate SE U-Net:

```bash
python src/evaluate_se_unet.py
```

Evaluate CBAM U-Net:

```bash
python src/evaluate_cbam_unet.py
```

---

## 📈 Model Comparison

The project includes a model comparison script:

```bash
python src/compare_models.py
```

This can be used to compare the implemented architectures based on the evaluation metrics generated during experimentation.

---

## 🖼️ Visualization

Dataset visualization:

```bash
python src/visualize_dataset.py
```

Visualize predictions:

```bash
python src/visualize_prediction.py
```

Visualize SE U-Net results:

```bash
python src/visualize_se_unet.py
```

Visualize CBAM U-Net results:

```bash
python src/visualize_cbam_unet.py
```

---

## 🔬 Experimental Workflow

The overall workflow is:

```text
              Biomedical Dataset
                     │
                     ▼
              Dataset Analysis
                     │
                     ▼
             Preprocessing
                     │
                     ▼
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
      U-Net       SE U-Net     CBAM U-Net
        │            │            │
        └────────────┼────────────┘
                     ▼
                  Training
                     │
                     ▼
                Evaluation
                     │
                     ▼
             Model Comparison
                     │
                     ▼
          Segmentation Visualization
```

---

## 🎯 Research Objective

The primary objective is to study the effect of attention mechanisms on biomedical image segmentation.

The project investigates:

- How a baseline U-Net performs on the selected biomedical segmentation task
- Whether channel attention improves feature representation
- Whether combined channel and spatial attention improves segmentation
- How the models compare using segmentation metrics
- Qualitative differences between predicted and ground-truth masks

---

## 🔮 Future Improvements

Possible extensions include:

- Attention U-Net
- Transformer-based segmentation
- Vision Transformer (ViT) components
- UNet++
- DeepLab-based architectures
- Hybrid CNN-Transformer architectures
- Advanced data augmentation
- Hyperparameter optimization
- Cross-validation
- Additional biomedical datasets
- Statistical comparison of model performance
- Deployment of the trained model for inference

---

## 👨‍💻 Author

**Jeffrin Jose**

GitHub: [@Jeffrin0507](https://github.com/Jeffrin0507)

---

## 📜 License

This project is intended for academic and research purposes. Add an appropriate open-source license if you decide to distribute the code publicly under specific licensing terms.
