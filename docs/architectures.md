# Neural Network Architectures: Theoretical Foundations & Implementation

This document provides an exhaustive, in-depth architectural and mathematical breakdown of the three segmentation models implemented in this repository:
1. **Baseline U-Net**
2. **Squeeze-and-Excitation U-Net (SE U-Net)**
3. **Convolutional Block Attention Module U-Net (CBAM U-Net)**

---

## Table of Contents
- [1. Baseline U-Net Architecture](#1-baseline-u-net-architecture)
  - [1.1 Conceptual Motivation](#11-conceptual-motivation)
  - [1.2 Encoder-Decoder Topology & Skip Connections](#12-encoder-decoder-topology--skip-connections)
  - [1.3 Mathematical Formulation of Operations](#13-mathematical-formulation-of-operations)
  - [1.4 Layer-by-Layer Feature Map Dimensions](#14-layer-by-layer-feature-map-dimensions)
- [2. SE U-Net (Squeeze-and-Excitation)](#2-se-u-net-squeeze-and-excitation)
  - [2.1 The Philosophy of Channel Recalibration](#21-the-philosophy-of-channel-recalibration)
  - [2.2 Squeeze Operation: Global Spatial Pooling](#22-squeeze-operation-global-spatial-pooling)
  - [2.3 Excitation Operation: Non-linear Inter-channel Dependencies](#23-excitation-operation-non-linear-inter-channel-dependencies)
  - [2.4 Feature Map Scaling & Rescaling](#24-feature-map-scaling--rescaling)
  - [2.5 Integration within U-Net Hierarchy](#25-integration-within-u-net-hierarchy)
- [3. CBAM U-Net (Convolutional Block Attention Module)](#3-cbam-u-net-convolutional-block-attention-module)
  - [3.1 Synergistic Attention: Channel vs. Spatial](#31-synergistic-attention-channel-vs-spatial)
  - [3.2 Channel Attention Sub-module Formulation](#32-channel-attention-sub-module-formulation)
  - [3.3 Spatial Attention Sub-module Formulation](#33-spatial-attention-sub-module-formulation)
  - [3.4 Sequential Arrangement & Information Flow](#34-sequential-arrangement--information-flow)
- [4. Architectural Comparison & Complexity](#4-architectural-comparison--complexity)

---

## 1. Baseline U-Net Architecture

Implemented in: [`models/unet.py`](../models/unet.py)

### 1.1 Conceptual Motivation
Medical and biomedical image segmentation presents unique challenges:
- High variability in object morphology (polyp boundaries are diffuse, irregular, and lack hard edges).
- Low contrast against background tissues (endoscopic mucosa has subtle texture transitions).
- Limited dataset sizes compared to consumer computer vision datasets (ImageNet/COCO).

Ronneberger et al. (2015) introduced **U-Net** to tackle these problems by coupling high-resolution contextual localization with deep semantic feature extraction.

```
Input (3, 256, 256)
  │
  ▼
[DoubleConv 64] ────────── Skip 1 ──────────► [Dec 1 + Conv 1x1] ──► Output (1, 256, 256)
  │ MaxPool 2x2                                    ▲ UpConv 2x2
  ▼                                                │
[DoubleConv 128] ───────── Skip 2 ──────────► [Dec 2: 128]
  │ MaxPool 2x2                                    ▲ UpConv 2x2
  ▼                                                │
[DoubleConv 256] ───────── Skip 3 ──────────► [Dec 3: 256]
  │ MaxPool 2x2                                    ▲ UpConv 2x2
  ▼                                                │
[DoubleConv 512] ───────── Skip 4 ──────────► [Dec 4: 512]
  │ MaxPool 2x2                                    ▲ UpConv 2x2
  ▼                                                │
[Bottleneck DoubleConv 1024] ──────────────────────┘
```

### 1.2 Encoder-Decoder Topology & Skip Connections
- **Contracting Path (Encoder)**: Captures context by applying repetitive sequences of two $3 \times 3$ convolutions followed by a $2 \times 2$ max pooling operation with stride 2. At each downsampling step, spatial dimensions halve while the channel capacity doubles ($64 \rightarrow 128 \rightarrow 256 \rightarrow 512$).
- **Bottleneck**: Consists of a double convolution transforming $512 \rightarrow 1024$ channels at the lowest spatial resolution ($16 \times 16$).
- **Expanding Path (Decoder)**: Restores spatial dimensions via $2 \times 2$ transposed convolutions (`ConvTranspose2d`), which halve the channel count and double height and width.
- **Skip Connections**: Concatenate the high-resolution feature maps from the contracting path to the upsampled decoder maps:
  $$\mathbf{F}_{\text{cat}} = [\mathbf{F}_{\text{up}}, \mathbf{F}_{\text{skip}}]$$
  This directly transfers fine-grained spatial representations lost during max-pooling, allowing accurate boundary reconstruction.

### 1.3 Mathematical Formulation of Operations

#### Double Convolution Unit
Each double convolution block executes:
$$\mathbf{Z}_1 = \text{ReLU}\big(\text{BN}(\mathbf{W}_1 * \mathbf{X} + \mathbf{b}_1)\big)$$
$$\mathbf{Z}_2 = \text{ReLU}\big(\text{BN}(\mathbf{W}_2 * \mathbf{Z}_1 + \mathbf{b}_2)\big)$$

Where:
- $*$ denotes 2D spatial convolution with kernel size $k = 3 \times 3$ and symmetric padding $p = 1$.
- $\text{BN}(\cdot)$ denotes Batch Normalization:
  $$\text{BN}(\mathbf{x}) = \gamma \frac{\mathbf{x} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta$$
- $\text{ReLU}(z) = \max(0, z)$ provides non-linear activation.

#### Transposed Convolution (Upsampling)
The spatial upsampling utilizes learned kernels:
$$\mathbf{X}_{\text{up}} = \text{ConvTranspose2d}(\mathbf{X}_{\text{in}}, \text{kernel\_size}=2, \text{stride}=2)$$

---

## 2. SE U-Net (Squeeze-and-Excitation)

Implemented in: [`models/se_unet.py`](../models/se_unet.py)

### 2.1 The Philosophy of Channel Recalibration
Standard convolutional layers treat all channel feature maps uniformly, performing spatial integration across all input channels with fixed weights. However, in biomedical images, certain feature channels (e.g., vascular texture filters or mucosal color gradient filters) carry significantly more discriminatory information than generic background channels.

Hu et al. (CVPR 2018) introduced **Squeeze-and-Excitation (SE)** to dynamically modulate channel-wise importance through explicit global inter-channel dependency modeling.

```
       Feature Map X (C x H x W)
                  │
                  ▼
        [Global Average Pooling]  (Spatial Squeeze)
                  │
                  ▼
            1 x 1 x C
                  │
                  ▼
            [Linear C -> C/r]     (Reduction: r=16)
                  │
                  ▼
                [ReLU]
                  │
                  ▼
            [Linear C/r -> C]     (Expansion)
                  │
                  ▼
               [Sigmoid]          (Channel weights s ∈ [0, 1])
                  │
                  ▼
       Scale: X̃ = X ⊗ s (Recalibration)
```

### 2.2 Squeeze Operation: Global Spatial Pooling
To aggregate global spatial context, spatial dimensions $H \times W$ are compressed into a channel descriptor vector $\mathbf{z} \in \mathbb{R}^C$ using Global Average Pooling:

$$z_c = \mathbf{F}_{\text{sq}}(\mathbf{u}_c) = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W u_c(i, j)$$

Where $\mathbf{u}_c$ is the $c$-th channel slice of feature map $\mathbf{U} \in \mathbb{R}^{C \times H \times W}$.

### 2.3 Excitation Operation: Non-linear Inter-channel Dependencies
To capture non-linear relationships and avoid enforcing mutually exclusive activations (which softmax would do), a two-layer bottleneck Multi-Layer Perceptron (MLP) with a reduction ratio $r = 16$ is employed:

$$\mathbf{s} = \mathbf{F}_{\text{ex}}(\mathbf{z}, \mathbf{W}) = \sigma\big(\mathbf{W}_2 \cdot \delta(\mathbf{W}_1 \cdot \mathbf{z})\big)$$

Where:
- $\mathbf{W}_1 \in \mathbb{R}^{\frac{C}{r} \times C}$ represents the dimensionality-reduction weight matrix.
- $\delta(\cdot)$ is the ReLU activation function.
- $\mathbf{W}_2 \in \mathbb{R}^{C \times \frac{C}{r}}$ represents the dimensionality-increasing weight matrix.
- $\sigma(\cdot)$ is the gating Sigmoid activation ensuring scalar outputs $s_c \in [0, 1]$.

### 2.4 Feature Map Scaling & Rescaling
The final recalibrated feature representation $\widetilde{\mathbf{X}} = [\widetilde{\mathbf{x}}_1, \widetilde{\mathbf{x}}_2, \dots, \widetilde{\mathbf{x}}_C]$ is obtained by channel-wise multiplication:

$$\widetilde{\mathbf{x}}_c = \mathbf{F}_{\text{scale}}(\mathbf{u}_c, s_c) = s_c \cdot \mathbf{u}_c$$

### 2.5 Integration within U-Net Hierarchy
In [`models/se_unet.py`](../models/se_unet.py), an `SEBlock` is placed immediately following every `DoubleConv` stage:
- After `enc1`, `enc2`, `enc3`, `enc4` (Encoder stages)
- After `bottleneck`
- After `dec4`, `dec3`, `dec2`, `dec1` (Decoder stages)

This allows both fine-grained encoder representations and upsampled decoder features to be recalibrated before downsampling or concatenation.

---

## 3. CBAM U-Net (Convolutional Block Attention Module)

Implemented in: [`models/cbam_unet.py`](../models/cbam_unet.py)

### 3.1 Synergistic Attention: Channel vs. Spatial
While SE-Net focuses exclusively on *what* channel features are meaningful, Woo et al. (ECCV 2018) established that knowing *where* informative features reside spatially is equally vital. 

The **Convolutional Block Attention Module (CBAM)** couples:
1. **Channel Attention Module (CAM)**: Answers *"What is salient?"*
2. **Spatial Attention Module (SAM)**: Answers *"Where is it salient?"*

```
 Input Feature F (C x H x W)
             │
             ▼
 ┌───────────────────────┐
 │   Channel Attention   │  AvgPool & MaxPool ──► Shared MLP ──► Sigmoid ──► Mc(F)
 └───────────────────────┘
             │
             ▼
        F' = Mc(F) ⊗ F
             │
             ▼
 ┌───────────────────────┐
 │   Spatial Attention   │  AvgPool & MaxPool along C ──► Conv 7x7 ──► Sigmoid ──► Ms(F')
 └───────────────────────┘
             │
             ▼
        F'' = Ms(F') ⊗ F'
```

### 3.2 Channel Attention Sub-module Formulation
CBAM improves upon SE-Net's channel descriptor by simultaneously exploiting **Average-pooling** and **Max-pooling**. While average pooling preserves background spatial statistics, max-pooling extracts the most distinctive object features (e.g., sharp polyp tissue edges).

$$\mathbf{M}_c(\mathbf{F}) = \sigma\Big(\text{MLP}\big(\text{AvgPool}(\mathbf{F})\big) + \text{MLP}\big(\text{MaxPool}(\mathbf{F})\big)\Big)$$

In mathematical detail:
$$\mathbf{M}_c(\mathbf{F}) = \sigma\Big(\mathbf{W}_1\big(\mathbf{W}_0(\mathbf{F}_{\text{avg}}^c)\big) + \mathbf{W}_1\big(\mathbf{W}_0(\mathbf{F}_{\text{max}}^c)\big)\Big)$$

Where:
- $\mathbf{W}_0 \in \mathbb{R}^{\frac{C}{r} \times C}$ and $\mathbf{W}_1 \in \mathbb{R}^{C \times \frac{C}{r}}$ are shared across both pooled vectors.
- $r = 16$ is the reduction ratio.
- The intermediate feature map is computed as:
  $$\mathbf{F}' = \mathbf{M}_c(\mathbf{F}) \otimes \mathbf{F}$$

### 3.3 Spatial Attention Sub-module Formulation
The spatial attention module generates a 2D spatial mask $\mathbf{M}_s(\mathbf{F}') \in \mathbb{R}^{1 \times H \times W}$ by aggregating channel information across every spatial coordinate:

1. Channel-wise Average Pooling:
   $$\mathbf{F}_{\text{avg}}^s = \frac{1}{C} \sum_{c=1}^C F'_c \in \mathbb{R}^{1 \times H \times W}$$
2. Channel-wise Maximum Pooling:
   $$\mathbf{F}_{\text{max}}^s = \max_{c \in \{1,\dots,C\}} F'_c \in \mathbb{R}^{1 \times H \times W}$$
3. Concatenation and Convolution:
   $$\mathbf{M}_s(\mathbf{F}') = \sigma\Big(f^{7 \times 7}\big([\mathbf{F}_{\text{avg}}^s ; \mathbf{F}_{\text{max}}^s]\big)\Big)$$

Where:
- $[\cdot ; \cdot]$ represents concatenation along the channel dimension (resulting in 2 channels).
- $f^{7 \times 7}$ denotes a 2D convolution with a large receptive field kernel ($7 \times 7$) and padding 3 to capture spatial context.
- $\sigma$ is the Sigmoid activation function.

The final refined feature map is:
$$\mathbf{F}'' = \mathbf{M}_s(\mathbf{F}') \otimes \mathbf{F}'$$

### 3.4 Sequential Arrangement & Information Flow
Empirical studies in the CBAM paper showed that sequential channel-first, spatial-second ordering outperforms spatial-first or parallel arrangements. 

In [`models/cbam_unet.py`](../models/cbam_unet.py):
- The `CBAM` module applies `ChannelAttention` followed immediately by `SpatialAttention`.
- This module is inserted after every encoder and decoder stage, creating a dense attention-guided pathway throughout the network.

---

## 4. Architectural Comparison & Complexity

| Model | Attention Mechanism | Spatial Receptive Focus | Extra Parameters | Computational Overhead |
| :--- | :--- | :--- | :--- | :--- |
| **U-Net** | None (Uniform weighting) | Local convolutional kernels | Baseline (~31.04M) | Baseline (1.0x) |
| **SE U-Net** | Squeeze-and-Excitation | Global spatial average $\rightarrow$ Channel reweighting | +0.15% (~31.09M) | ~1.02x FLOPs |
| **CBAM U-Net** | Channel + Spatial Attention | Dual-pooled channels + $7 \times 7$ spatial context | +0.22% (~31.11M) | ~1.05x FLOPs |

---

For execution and training details of these architectures, see:
- [Dataset Pipeline Documentation](dataset.md)
- [Training & Loss Functions](training_and_losses.md)
- [Metrics & Evaluation Guide](metrics_and_evaluation.md)
