# Project Roadmap, Contributing Guidelines & Citations

This document outlines future development directions, instructions for contributing to the repository, pull request standards, and academic citations.

---

## Table of Contents
- [1. Research & Engineering Roadmap](#1-research--engineering-roadmap)
  - [1.1 Advanced Architectural Extensions](#11-advanced-architectural-extensions)
  - [1.2 Inference Optimization & Real-Time Deployment](#12-inference-optimization--real-time-deployment)
  - [1.3 Multi-Dataset Validation](#13-multi-dataset-validation)
- [2. Contributing Guide](#2-contributing-guide)
  - [2.1 Branching Strategy](#21-branching-strategy)
  - [2.2 Pull Request Workflow](#22-pull-request-workflow)
  - [2.3 Code Style & Formatting](#23-code-style--formatting)
- [3. Academic Citations & References](#3-academic-citations--references)

---

## 1. Research & Engineering Roadmap

### 1.1 Advanced Architectural Extensions
- [ ] **Attention U-Net**: Implement Oktay et al. attention gates directly within the skip connections to filter irrelevant background activations prior to decoder concatenation.
- [ ] **UNet++ (Nested U-Net)**: Incorporate dense, nested skip pathways to reduce the semantic gap between encoder and decoder sub-networks.
- [ ] **Transformer Encoders (TransUNet / Swin-Unet)**: Pair convolutional decoders with Vision Transformer (ViT) or Swin Transformer backbones to capture global long-range self-attention.
- [ ] **Deep Supervision**: Add auxiliary segmentation heads at intermediate decoder resolutions to facilitate gradient flow during early training stages.

### 1.2 Inference Optimization & Real-Time Deployment
- [ ] **Half-Precision (FP16) & INT8 Quantization**: Quantize model weights for minimal loss in Dice score while slashing memory footprint.
- [ ] **ONNX & TensorRT Export**: Export PyTorch computation graphs to ONNX models to achieve >45 FPS real-time throughput on edge endoscopy video streams.
- [ ] **Interactive Web Interface**: Build a lightweight Gradio / Streamlit application allowing practitioners to upload colonoscopy frames and view real-time mask overlays with opacity controls.

### 1.3 Multi-Dataset Validation
- [ ] Benchmark generalizability on additional colorectal benchmarks:
  - **CVC-ClinicDB** (612 frames)
  - **ETIS-LaribPolypDB** (196 frames)
  - **BKAI-IGH NeoPolyp** (1,200 frames)

---

## 2. Contributing Guide

We welcome contributions from researchers and developers! Please follow this workflow to maintain a clean git history.

### 2.1 Branching Strategy
1. Always create a feature or topic branch off `main`:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/attention-gate-unet
   # or: docs/update-tutorial
   # or: fix/dataloader-num-workers
   ```

2. Branch naming convention:
   - `feature/<feature-name>`: New models, training routines, or loss functions.
   - `fix/<bug-description>`: Bug fixes or data handling corrections.
   - `docs/<doc-topic>`: Improvements to documentation and guides.

### 2.2 Pull Request Workflow
1. Commit your changes with clear, descriptive commit messages:
   ```bash
   git commit -m "feat(models): add Attention U-Net with additive gating"
   ```
2. Push your branch to the remote repository:
   ```bash
   git push -u origin feature/<feature-name>
   ```
3. Open a Pull Request on GitHub targeting `main`.
4. Ensure your PR description includes:
   - Summary of proposed changes.
   - Associated issue (if applicable).
   - Training or evaluation results demonstrating correctness.
5. After review, merge the PR (squash-and-merge or rebase).

### 2.3 Code Style & Formatting
- Adhere to **PEP 8** standards.
- Maintain comprehensive comments and modular block headers across all model definitions.
- Ensure all tensors are moved to `DEVICE` via `.to(DEVICE, non_blocking=True)`.

---

## 3. Academic Citations & References

If you use this repository or its implementations in your academic research, please cite the respective foundational papers:

### Kvasir-SEG Dataset
```bibtex
@inproceedings{jha2020kvasir,
  title={Kvasir-SEG: A Segmented Polyp Dataset},
  author={Jha, Debesh and Smedsrud, Pia H and Riegler, Michael A and Halvorsen, P{\aa}l and de Lange, Thomas and Johansen, Dag and Carstens, H{\aa}vard D},
  booktitle={International Conference on Multimedia Modeling},
  pages={451--462},
  year={2020},
  organization={Springer}
}
```

### Baseline U-Net
```bibtex
@inproceedings{ronneberger2015u,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={234--241},
  year={2015},
  organization={Springer}
}
```

### Squeeze-and-Excitation Networks (SE-Net)
```bibtex
@inproceedings{hu2018squeeze,
  title={Squeeze-and-Excitation Networks},
  author={Hu, Jie and Shen, Li and Sun, Gang},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={7132--7141},
  year={2018}
}
```

### Convolutional Block Attention Module (CBAM)
```bibtex
@inproceedings{woo2018cbam,
  title={CBAM: Convolutional Block Attention Module},
  author={Woo, Sanghyun and Park, Jongchan and Lee, Joon-Young and Kweon, In So},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={3--19},
  year={2018}
}
```
