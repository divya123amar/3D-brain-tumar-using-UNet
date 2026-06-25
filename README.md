# 🧠 3D U-Net for Brain Tumor Segmentation Using Multi-Modal MRI

## Overview

This repository presents a deep learning framework for automated brain tumor segmentation from multi-modal Magnetic Resonance Imaging (MRI) using a **3D U-Net architecture**. The model leverages volumetric convolutional neural networks to capture rich spatial information across three-dimensional MRI volumes, enabling accurate segmentation of brain tumor sub-regions.

The implementation is developed using **PyTorch** and is compatible with the **BraTS 2020/2021 datasets**.

---

## Research Motivation

Brain tumor segmentation plays a crucial role in computer-aided diagnosis, treatment planning, and disease monitoring. Manual delineation by radiologists is time-consuming and subject to inter-observer variability.

This project aims to:

- Develop an automated brain tumor segmentation framework.
- Utilize 3D contextual information from volumetric MRI.
- Improve segmentation accuracy for heterogeneous tumors.
- Establish a baseline model for future research in Federated Learning and Medical AI.

---

# Dataset

This project uses the **BraTS (Brain Tumor Segmentation Challenge)** dataset.

Multi-modal MRI sequences include:

- T1
- T1-Contrast (T1ce)
- T2
- FLAIR

Ground truth annotations include:

- Enhancing Tumor (ET)
- Tumor Core (TC)
- Whole Tumor (WT)

Dataset Link:

https://www.med.upenn.edu/cbica/brats2020/data.html

> The dataset is not included in this repository due to licensing restrictions.

---

# Folder Structure

```
3D-UNet-Brain-Tumor-Segmentation/

│
├── dataset/
│   ├── images/
│   ├── masks/
│
├── models/
│   ├── unet3d.py
│
├── preprocessing/
│
├── training/
│
├── evaluation/
│
├── utils/
│
├── notebooks/
│
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

---

# Workflow

```
MRI Dataset

↓

Preprocessing

↓

Normalization

↓

3D Volume Generation

↓

3D U-Net Training

↓

Prediction

↓

Performance Evaluation
```

---

# Data Preprocessing

The preprocessing pipeline includes:

- Loading NIfTI MRI volumes
- Intensity normalization
- Resizing / Cropping
- Multi-modal volume construction
- Mask encoding
- Dataset splitting
- Patch generation (if applicable)

---

# Network Architecture

The implemented model follows the classical **3D U-Net** architecture consisting of:

Encoder

- 3D Convolution
- Batch Normalization
- ReLU
- Max Pooling

Decoder

- Transposed Convolution
- Skip Connections
- Feature Fusion

Output Layer

- Softmax Activation
- Multi-class Segmentation

---

# Technologies Used

- Python
- PyTorch
- NumPy
- MONAI
- NiBabel
- OpenCV
- Matplotlib
- Scikit-learn

---

# Training Configuration

| Parameter | Value |
|------------|-------|
| Input Size | 128 × 128 × 128 |
| Batch Size | 1 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Dice Loss + Cross Entropy (modify if different) |
| Epochs | 10 (modifiable) |

---

# Evaluation Metrics

The following metrics are used:

- Dice Similarity Coefficient
- Intersection over Union (IoU)
- Pixel Accuracy
- Precision
- Recall
- F1-Score

---

# Results

Example segmentation output:

```
Input MRI
↓

Ground Truth

↓

Predicted Mask
```

(Add screenshots here.)

---

# Future Improvements

Future research directions include:

- Attention 3D U-Net
- Residual 3D U-Net
- UNETR
- Swin UNETR
- Federated Learning
- Differential Privacy
- Explainable AI
- Domain Adaptation

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/3D-UNet-Brain-Tumor-Segmentation.git
```

Navigate to the project

```bash
cd 3D-UNet-Brain-Tumor-Segmentation
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run training

```bash
python train.py
```

Run inference

```bash
python predict.py
```

---

# Repository Highlights

✔ 3D volumetric segmentation

✔ Multi-modal MRI

✔ PyTorch implementation

✔ Medical image preprocessing

✔ Baseline model for advanced research

✔ Research-oriented codebase

---

# Citation

If you use this repository in your research, please cite the corresponding publication (when available).

---

# Author

**Divyashree A**

PhD Research Scholar

Artificial Intelligence | Medical Image Analysis | Computer Vision | Deep Learning | Federated Learning

---

# License

This project is released for academic and research purposes.
