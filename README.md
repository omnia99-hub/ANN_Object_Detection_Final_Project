Analyzing Neural Network Perception Errors Under Visual Distortions Using KITTI and CARLA Datasets

Author: Omnia Dafalla
Course: CS 5640/6640 – Artificial Neural Networks
Project Type: Final Research Project
Frameworks: PyTorch, YOLOv5, OpenCV, NumPy, Matplotlib

# Project Overview

This repository contains the full implementation for the project “Analyzing Neural Network Perception Errors Under Visual Distortions Using KITTI and CARLA Datasets.”
The code evaluates YOLOv5 robustness under six different distortion types (fog, blur, occlusion, glare, shadows, and clean) and implements two mitigation strategies:

Distortion-aware data augmentation

Adaptive confidence filtering

The repository includes:

Data preprocessing scripts

Distortion generation functions

YOLOv5 baseline & augmented training pipelines

Evaluation scripts

Error analysis tools

CARLA-inspired synthetic scenario generator

All plots used in the report

Full experiment reproducibility instructions

# Repository Structure
ANN-Perception-Errors/
│
├── data/
│   ├── kitti_raw/                 # Original downloaded KITTI images + labels
│   ├── kitti_yolo_format/         # Converted YOLO labels
│   ├── distortions/               # Fog, blur, occlusion, glare, shadows
│   ├── carla_scenarios/           # Synthetic illusion-based images
│
├── src/
│   ├── convert_kitti_to_yolo.py   # Annotation conversion
│   ├── generate_distortions.py    # Fog, blur, glare, shadow, occlusion
│   ├── train_baseline.py          # Train YOLOv5 baseline model
│   ├── train_augmented.py         # Train augmented model
│   ├── evaluate_models.py         # mAP, precision, recall evaluations
│   ├── confidence_analysis.py     # Histogram + threshold computation
│   ├── apply_confidence_filter.py # Adaptive confidence filtering
│   ├── error_analysis.py          # FP/FN categorization framework
│   ├── generate_carla_scenarios.py# Synthetic illusion scenes
│   ├── utils.py                   # Helper functions
│
├── results/
│   ├── plots/                     # All graphs used in the report
│   ├── inference_samples/         # Example detections
│   ├── metrics/                   # mAP, precision, recall .txt files
│
├── models/
│   ├── baseline.pt                # Baseline YOLOv5s model
│   ├── augmented.pt               # Augmented YOLOv5s model
│
├── README.md                      # Execution instructions
└── requirements.txt               # Python dependencies

# Dependencies

Install all dependencies using:
```
pip install -r requirements.txt
```
requirements.txt includes:
torch
torchvision
opencv-python
numpy
matplotlib
pandas
seaborn
pyyaml
tqdm
ultralytics==8.0.20


GPU (CUDA) is highly recommended for training.

# Dataset Download Instructions
1. Download KITTI Object Detection Dataset

Download the following two files:
```
data_object_image_2.zip
data_object_label_2.zip
```
Official link:
[https://www.kaggle.com/datasets/klemenko/kitti-dataset]

Unzip them into:
```
data/kitti_raw/
```
2. CARLA Scenarios

These are generated automatically by:
```
python src/generate_carla_scenarios.py
```
# FULL EXECUTION GUIDE
Step 1 — Convert KITTI Labels to YOLO Format
```
python src/convert_kitti_to_yolo.py
```
Output goes to:
```
data/kitti_yolo_format/
```
Step 2 — Generate Visual Distortions
```
python src/generate_distortions.py
```
This script creates folders:

data/distortions/fog/
data/distortions/blur/
data/distortions/glare/
data/distortions/shadows/
data/distortions/occlusion/

Step 3 — Train Baseline YOLOv5 Model
```
python src/train_baseline.py
```
Model saved to:
```
models/baseline.pt
```
Step 4 — Train Augmented Model

This expands the dataset 6× and trains YOLOv5 on clean + distorted images.
```
python src/train_augmented.py
```

Model saved to:
```
models/augmented.pt
```
Step 5 — Evaluate Both Models
```
python src/evaluate_models.py
```

Outputs stored in:
```
results/metrics/
results/plots/
```
Step 6 — Error Type Analysis
```
python src/error_analysis.py
```

Produces:

Shadow illusion heatmaps

Glare FN visualizations

Occlusion-type misses

Reflection-based false positives

Stored in:
```
results/error_analysis/
```
Step 7 — Confidence Distribution Analysis
```
python src/confidence_analysis.py
```

Produces:

Confidence histograms

Baseline vs Augmented comparisons

Threshold recommendations

Step 8 — Apply Adaptive Confidence Filtering
```
python src/apply_confidence_filter.py
```
Step 9 — Generate CARLA-Like Synthetic Scenarios
```
python src/generate_carla_scenarios.py
```

Results saved into:
```
data/carla_scenarios/
```
Step 10 — Run All Experiments At Once

To reproduce the full pipeline with one command:
```bash
python run_all.py
```

# Reproducing All Figures

All figures in the report come from:
```bash
results/plots/
results/error_analysis/
results/confidence/
```

This includes:

mAP comparison (baseline vs augmented)

Augmentation improvement bar charts

Confidence distributions

Error type distributions

CARLA illusion results

# Hardware Requirements

Minimum:

GPU: NVIDIA Tesla T4 / GTX 1660 / RTX series recommended

RAM: 8–16 GB

Disk space: ~15 GB for KITTI + distortions + models

The repo is executable on:

Google Colab (preferred)

Local machine with CUDA

Any Linux/Windows environment with PyTorch installed
  booktitle={CVPR},
  year={2012}
}
