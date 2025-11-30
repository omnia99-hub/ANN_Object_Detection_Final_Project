# ANN Final Project — Neural Network Perception Under Visual Distortions (KITTI + YOLOv5)

This repository contains the full code and analysis for my CS 5640/6640 Artificial Neural Networks final project.  
The project evaluates YOLOv5 robustness under synthetic distortions (fog, blur, motion blur, and occlusion) using the KITTI dataset.

## 1. Contents of this Repository
- `ANN_Final_Project_Object_Detection.ipynb`  
  Main notebook containing:
  - KITTI image loading  
  - Distortion generation  
  - YOLO baseline evaluation  
  - Augmented model evaluation  
  - Plots (mAP, confidence, detection count, severity)  
  - Object detection visualization  

- `requirements.txt` (full list of dependencies)

- `figures/` (optional folder with saved plots)

## 2. Install Dependencies

```bash
pip install -r requirements.txt
