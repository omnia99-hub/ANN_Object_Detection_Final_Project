# ANN Final Project — YOLOv5 Robustness Under Visual Distortions (KITTI Dataset)

This repository contains the full code and notebook for CS 5640/6640 *Artificial Neural Networks* Final Project.  
The project evaluates how object detection performance degrades under synthetic visual distortions—such as fog, blur, occlusion, and motion blur—using a subset of the KITTI dataset and the YOLOv5 object detector.

All experiments, plots, metrics, and results included in the final report were produced directly from the notebook in this repository.

---

## 1. Repository Contents

ANN_Object_Detection_Final_Project/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│ └── ANN_Final_Project_Object_Detection.ipynb
│
└── figures/


### **`notebooks/ANN_Final_Project_Object_Detection.ipynb` Includes:**

- Loading KITTI dataset images  
- Applying synthetic distortions (fog, blur, motion blur, occlusion)  
- Running YOLOv5 baseline detection  
- Running YOLOv5 retrained with distortion-aware augmentation  
- Computing all evaluation metrics:
  - mAP50  
  - Precision  
  - Confidence distributions  
  - Detection counts  
  - Severity-based confidence drop  
- Generating all plots used in the final PDF report  
- Producing qualitative detection visualizations  

The notebook runs on both **Google Colab** and **local Jupyter Notebook** environments.

---

## 2. Installation & Dependencies

All required packages are in `requirements.txt`.

Install dependencies:

```bash
pip install -r requirements.txt
```
## 3. Dataset (Custom KITTI Subset — 202 Images + 101 Labels)

This project uses the KITTI Object Detection dataset, downloaded from Kaggle:

Kaggle dataset link:

[https://www.kaggle.com/datasets/klemenko/kitti-dataset]

Because the full dataset is large, a custom subset was created.

Images Used

Downloaded:

data_object_image_2.zip

Which contains:

training/image_2/
testing/image_2/


From these, selected:

101 images from training/image_2

101 images from testing/image_2

Total images used: 202

Labels Used

Downloaded:

data_object_label_2.zip

From:

training/label_2/


Selected 101 label files matching the 101 training images used.

Dataset Not Included in the Repository

KITTI cannot be uploaded to GitHub due to licensing restrictions and size limits.

Dataset Setup Instructions

After extracting the KITTI data:

Choose any subset of images
(101 training + 101 testing recommended, as used in this project).

(Optional) Include matching labels for the training images.

Place your chosen images into a folder, e.g.:

/kitti_subset/image_2/


Update the dataset path inside the notebook:

data_path = "C:/Users/Documents/kitti_subset/image_2"


The notebook automatically loads all images from this directory.

## 4. Running the Notebook
Google Colab (Recommended)

Upload the notebook

Upload your kitti_subset/image_2 folder

Run all cells in order

All plots and metrics will be generated automatically.

Local Jupyter Notebook
jupyter notebook notebooks/ANN_Final_Project_Object_Detection.ipynb

## 5. YOLOv5 Setup (Preconfigured)

The notebook contains the required YOLO installation steps:
```bash
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -r requirements.txt
```

No manual setup is needed.

## 6. Analysis Performed

The analysis evaluates YOLOv5’s robustness under visual distortions commonly encountered in autonomous driving scenarios.

1. Baseline Performance Evaluation

The baseline YOLOv5s (COCO-pretrained) was evaluated on 202 KITTI images.
Computed metrics:

mAP50

Precision

Confidence scores

This serves as the clean reference model.

2. Synthetic Distortion Generation

Using custom OpenCV-based distortion functions, images were transformed with:

Fog (Perlin-noise based)

Gaussian blur

Motion blur

Occlusion (random block occlusion)

These distortions reflect real-world visual challenges.

3. Performance Under Distortions

For each distortion type, the notebook computes:

mAP50 drop

Precision changes

Detection count changes (false negatives)

Confidence distribution changes

This identifies which distortions impact YOLOv5 the most.

4. Data Augmentation Training

A second YOLOv5 model was trained where:

50% of images were clean

50% were synthetically distorted

The notebook compares:

Baseline YOLOv5

Augmented YOLOv5

using identical metrics.

5. Runtime Analysis

Runtime FPS was measured under all distortions.
Findings:

Distortions affect accuracy, not speed

YOLOv5 remained real-time at ~30 FPS

6. Qualitative Visualization

YOLO predictions were visualized to inspect:

Bounding box quality

False positives

Missed detections

Changes in confidence

Summary

Fog and blur produce the steepest accuracy decline

Occlusion reduces the detection count significantly

Augmented model improves robustness across all distortions

Runtime remains stable regardless of distortion

Augmentation improves bounding box stability and confidence

## 7. Experimental Results (Plots)
1. Baseline mAP50 Under Visual Distortions

File: baseline_plot.png
Description: Shows how the baseline YOLOv5 model performs on clean, fog, blur, and occlusion images.

2. Effect of Data Augmentation on mAP50

File: augmented_plot.png
Description: Comparison of baseline vs. augmented YOLOv5 performance under distortions.

3. Confidence Score Distribution (Clean vs Fog vs Augmented Fog)

File: confidence_histogram.png
Description: Shows how distortions and augmentation affect model confidence.

4. Object Detection Counts Across Distortions

File: detection_counts.png
Description: Number of detected vehicles under each distortion type.

5. mAP Comparison Across Distortions (Baseline vs Augmented)

File: map_comparison.png
Description: Side-by-side mAP50 comparison for each distortion category.

6. YOLOv5 Detection Example on KITTI Image

File: object_detection_example.png
Description: Visualization of YOLOv5 detections on one of your KITTI images.

7. Runtime Performance Across Distortions (FPS)

File: runtime.png
Description: YOLO inference speed on clean and distorted images.

8. Confidence Drop vs Distortion Severity Level

File: severity_confidence.png
Description: Shows how average confidence decreases as distortion severity increases.
