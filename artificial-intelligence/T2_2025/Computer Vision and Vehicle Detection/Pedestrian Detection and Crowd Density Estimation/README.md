Case 6: Pedestrian Detection with YOLOv8x6
Casey McDonald Trimester 2 2025
A computer vision project using YOLOv8 to detect pedestrians in images.

This project trains a YOLOv8 model to detect pedestrians in real-world images.
It includes training, evaluation, and visualisation of results, comparing ground truth annotations with model predictions.

├── SIT374_usecase6_CaseyMcDonald.ipynb # Jupyter/Colab code
├── README.md
├── Output sample images # folder containing output samples

Dataset citation:
 ZSL. (2024). pedestrian dataset. Roboflow Universe. https://universe.roboflow.com/zsl-hmu2b/pedestrian-w16dt
 (Accessed 2025-09-15)

Requirements:
- Python 3.8+
- GPU recommended
- Download dataset
- GoogleDrive
Install dependencies: ultralytics, matplotlib, yaml, os, shutil, random, torch, glob, pathlib

YOLO format with:
- images/ → .jpg / .png
- labels/ → .txt (format: class x_center y_center width height, normalized)
- Single class: pedestrian
- Dataset YAML (data.yaml) defines paths to train/, val/, and test

Training metrics:
data="pedestrian.v1i.yolov8/data.yaml",
epochs=20,
imgsz=704,
batch=8,
optimizer="SGD",
cos_lr=True,
momentum=0.937,
lr0=0.01,
lrf=0.01,
weight_decay=0.0005,
warmup_epochs=5,
device=0,
project="pedestrian_detection",
name="yolov8x6_pedestrian_1",
single_cls=True