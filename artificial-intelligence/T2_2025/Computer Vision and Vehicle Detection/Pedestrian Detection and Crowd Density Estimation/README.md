Case 6: Pedestrian Detection with YOLOv8x6

Casey McDonald Trimester 2 2025

A computer vision project using YOLOv8 to detect pedestrians in images.



This project trains a YOLOv8 model to detect pedestrians in real-world urban images.

It includes training, evaluation, and visualisation of results, comparing ground truth annotations with model predictions.



├── SIT374\_usecase6\_CaseyMcDonald.ipynb # Jupyter/Colab code with further details of model

├── README.md

├── Output sample images # folder containing output samples



Dataset citation:

 ZSL. (2024). pedestrian dataset. Roboflow Universe. https://universe.roboflow.com/zsl-hmu2b/pedestrian-w16dt

 (Accessed 2025-09-15)

Please download data and upload to your Google Drive, find the path and recplace the current ipynb path.



Datasets for potential future exploration (recommended better GPU):

https://universe.roboflow.com/682projectpmp/682\_final\_project/dataset/9

https://universe.roboflow.com/objectdetectiontuto/pedestrian-wri5e/dataset/2



Requirements:

\- Python 3.8+

\- GPU recommended (T4)

\- Download dataset from Roboflow

\- GoogleDrive

\- Google Colab

Install dependencies: ultralytics, matplotlib, yaml, os, shutil, random, torch, glob, pathlib



YOLO format with:

\- images/ --> .jpg / .png

\- labels/ --> .txt (format: class x\_center y\_center width height, normalized)

\- Single class: pedestrian

\- Dataset YAML (data.yaml) defines paths to train/, val/, and test



Training metrics selected (see ipynb):

data="pedestrian.v1i.yolov8/data.yaml",

epochs=20,

imgsz=704,

batch=8,

optimizer="SGD",

cos\_lr=True,

momentum=0.937,

lr0=0.01,

lrf=0.01,

weight\_decay=0.0005,

warmup\_epochs=5,

device=0,

project="pedestrian\_detection",

name="yolov8x6\_pedestrian\_1",

single\_cls=True

