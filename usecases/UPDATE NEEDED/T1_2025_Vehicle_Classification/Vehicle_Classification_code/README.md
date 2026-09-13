# Austroads Vehicle Classification Pipeline

A deep-learning pipeline that classifies vehicles into all 12 Austroads categories in real time. It combines YOLOv8 for object detection with a fine-tuned MobileNetV2 backbone for high-accuracy classification, and is packaged for easy deployment.

---

##  Repository Structure

├── models/
│ ├── train.py # data prep & model training
│ └── test.py # evaluation & analysis
├── requirements.txt # Python dependencies
└── README.md # This file


## 🛠️ Prerequisites

- Python 3.8+  
- NVIDIA GPU + CUDA (optional, for faster training/inference)  
- `pip install -r requirements.txt
