# Vehicle Analysis System

This project provides an end-to-end vehicle analysis system with three core components for classification, damage detection, and accident recognition.

## Features

1. **Vehicle Classification** – Classifies vehicle types using EfficientNet-B0  
2. **Car Damage Detection** – Identifies and locates vehicle damage using YOLOv8  
3. **Traffic Accident Detection** – Detects crash scenarios in traffic imagery  

## Project Structure

```
vehicle-analysis-system/
├── models/
│   ├── vehicle_classifier.py    # Vehicle classification model
│   ├── damage_detector.py       # Car damage detection model
│   ├── accident_detector.py     # Traffic accident detection model
├── files/                      # Directory for saved model weights
│   ├── .gitkeep
├── datasets/                    # Directory for training/test data
├── README.md
├── requirements.txt                 
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

Each component can be run separately:

```bash
# Vehicle classification
python models/vehicle_classifier.py

# Damage detection
python models/damage_detector.py

# Accident detection
python models/accident_detector.py
```

## Dependencies

* PyTorch and torchvision for deep learning
* `timm` for EfficientNet implementation
* `ultralytics` for YOLOv8
* scikit-learn for evaluation metrics
* OpenCV for image processing
* matplotlib and seaborn for visualization
* kagglehub for dataset downloads