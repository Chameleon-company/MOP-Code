# Food-101 EfficientNetB0 – Dietary Monitoring (Alen, T2 2025)

This folder contains the training and evaluation code developed for the **Dietary Monitoring and Nutrition Analysis Using AI** use case under the Health Behaviour team.

The goal of this work is to build a food image classification pipeline using the **Food-101 dataset** to support dietary tracking, calorie estimation, and personalised nutrition guidance.

## Project Overview
- **Dataset:** [Food-101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) – 101 food categories with 1,000 images each.
- **Model Architecture:** EfficientNetB0 (transfer learning with TensorFlow/Keras).
- **Training Strategy:** Two-stage approach (frozen head → fine-tuning top layers) and an ultra-fast subset training mode for rapid prototyping.
# Food-101 EfficientNetB0 – Dietary Monitoring (SIT782 Project)

**Python | TensorFlow | Open-Source License**

A comprehensive food image classification pipeline developed for the SIT782 coursework (Capstone – Health Behaviour Use Case).  
Implements transfer learning with EfficientNetB0 for **multi-class food recognition** using the **Food-101 dataset**, enabling dietary monitoring, calorie estimation, and personalised nutrition guidance.

---

## 🎯 Project Overview
This project builds an end-to-end deep learning pipeline to classify food images into one of **101 categories**.  
The system is designed to support AI-powered dietary tracking by accurately recognising food items from photographs.

**Key Features:**
- Official Food-101 train/test splits with custom validation set
- Automated data preprocessing and optional augmentation
- Transfer learning using EfficientNetB0 (ImageNet weights)
- Two-stage training: frozen head → fine-tuning top layers
- Ultra-fast training mode for rapid prototyping
- Accuracy/loss plots, confusion matrix, per-class classification report
- Sample prediction outputs for quick validation

---

## 📊 Dataset
**Dataset Source:** [Food-101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)

**Food-101 Dataset Statistics:**
- **Total Images:** 101,000 (1,000 per class)
- **Official Train Set:** 75,750 images
- **Official Test Set:** 25,250 images
- **Validation Set:** Created from 10% of train set
- **Classes:** 101 food categories (e.g., apple pie, sushi, pizza, ice cream)
- **Image Format:** JPEG, varied resolution (resized to 224×224×3 RGB for training)

⚠ **Important:**  
You must download the dataset from the [official Food-101 page](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) and place it in the same directory as the notebook before running.

---

## 🏗️ Architecture
**Model Architecture:** EfficientNetB0 (Transfer Learning)  
**Input Shape:** 224×224×3 RGB images  
**Output:** 101-class food classification  
**Training Strategy:**
- Stage 1: Freeze EfficientNetB0 base and train classification head
- Stage 2: Unfreeze last 40 layers for fine-tuning
- Optional ultra-fast subset training for experimentation
**Optimization:** Adam (stage 1), Adam (lower LR for fine-tuning)  
**Batch Size:** 64  

---

## 🚀 Getting Started

### **Prerequisites**

pip install tensorflow scikit-learn matplotlib pandas opencv-python


Download the Food-101 dataset
Go to Food-101 Dataset

Download and extract into the same directory as Copy_of_Food_101_dataset.ipynb

📁 Project Structure When Run

Copy_of_Food_101_dataset/
README.md                        # Project documentation
Copy_of_Food_101_dataset.ipynb    # Main notebook (preprocessing + training)
food-101/                         # Dataset directory
images/                       # All images in subfolders per class
meta/                         # Official splits and labels
checkpoints/                      # Saved model checkpoints
export/                           # Final model + labels
results/                          # Accuracy/loss plots, confusion matrices, sample predictions

🔬 Methodology
Data Pipeline
Load official train/test splits from meta files

Create validation set (10% of train data)

Resize images to 224×224

Normalise for EfficientNet preprocessing

Optional: augment with flips, rotation, zoom, brightness, contrast

Model Training
Stage 1: Train only classification head

Stage 2: Fine-tune last 40 layers

Optional ultra-fast mode: balanced subset, capped steps

Evaluation
Training & validation accuracy/loss plots

Confusion matrix and classification report

Sample prediction visualisations


📈 Results (Ultra-Fast Mode Example)
Metric	Value
Test Accuracy	~0.22
Epochs	2
Images/Class	30
Training Time	~3 min

⚠ Accuracy in ultra-fast mode is lower due to reduced dataset size and epochs. Full dataset training yields higher performance.

🛠️ Technical Implementation
Key Components:

Data Loaders: tf.data pipelines with prefetching and optional augmentation

Model: EfficientNetB0 with dropout and dense classification head

Training: Two-phase transfer learning

Evaluation: sklearn metrics + Matplotlib visualisation

Technologies Used:

TensorFlow/Keras

scikit-learn

Matplotlib, Seaborn

OpenCV

🔮 Future Improvements
Train on full dataset with 20–30 epochs

Advanced augmentation (MixUp, CutMix)
Hyperparameter tuning
Integration into dietary tracking platform

