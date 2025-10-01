# Food-101 & Nutrition5k – Dietary Monitoring (Alen, T2 2025)

This repository contains the training and evaluation code developed for the Dietary Monitoring and Nutrition Analysis Using AI use case under the Health Behaviour team.

The goal of this work is to build a food image classification pipeline using the Food-101 dataset, and extend it with the Nutrition5k dataset for enhanced dietary tracking, calorie estimation, and personalised nutrition guidance.

## Project Overview
- **Dataset:** [Food-101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) – 101 food categories with 1,000 images each.
- **Model Architecture:** EfficientNetB0 (transfer learning with TensorFlow/Keras).
- **Training Strategy:** Two-stage approach (frozen head → fine-tuning top layers) and an ultra-fast subset training mode for rapid prototyping.
# Food-101 EfficientNetB0 – Dietary Monitoring.
# Nutrition5k:

5,000+ meals with paired video frames, metadata, and nutritional breakdowns

Side-angle frames extracted using FFmpeg

Cleaned and standardised metadata (calories, protein, carbs, fat, ingredients)
**Python | TensorFlow | Open-Source License**

A comprehensive food image classification pipeline developed for the SIT782 coursework (Capstone – Health Behaviour Use Case).  
Implements transfer learning with EfficientNetB0 for **multi-class food recognition** using the **Food-101 dataset**, enabling dietary monitoring, calorie estimation, and personalised nutrition guidance.

---

## 🎯 Project Overview
This project builds an end-to-end deep learning pipeline to classify food images into one of **101 categories**.  
The system is designed to support AI-powered dietary tracking by accurately recognising food items from photographs.

**Key Features:**
Automated preprocessing & augmentation

Modular tf.data pipeline with caching & prefetching

Transfer learning with EfficientNetB0

Training strategies:

Full dataset (Food-101 / Nutrition5k)

Balanced subset with capped steps (for faster experiments)

Evaluation tools: accuracy/loss plots, confusion matrices, per-class classification reports

Nutrition5k extension: integrated video frame → metadata pipeline
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

📂 Food-Nutrition-Monitoring
 ┣ 📜 Copy_of_Food_101_dataset.ipynb     # Food-101 preprocessing + training
 ┣ 📜 Nutrition5k.ipynb                  # Nutrition5k preprocessing + training
 ┣ 📜 README.md                          # Project documentation
 ┣ 📂 food-101/                          # Food-101 dataset (images + meta)
 ┣ 📂 nutrition5k/                       # Nutrition5k dataset (frames + metadata)
 ┣ 📂 checkpoints/                       # Saved model checkpoints
 ┣ 📂 export/                            # Exported models + labels
 ┣ 📂 results/                           # Accuracy/loss plots, confusion matrices


🔬 Methodology
Food-101

Load official splits → resize to 224×224×3 RGB

Normalise for EfficientNet preprocessing

Augmentation (flip, rotate, zoom, brightness, contrast)

Nutrition5k

Extract side-angle frames via FFmpeg

Map frames to dish metadata (ingredients, nutrition totals)

Clean & standardise missing values

Build TF pipeline for image + metadata training

Training

Stage 1: Train classification head (Adam, lr=1e-3)

Stage 2: Fine-tune last 40 layers (Adam, lr=1e-4)

Subset training: capped steps for quick validation

Evaluation

Metrics: accuracy, precision, recall, F1

Visualisations: accuracy/loss curves, confusion matrices

Sample predictions for qualitative validation


📈 Results (Ultra-Fast Mode Example)
Food-101 (Ultra-Fast Mode, Example)

Test Accuracy: ~0.22 (subset, 2 epochs, 30 imgs/class)

Training Time: ~3 min

Full dataset training yields significantly higher performance

Nutrition5k (Early Experiments)

Successfully built preprocessing pipeline

Metadata cleaning + frame alignment complete

Next step: integrate joint model (image + nutrition metadata)

🛠️ Technical Implementation
Key Components:

TensorFlow/Keras – Model development & training

scikit-learn – Evaluation metrics

Matplotlib/Seaborn – Visualisation

OpenCV / FFmpeg – Image & video frame processing

Google Colab – Experimentation environment

GitHub – Version control & collaboration

🔮 Future Improvements
Train Nutrition5k model on full dataset (20–30 epochs)

Explore advanced augmentations (MixUp, CutMix)

Hyperparameter tuning (learning rate, batch size, dropout)

Extend model to multi-modal input (image + metadata fusion)

Deploy pipeline to cloud for scalable dietary monitoring



