# Food-101 EfficientNetB0 – Dietary Monitoring (Alen, T2 2025)

This folder contains the training and evaluation code developed for the **Dietary Monitoring and Nutrition Analysis Using AI** use case under the Health Behaviour team.

The goal of this work is to build a food image classification pipeline using the **Food-101 dataset** to support dietary tracking, calorie estimation, and personalised nutrition guidance.

## Project Overview
- **Dataset:** [Food-101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) – 101 food categories with 1,000 images each.
- **Model Architecture:** EfficientNetB0 (transfer learning with TensorFlow/Keras).
- **Training Strategy:** Two-stage approach (frozen head → fine-tuning top layers) and an ultra-fast subset training mode for rapid prototyping.
- **Evaluation:** Accuracy/loss plots, confusion matrix, per-class classification report, and sample prediction outputs.

## Folder Structure
food101_alen/
│
├── copy_of_food_101_dataset.ipynb # Compact, Colab-friendly pipeline
├── README.md # This file
└── results/ # Small PNG/JPG plots and sample predictions

## How to Use
1. Clone the repository and navigate to this folder.
2. Ensure you have the Food-101 dataset in the expected directory structure:

food-101/
├── images/
└── meta/

3. Run the `training_pipeline_fast.py` script in Google Colab or locally with Python 3.9+.
4. Adjust parameters such as `IMAGES_PER_CLASS` and `EPOCHS` to control runtime.

## Features
- **Official Food-101 splits** with a separate validation set.
- **TensorFlow tf.data** pipeline with optional real-time augmentation.
- **Transfer Learning** using EfficientNetB0 pre-trained on ImageNet.
- **Rapid Prototyping Mode** for quick experiments.
- **Evaluation Tools** for visualising model performance.

## Notes
- Large datasets and trained model files are **not** committed to the repository.
- This code is optimised for **Google Colab GPU** but can run on local GPU setups with minor adjustments.

---

**Author:** Alen Antony (T2 2025)  
**Team:** Health Behaviour – MOP AI and IoT Planner  
