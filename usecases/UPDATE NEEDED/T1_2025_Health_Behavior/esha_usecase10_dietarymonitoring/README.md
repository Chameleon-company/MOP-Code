# Dietary Monitoring: Food Classification Using EfficientNetB0

This project is part of the **AI Capstone - Use Case 10: Dietary Monitoring & Nutrition Analysis**, where the goal is to classify food images into known categories to support nutrition tracking.

## Project Folder
`esha_usecase10_dietarymonitoring`

## Objective
Build a deep learning model capable of classifying food images into 5 categories:
- Cheesecake
- Cup Cakes
- Garlic Bread
- Pizza
- Sushi

## Model
- **Base Model**: EfficientNetB0 (CNN)
- **Training Platform**: Google Colab
- **Final Architecture**:
  - EfficientNetB0 base (unfrozen)
  - Dropout(0.5) + Dense(64)
  - RMSprop optimizer
  - L2 kernel regularization
- **Dataset**: 5-class subset of Food-101
- **Final Validation Accuracy**: **84%**

## Experiment Overview
Eight structured experiments were conducted:
1. Frozen EfficientNet baseline
2. Fine-tuned with Adam
3. Dropout + Dense(64)
4. RMSprop optimizer
5. Minimal augmentation
6. Batch size 64
7. Lower learning rate
8. Final model: RMSprop + L2 regularization

Each experiment was evaluated using:
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix

## Live Demo
The final notebook includes:
- Real image predictions
- “Unknown” prediction handling using a confidence threshold
- Evaluation against unseen images (including live cheesecake!)

## Notebook File
[`final_model_training_84acc.ipynb`](./final_model_training_84acc.ipynb)

## Author
Esha — AI Capstone 2025, Use Case 10 (Dietary Monitoring)
