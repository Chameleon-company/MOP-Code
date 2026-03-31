# 🔍 Bridge & Road Crack Detection System
## Component 1: Crack Segmentation Model

> **Status: Work In Progress** — Training and evaluation complete. Inference and mask output not implemented yet.

---

## 📌 Project Context

This is part of a MOP Project 6 building an end-to-end crack detection pipeline.

**Full Pipeline:**
```
Image → Crack Detection Model → Crack Mask → Crack Metrics → LLM Report → Streamlit App
```

**My responsibility (Team Member 1):** Build the segmentation model that takes an infrastructure image and outputs a binary crack mask. This is the foundation for all downstream components.

---

## 🧠 Models Trained

| Model | Encoder | Epochs | Loss Function | Optimiser |
|---|---|---|---|---|
| UNet | ResNet34 (ImageNet) | 15 | Dice Loss | Adam (lr=0.0001) |
| DeepLabV3+ | ResNet50 (ImageNet) | 30 | Dice Loss | Adam (lr=0.0001) |

---

## 📊 Results

| Model | Best Dice | Best IoU | Best Epoch |
|---|---|---|---|
| UNet + ResNet34 | 0.7348 | 0.6102 | 13 |
| DeepLabV3+ + ResNet50 | 0.7334 | 0.6067 | 29 |

**Selected Model: UNet + ResNet34** — achieved higher Dice and IoU scores in fewer epochs.

---

## 📁 Dataset

- **Source:** [Crack Segmentation Dataset — Kaggle](https://www.kaggle.com/datasets/lakshaymiddha/crack-segmentation-dataset)
- **Total images:** 9,603 (crack images only, non-crack excluded)
- **Split:** 7,682 train / 1,921 validation (80/20, random_state=42)
- **Input size:** 224×224
- **Mask format:** Binary (0 = background, 1 = crack)

---

## 🔄 Data Augmentation

**Training:** Horizontal flip, vertical flip, shift/scale/rotate (±45°), random brightness/contrast, normalisation  
**Validation:** Resize + normalisation only

---

## 📁 Notebook Structure

| Section | Description |
|---|---|
| 1. Imports | Libraries and dependencies |
| 2. Environment Setup | Kaggle dataset download |
| 3. Dataset Preparation | Load 9,603 image-mask pairs, 80/20 split |
| 4. Transforms | Augmentation for training, normalisation for val |
| 5. Dataset Class & DataLoaders | Custom PyTorch Dataset, batch size 32 |
| 6. Sample Visualisation | Preview images and binary masks |
| 7. Metric Functions | IoU and Dice Score |
| 8. Training Loop | Dice Loss + Adam, per-epoch logging |
| 9. Model 1 — UNet + ResNet34 | 15 epochs, best Dice 0.7348 @ Epoch 13 |
| 10. Model 2 — DeepLabV3+ + ResNet50 | 30 epochs, best Dice 0.7334 @ Epoch 29 |
| 11. Model Selection | Side-by-side Dice and IoU comparison |

---

## 🛠️ Tech Stack

- Python, PyTorch (CUDA)
- segmentation-models-pytorch
- Albumentations
- OpenCV, NumPy, Matplotlib

---

## 🚧 What's Next

- [ ] Hyperparameter tuning the models to optimise and improve performance
- [ ] Run inference on unseen images
- [ ] Output `crack_mask.png` for each input image
- [ ] Pass masks to Component 2 (Crack Metrics module)

---

## ⚠️ Note
Do not hardcode Kaggle API credentials. Use Kaggle's secrets manager or environment variables.
