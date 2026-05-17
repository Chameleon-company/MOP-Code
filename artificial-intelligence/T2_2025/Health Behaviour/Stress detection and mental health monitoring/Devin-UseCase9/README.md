# Facial Emotion Recognition System - SIT782 Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](LICENSE)

A comprehensive facial emotion recognition system built for the SIT782 (Master of Data Science) coursework, implementing deep learning techniques for multi-class emotion classification using the FER2013 dataset.

## 🎯 Project Overview

This project develops an end-to-end machine learning pipeline for recognizing facial emotions from images. Using transfer learning with EfficientNetV2-S architecture, the system classifies faces into 7 emotional categories: angry, disgust, fear, happy, sad, surprise, and neutral.

**Key Features:**

- Automated data preprocessing and augmentation
- Transfer learning with state-of-the-art CNN architecture
- Class balancing for imbalanced emotion datasets
- Multiple model export formats for deployment
- Comprehensive evaluation metrics and training logs

## 📊 Dataset

**Dataset Source:** [FER2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

**⚠️ Important:** You must download the dataset manually from the Kaggle link above and place it in the same directory as the notebook before running the code.

**FER2013 Dataset Statistics:**

- **Total Images:** 35,887
- **Training Set:** 25,838 images
- **Validation Set:** 2,871 images
- **Test Set:** 7,178 images
- **Classes:** 7 emotions (angry, disgust, fear, happy, sad, surprise, neutral)
- **Image Format:** 48×48 grayscale (upsampled to 224×224×3 for training)

## 🏗️ Architecture

**Model Architecture:** EfficientNetV2-S

- **Input Shape:** 224×224×3 RGB images
- **Output:** 7-class emotion classification
- **Training Strategy:** Transfer learning with fine-tuning
- **Optimization:** Adam optimizer (lr=0.0001)
- **Batch Size:** 16

**Performance:**

- **Final Validation Accuracy:** ~21% (5 epochs)
- **Training Time:** ~5 epochs with early stopping
- **Model Size:** 92.4MB (full model)

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow-datasets datasets torch torchvision imbalanced-learn tqdm pandas numpy matplotlib pillow scikit-learn
```

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/fer2013-emotion-recognition.git
cd fer2013-emotion-recognition
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. **Download the FER2013 dataset:**

   - Go to [FER2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
   - Download the dataset archive
   - Extract and place the data in the project directory (same folder as the notebook)
   - The notebook will automatically detect and process the dataset

4. Run the Jupyter notebook:

```bash
jupyter notebook fer2013_preprocessing_report.ipynb
```

## 📁 Project Structure When Run

```
SIT_782_TPB/
├── README.md                              # Project documentation
├── fer2013_preprocessing_report.ipynb     # Main notebook (preprocessing + training)
├── requirements.txt                       # Python dependencies
├── model_metadata.json                   # Model configuration
├── fer2013_dataset_info.json            # Dataset statistics
├── training_log.csv                      # Training history
├── fer2013_local_manifest.parquet       # Dataset manifest
├── data_fer2013/                         # Dataset directory
│   ├── train/                           # Training images by class
│   └── test/                            # Test images by class
```

## 🔬 Methodology

### Data Pipeline

1. **Data Loading:** Manual download from Kaggle with automatic detection and processing
2. **Preprocessing:** Image resizing, normalization, and augmentation
3. **Split Generation:** 90/10 train/validation split from original training data
4. **Class Balancing:** Weighted sampling for imbalanced emotion classes

### Model Training

1. **Transfer Learning:** Pre-trained EfficientNetV2-S backbone
2. **Fine-tuning:** Custom classification head for 7 emotions
3. **Augmentation:** Random horizontal flip, brightness/contrast adjustment
4. **Regularization:** Dropout and batch normalization

### Evaluation

- Training and validation accuracy/loss tracking
- Per-epoch metric logging
- Model checkpointing (best validation accuracy)
- Multiple export formats for deployment

## 📈 Results

| Metric                    | Value       |
| ------------------------- | ----------- |
| Final Validation Accuracy | ~21%        |
| Training Epochs           | 5           |
| Best Model Size           | 92.4MB      |
| Dataset Processing Time   | ~30 minutes |
| Training Time             | ~2 hours    |

_Note: The relatively low accuracy suggests the model would benefit from additional training epochs, hyperparameter tuning, and advanced augmentation techniques._

## 🛠️ Technical Implementation

**Key Components:**

- **Data Loaders:** TensorFlow and PyTorch compatible pipelines
- **Augmentation:** Real-time image transformations
- **Model Architecture:** EfficientNetV2-S with custom head
- **Training Loop:** Custom training with metric tracking
- **Export Pipeline:** Multiple model formats (.keras, .h5, SavedModel)

**Technologies Used:**

- **Deep Learning:** TensorFlow/Keras, PyTorch
- **Data Processing:** pandas, NumPy, scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Dataset Management:** TensorFlow Datasets, Hugging Face Datasets

## 🔮 Future Improvements

- [ ] Extended training with more epochs
- [ ] Hyperparameter optimization (learning rate scheduling, batch size)
- [ ] Advanced data augmentation (CutMix, MixUp)
- [ ] Ensemble methods for improved accuracy
- [ ] Real-time emotion detection pipeline
- [ ] Model quantization for mobile deployment

## 📚 References

1. Goodfellow, I. J. et al. (2013). "Challenges in representation learning: A report on three machine learning contests." _Neural Networks_, 64, 59-63.
2. Tan, M., & Le, Q. V. (2021). "EfficientNetV2: Smaller Models and Faster Training." _ICML_.
3. FER2013 Dataset: [msambare/fer2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

---
