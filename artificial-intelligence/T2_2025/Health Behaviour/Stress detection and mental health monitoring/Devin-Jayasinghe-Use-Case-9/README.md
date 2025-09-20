# FER2013 Facial Emotion Recognition with GPU-Optimized PyTorch

A high-performance facial emotion recognition system using deep learning, optimized for GPU training on the FER2013 dataset. This project implements an EfficientNetV2-S architecture with transfer learning to classify facial expressions into 7 emotion categories.

## 🎯 Project Overview

This repository contains a complete pipeline for training, evaluating, and deploying a facial emotion recognition model. The implementation leverages GPU acceleration, mixed precision training, and advanced optimization techniques to achieve real-time inference capabilities.

### Key Features

- **GPU-Optimized Training**: Full CUDA acceleration with mixed precision (FP16) training
- **State-of-the-art Architecture**: EfficientNetV2-S with custom classification head
- **Transfer Learning**: Two-phase training with backbone freezing and fine-tuning
- **Real-time Performance**: Inference speed >900 FPS (batch size 64) on RTX 3070 Ti
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and ROC curves
- **Production Ready**: Multiple export formats (PyTorch, TorchScript, ONNX)

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Model Size | 83.9 MB |
| Training Time | ~48 minutes |
| Inference Speed (BS=1) | 38.4 FPS |
| Inference Speed (BS=64) | 944.2 FPS |
| Total Parameters | 20,968,023 |
| Trainable Parameters | 790,535 (initial phase) |

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (tested on RTX 3070 Ti)
- CUDA 12.1+ and cuDNN 8.9+
- 8GB+ VRAM recommended

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FER2013_Model_Files.git
cd FER2013_Model_Files
```

2. Install required packages:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas matplotlib seaborn pillow tqdm scikit-learn tensorboard jupyter
```

3. Download the FER2013 dataset:
   - Visit [Kaggle FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013/data)
   - Download and extract the dataset
   - Place the files in the `data_fer2013/` folder with the following structure:
```
data_fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── sad/
    ├── surprise/
    └── neutral/
```

### Running the Project

1. Open the Jupyter notebook:
```bash
jupyter notebook fer2013_gpu_optimized.ipynb
```

2. Run all cells to train the model from scratch, or use the pre-trained models provided

### Using Pre-trained Models

Pre-trained models are included in the repository:

```python
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Load the model
checkpoint = torch.load('best_emotion_model_finetuned_gpu.pth')
model = EmotionClassifier(num_classes=7)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Predict emotion
def predict_emotion(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()

    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    return emotions[predicted_class], probabilities[0][predicted_class].item()
```

## 🏗️ Model Architecture

### Base Model
- **Architecture**: EfficientNetV2-S (pretrained on ImageNet)
- **Input Size**: 224 x 224 x 3 (RGB)
- **Feature Extraction**: Convolutional backbone with compound scaling

### Custom Classification Head
```
Dropout(0.3) → Linear(1280, 512) → ReLU → BatchNorm1d(512) →
Dropout(0.3) → Linear(512, 256) → ReLU → BatchNorm1d(256) →
Dropout(0.15) → Linear(256, 7)
```

### Training Strategy

1. **Phase 1: Transfer Learning** (30 epochs)
   - Frozen backbone weights
   - Train only classification head
   - Learning rate: 2e-4
   - Optimizer: AdamW with weight decay

2. **Phase 2: Fine-tuning** (20 epochs)
   - Unfreeze last 30 layers
   - Lower learning rate: 2e-5
   - Continue training entire network

## 📁 Repository Structure

```
FER2013_Model_Files/
├── fer2013_gpu_optimized.ipynb          # Main training notebook
├── best_emotion_model_gpu.pth           # Best model from initial training
├── best_emotion_model_finetuned_gpu.pth # Best fine-tuned model
├── fer2013_emotion_model_gpu_final.pth  # Final model with metadata
├── fer2013_emotion_model_torchscript.pt # TorchScript for deployment
├── fer2013_gpu_training_metadata.json   # Training configuration and metrics
├── data_fer2013/                        # Dataset folder (user provided)
├── logs/                                 # TensorBoard logs
└── README.md                             # This file
```

## 🔧 Configuration

Key training parameters (optimized for RTX 3070 Ti):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `IMG_SIZE` | 224 | Input image dimensions |
| `BATCH_SIZE` | 64 | Optimized for 8GB VRAM |
| `EPOCHS` | 100 | Maximum epochs (with early stopping) |
| `LEARNING_RATE` | 2e-4 | Initial learning rate |
| `MIXED_PRECISION` | True | FP16 training for speed |
| `NUM_WORKERS` | 4 | Data loading threads |

## 📈 Results

### Confusion Matrix
The model achieves strong performance on happy (91% precision) and surprise (74% precision) emotions, with more challenging results on fear and sad emotions due to visual similarity.

### Per-Class Performance

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Angry | 0.54 | 0.61 | 0.57 | 958 |
| Disgust | 0.67 | 0.71 | 0.69 | 111 |
| Fear | 0.54 | 0.39 | 0.45 | 1024 |
| Happy | 0.91 | 0.82 | 0.86 | 1774 |
| Sad | 0.52 | 0.50 | 0.51 | 1247 |
| Surprise | 0.74 | 0.83 | 0.78 | 831 |
| Neutral | 0.56 | 0.69 | 0.62 | 1233 |

## 🚄 Inference Performance

Benchmarked on NVIDIA GeForce RTX 3070 Ti:

| Batch Size | Avg Time (ms) | FPS |
|------------|---------------|-----|
| 1 | 26.07 | 38.4 |
| 8 | 23.10 | 346.4 |
| 16 | 20.93 | 764.5 |
| 32 | 35.62 | 898.4 |
| 64 | 67.78 | 944.2 |

## 🛠️ Deployment

The model is exported in multiple formats for different deployment scenarios:

1. **PyTorch (.pth)**: Full model checkpoint with training state
2. **TorchScript (.pt)**: Optimized for production inference
3. **ONNX (.onnx)**: Cross-platform deployment (optional)

### Example Deployment with TorchScript

```python
import torch

# Load TorchScript model
model = torch.jit.load('fer2013_emotion_model_torchscript.pt')
model.eval()

# Inference
with torch.no_grad():
    output = model(input_tensor)
```

## 📊 Monitoring Training

TensorBoard logs are generated during training:

```bash
tensorboard --logdir=./logs
```

Navigate to `http://localhost:6006` to view:
- Training/validation loss curves
- Accuracy metrics
- Learning rate scheduling
- GPU memory usage

## 🔮 Future Improvements

- [ ] Implement ensemble methods for improved accuracy
- [ ] Add data augmentation techniques (CutMix, MixUp)
- [ ] Optimize for mobile deployment (quantization, pruning)
- [ ] Create real-time webcam demonstration
- [ ] Expand to additional emotion categories
- [ ] Multi-GPU distributed training support

## 🙏 Acknowledgments

- FER2013 dataset creators and [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013/data) for hosting
- PyTorch team for the excellent deep learning framework
- EfficientNet authors for the model architecture
- NVIDIA for CUDA and mixed precision training support

---

**Note**: This project requires downloading the FER2013 dataset separately from Kaggle due to licensing restrictions. Please ensure you have the appropriate rights to use the dataset for your intended purpose.
