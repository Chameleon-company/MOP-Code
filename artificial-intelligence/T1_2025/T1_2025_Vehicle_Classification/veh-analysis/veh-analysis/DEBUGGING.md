```markdown
# Computer Vision Models - Debugging Guide

## Overview
This document captures debugging solutions, common issues, and best practices developed during the implementation of three computer vision models:
- **Vehicle Damage Detection** (YOLO-based object detection)
- **Vehicle Classification** (EfficientNet-based classification)  
- **Accident Detection** (MobileNetV2-based binary classification)

## Quick Reference - Common Fixes
- **Dataset not loading**: Check annotation format, file paths, and extensions
- **Poor validation accuracy**: Implement class weighting and data augmentation
- **Overfitting**: Add dropout, batch norm, early stopping, and reduce learning rate
- **Out of memory**: Reduce batch size, use gradient accumulation, or mixed precision
- **Model not converging**: Check learning rate, loss function, and class balance

## Dataset Loading Issues

### Multiple Annotation Formats
**Problem**: Datasets come with different annotation formats (COCO JSON, Excel, folder structure)

**Solution**: Cascading fallback mechanisms
```python
# Primary: Try COCO format
try:
    coco = COCO(coco_json_path)
except:
    # Fallback: Manual JSON parsing
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    # Process manually...

# Last resort: Infer from folder structure
if not annotations_found:
    # Scan directories and infer labels from folder names
```

### Missing or Corrupted Files
**Problem**: Annotation files reference non-existent images

**Solution**: Robust file validation
```python
def find_image_file(image_dir, filename):
    # Try exact match
    img_path = os.path.join(image_dir, filename)
    if os.path.exists(img_path):
        return img_path
    
    # Try different extensions
    base_name = os.path.splitext(filename)[0]
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        alt_path = os.path.join(image_dir, base_name + ext)
        if os.path.exists(alt_path):
            return alt_path
    
    return None
```

### COCO to YOLO Format Conversion
**Problem**: Bounding box format differences

**Solution**: Proper coordinate conversion
```python
def coco_to_yolo(bbox, img_width, img_height):
    """Convert COCO [x,y,width,height] to YOLO [x_center,y_center,width,height] (normalized)"""
    x, y, w, h = bbox
    x_center = (x + w/2) / img_width
    y_center = (y + h/2) / img_height
    width = w / img_width
    height = h / img_height
    return [x_center, y_center, width, height]
```

## Class Imbalance Issues

### Poor Performance on Minority Classes
**Problem**: Model biased toward majority class

**Solutions Implemented**:

1. **Weighted Random Sampling**
```python
def create_weighted_sampler(dataset):
    class_counts = Counter(dataset.labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    weights = [class_weights[label] for label in dataset.labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)
```

2. **Focal Loss Implementation**
```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, class_weights=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.class_weights)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

3. **Class Weight Calculation**
```python
def calculate_class_weights(dataset):
    class_counts = Counter(dataset.labels)
    total_samples = sum(class_counts.values())
    class_weights = []
    
    for i in range(len(dataset.classes)):
        weight = total_samples / (len(dataset.classes) * class_counts[i])
        class_weights.append(weight)
    
    return torch.FloatTensor(class_weights)
```

## Overfitting Prevention

### Progressive Regularization Strategy
**Problem**: Model memorizing training data

**Solution**: Layer-by-layer regularization
```python
class RegularizedClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),  # High dropout for first layer
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Medium dropout
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),  # Lower dropout
            nn.Linear(64, num_classes)
        )
```

### Early Stopping Implementation
```python
best_val_acc = 0
patience_counter = 0
patience = 7

for epoch in range(epochs):
    # Training and validation...
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
```

## Memory and Performance Optimization

### Out of Memory (OOM) Fixes
**Problem**: CUDA out of memory errors

**Solutions**:
1. **Reduce batch size progressively**: 64 → 32 → 16 → 8
2. **Use gradient accumulation**:
```python
accumulation_steps = 4
for i, (inputs, targets) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

3. **Mixed precision training** (for YOLO):
```python
# Enable AMP in YOLO training
results = model.train(
    data=yaml_path,
    amp=True,  # Automatic Mixed Precision
    # other parameters...
)
```

### Gradient Issues
**Problem**: Exploding or vanishing gradients

**Solution**: Gradient clipping
```python
# Clip gradients to prevent explosion
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

## Model-Specific Debugging

### Vehicle Damage Detection (YOLO)
**Issues Encountered**:
- Empty label files causing training crashes
- Inconsistent image/label pairs
- COCO annotation parsing failures

**Solutions**:
```python
# Generate default labels for images without annotations
if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
    with open(label_path, 'w') as f:
        f.write(f"0 0.5 0.5 0.6 0.6")  # Default centered box
```

### Vehicle Classification (EfficientNet)
**Issues Encountered**:
- Severe class imbalance
- Model not learning rare vehicle types
- High validation loss despite good training metrics

**Solutions**:
- WeightedRandomSampler for balanced batches
- Focal Loss to focus on hard examples
- Extensive data augmentation for minority classes

### Accident Detection (MobileNetV2)
**Issues Encountered**:
- High false positive rate (critical for safety)
- Limited training data
- Binary classification instability

**Solutions**:
```python
# Heavy regularization for safety-critical application
self.classifier = nn.Sequential(
    nn.Dropout(0.6),  # High dropout
    nn.Linear(num_features, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(64, 2)
)
```

## Data Pipeline Debugging

### Essential Validation Functions
```python
def validate_dataset(dataset_path):
    """Comprehensive dataset validation"""
    issues = []
    
    # Check directory structure
    required_dirs = ['train', 'val', 'test']
    for dir_name in required_dirs:
        if not os.path.exists(os.path.join(dataset_path, dir_name)):
            issues.append(f"Missing {dir_name} directory")
    
    # Check file counts
    train_count = len(os.listdir(os.path.join(dataset_path, 'train')))
    val_count = len(os.listdir(os.path.join(dataset_path, 'val')))
    
    if val_count < train_count * 0.1:
        issues.append("Validation set too small (< 10% of training)")
    
    # Check class balance
    # ... add class distribution checks
    
    return issues

def visualize_batch(dataloader, num_samples=8):
    """Debug data pipeline by visualizing a batch"""
    batch = next(iter(dataloader))
    images, labels = batch
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        # Denormalize image
        img = images[i].permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f'Label: {labels[i].item()}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
```

## Best Practices Developed

### 1. Reproducibility
```python
def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

### 2. Robust Error Handling
```python
def safe_image_load(image_path):
    """Safely load image with fallbacks"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"OpenCV couldn't load {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        # Return placeholder image
        return np.zeros((224, 224, 3), dtype=np.uint8)
```

### 3. Configuration Management
```python
# Centralized configuration
CONFIG = {
    'BATCH_SIZE': 32,
    'LEARNING_RATE': 1e-4,
    'EPOCHS': 50,
    'IMAGE_SIZE': 224,
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'RANDOM_SEED': 42,
    'PATIENCE': 7,
    'MODEL_SAVE_PATH': 'models/',
    'RESULTS_DIR': 'results/'
}
```

## Troubleshooting Checklist

### Before Training
- [ ] Dataset structure validated (`train/`, `val/`, `test/` directories exist)
- [ ] Image-label pairs verified (matching counts)
- [ ] Class distribution analyzed (check for severe imbalance)
- [ ] Sample batch visualized (verify transforms work correctly)
- [ ] Data augmentation tested (ensure labels remain valid)
- [ ] File paths are absolute or consistently relative
- [ ] GPU memory sufficient for batch size
- [ ] Random seeds set for reproducibility

### During Training
- [ ] Loss decreasing for both train/validation
- [ ] Validation accuracy improving
- [ ] No NaN values in loss
- [ ] Learning rate appropriate (not too high/low)
- [ ] Gradients not exploding (use gradient clipping if needed)
- [ ] Memory usage stable (no memory leaks)
- [ ] Model checkpoints saving correctly

### After Training
- [ ] Test accuracy reasonable compared to validation
- [ ] Confusion matrix analyzed for class-specific issues
- [ ] Model predictions visualized on sample images
- [ ] False positives/negatives examined
- [ ] Model file saved and loadable

## Performance Optimization Tips

### Training Speed
- Use `num_workers=4-8` in DataLoader
- Enable `pin_memory=True` for GPU training
- Use mixed precision training (`amp=True`)
- Optimize batch size for your GPU memory
- Use `torch.compile()` for PyTorch 2.0+

### Memory Efficiency
- Clear cache regularly: `torch.cuda.empty_cache()`
- Use gradient checkpointing for large models
- Process images in smaller batches if needed
- Delete unused variables: `del variable_name`

### Model Architecture
- Start with pretrained models when possible
- Use appropriate model size for dataset size
- Add regularization progressively, not all at once
- Consider model ensemble for critical applications

## Common Error Messages & Solutions

### `CUDA out of memory`
- Reduce batch size
- Clear GPU cache: `torch.cuda.empty_cache()`
- Use gradient accumulation instead of large batches

### `RuntimeError: Expected all tensors to be on the same device`
```python
# Ensure all tensors on same device
inputs = inputs.to(device)
targets = targets.to(device)
model = model.to(device)
```

### `ValueError: Found input samples with inconsistent numbers of samples`
- Check that all arrays have same first dimension
- Verify train/validation split maintains consistency

### `FileNotFoundError: No such file or directory`
- Use absolute paths or verify working directory
- Check file extensions and case sensitivity
- Implement robust file finding functions

## Debugging Tools & Visualization

### Essential Plots
```python
def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training History - Loss')
    ax1.legend()
    
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_title('Training History - Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
```

## Version Information
- **PyTorch**: 2.0+
- **YOLO**: Ultralytics YOLOv8
- **Timm**: 0.9.0+
- **OpenCV**: 4.0+
- **Scikit-learn**: 1.0+

---

*Last updated: [5/26/2025]*
```
