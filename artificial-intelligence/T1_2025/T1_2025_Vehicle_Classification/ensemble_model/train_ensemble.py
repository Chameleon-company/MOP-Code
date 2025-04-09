# ==============================================================================
#  Vehicle Classification System - PyTorch + YOLO (Final Enhancement Attempt)
# ==============================================================================
#
# GOAL: Improve classification balance for difficult classes without changing data.
#
# KEY STRATEGIES IMPLEMENTED:
#   1. PyTorch uses Focal Loss (gamma=2.5) to focus on harder examples.
#   2. PyTorch uses MixUp augmentation for regularization and feature robustness.
#   3. PyTorch uses Early Stopping based on VALIDATION MACRO F1-SCORE to prioritize balance.
#   4. YOLO uses standard 'auto' training ('yolov8l-cls').
#   5. Trains for 20 epochs (with early stopping for PyTorch).
#   6. Evaluates BEST INDIVIDUAL models on VALIDATION set and prints reports
#      --> User analyzes these reports to decide on final ensemble weights for future runs.
#   7. Evaluates the ENSEMBLE (using initial weights 1.0 each for this run) on VAL/TEST.
#   8. Visualizes ENSEMBLE predictions.
#   9. Comments improved for clarity.

import os
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score, f1_score
import warnings
import traceback
import shutil
import cv2
from pathlib import Path
import torch.nn.functional as F # Needed for Focal Loss
import contextlib # For conditional no_grad in evaluation

warnings.filterwarnings('ignore')

# -------------- PyTorch imports --------------
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm
import ttach as tta

# -------------- YOLO imports --------------
from ultralytics import YOLO


# ===============================
#      USER CONFIGURATION
# ===============================
DATASET_PATH = 'datasets/'
# --- Model/Output Filenames ---
MODEL_SUFFIX = 'final_enhancement_20e'
PT_MODEL_SAVE_PATH = f'efficientnet_best_macrof1_{MODEL_SUFFIX}.pth'
YOLO_MODEL_DIR = f'yolo_runs_{MODEL_SUFFIX}'
YOLO_BEST_MODEL_PATH = os.path.join(YOLO_MODEL_DIR, 'classify/train/weights/best.pt') 

ENS_VAL_CM_PATH = f'ensemble_confusion_matrix_val_{MODEL_SUFFIX}.png'
ENS_TEST_CM_PATH = f'ensemble_confusion_matrix_test_{MODEL_SUFFIX}.png'
PREDICTIONS_PLOT_PATH = f'prediction_examples_val_{MODEL_SUFFIX}.png'

# --- Training Parameters ---
RANDOM_SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 0 # Determined dynamically

PT_EPOCHS = 20 # Train longer, but use early stopping
YOLO_EPOCHS = 20

PT_LEARNING_RATE = 1e-4
# YOLO_LEARNING_RATE = 1e-4 # Using 'auto' optimizer for YOLO

# Early stopping patience (for PyTorch, based on Macro F1)
# Stop if validation Macro F1 doesn't improve for this many epochs
PT_PATIENCE = 5
YOLO_PATIENCE = 10 # Standard YOLO patience (less critical here)

# --- Focal Loss Gamma ---
FOCAL_LOSS_GAMMA = 2.5 # Tunable parameter (2.0, 2.5, 3.0 are common)

# --- MixUp Alpha ---
MIXUP_ALPHA = 0.4 # Alpha parameter for Beta distribution in MixUp (0.2-0.4 common)

# !!! Ensemble Weighting - PLACEHOLDERS FOR THIS RUN !!!
# Analyze the individual validation reports printed by the script
# to decide how you would adjust these for a future run.
PT_MODEL_WEIGHT = 1.0
YOLO_MODEL_WEIGHT = 1.2

# --- System ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0 # Set > 0 if using multi-processing for dataloader

# ===============================
# UTILITY FUNCTIONS
# ===============================
def set_seeds(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def sort_class_dirs_numerically(dir_list):
    """Sorts directory names based on trailing numbers (e.g., class_1, class_10)."""
    def sort_key(dir_name):
        try: return int(dir_name.split('_')[-1])
        except: return float('inf') # Place non-numeric names last
    return sorted(dir_list, key=sort_key)

# ===============================
# FOCAL LOSS IMPLEMENTATION
# ===============================
class FocalLoss(nn.Module):
    """
    Focal Loss implementation for multi-class classification.
    Reduces the loss contribution from easy examples, focusing on hard ones.
    """
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure targets are long integers
        targets = targets.long()
        # Calculate standard cross-entropy loss without reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # Calculate pt (probability of the true class)
        pt = torch.exp(-ce_loss)
        # Calculate focal loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # 'none'
            return focal_loss

# ===============================
# MIXUP IMPLEMENTATION
# ===============================
def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1 # No mixup

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''Calculates loss for mixed up targets'''
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ===============================
# DATA PREPARATION & AUGMENTATION
# ===============================
def make_train_transform():
    """Defines the transformations for the training set."""
    # Using reasonably strong standard augmentations
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])
    ])

def make_val_or_test_transform():
    """Defines the transformations for the validation/test set (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Utility to denormalize images for visualization
denormalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

class FastDataset(Dataset):
    """Custom Dataset class for loading images efficiently."""
    # (Dataset class remains largely the same as previous versions)
    def __init__(self, root_dir, split='train', transform=None,
                 reference_classes=None, reference_class_to_idx=None):
        if split not in ['train', 'val', 'test']: raise ValueError(f"Invalid split: {split}")
        self.split_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = []
        self.class_to_idx = {}
        if not os.path.isdir(self.split_dir): raise FileNotFoundError(f"Dir not found: {self.split_dir}")

        try:
            all_dirs = [d for d in os.listdir(self.split_dir) if os.path.isdir(os.path.join(self.split_dir, d))]
            potential_classes = sort_class_dirs_numerically(all_dirs)
        except Exception as e: raise ValueError(f"Error scanning class dirs: {e}")
        if not potential_classes: raise ValueError(f"No class subdirs found: {self.split_dir}")

        global NUM_CLASSES
        if split == 'train':
            self.classes = potential_classes
            NUM_CLASSES = len(self.classes)
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        else:
            if reference_classes is None or reference_class_to_idx is None: raise ValueError("Val/Test need reference classes.")
            self.classes = [c for c in potential_classes if c in reference_classes]
            if not self.classes: raise ValueError(f"No classes in '{split}' match train.")
            self.class_to_idx = {cls: reference_class_to_idx[cls] for cls in self.classes}

        allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        for class_name in self.classes:
            if class_name not in self.class_to_idx: continue # Skip if class not in reference
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(self.split_dir, class_name)
            try:
                for fname in os.listdir(class_dir):
                    if os.path.splitext(fname)[1].lower() in allowed_extensions:
                        self.image_paths.append(os.path.join(class_dir, fname))
                        self.labels.append(class_idx)
            except Exception as e: print(f"Warning: Error reading dir {class_dir}: {e}") # Keep this warning
        if not self.image_paths: raise ValueError(f"No valid image files found: {self.split_dir}")

        self.labels = np.array(self.labels) # Convert labels to numpy array

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            from PIL import Image, UnidentifiedImageError
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except (UnidentifiedImageError, OSError, IOError) as e:
            # Return a blank tensor and label -1 if image is corrupt
            print(f"Warning: Skipping corrupted image {img_path}: {e}") # Keep this warning
            return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)), -1
        except Exception as e:
            print(f"Warning: Unexpected error loading {img_path}: {e}") # Keep this warning
            return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)), -1

def make_weighted_sampler(dataset):
    """Creates a WeightedRandomSampler to handle class imbalance in training."""
    label_counts = Counter(dataset.labels)
    total = len(dataset)
    num_classes = len(label_counts)
    if num_classes == 0 or total == 0:
        print("Warning: Cannot create sampler, no classes/samples found.")
        return None

    # Calculate weight for each class (inverse frequency)
    class_weights_dict = {}
    for class_idx, count in label_counts.items():
        if count > 0:
             # Ensure class_idx is within the expected range if NUM_CLASSES is set globally
             if 0 <= class_idx < NUM_CLASSES:
                 class_weights_dict[class_idx] = total / (num_classes * count)
             else:
                 print(f"Warning: Class index {class_idx} out of range {NUM_CLASSES}. Using weight 1.0.")
                 class_weights_dict[class_idx] = 1.0 # Assign default weight
        else:
             class_weights_dict[class_idx] = 1.0 # Should not happen for existing labels

    # Create weights for each sample in the dataset
    sample_weights = [class_weights_dict.get(label, 1.0) for label in dataset.labels] # Use .get for safety
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
    return sampler

# ===============================
# PYTORCH MODEL (EfficientNet)
# ===============================
class EfficientNetModel(nn.Module):
    """PyTorch EfficientNet B0 model wrapper."""
    def __init__(self, num_classes, freeze_layers=False): # Always fine-tuning here
        super().__init__()
        try: self.model = timm.create_model('efficientnet_b0', pretrained=True)
        except Exception as e: raise RuntimeError(f"Failed to create timm model: {e}")
        num_ftrs = self.model.classifier.in_features
        # Replace classifier with a new one
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )
        # Fine-tuning the whole network (freeze_layers=False implicit)

    def forward(self, x): return self.model(x)

def train_efficient_model(model, train_loader, val_loader):
    """Trains the PyTorch model using Focal Loss, MixUp, and Macro F1 Early Stopping."""
    start_time = time.time()
    best_val_macro_f1 = -1.0 # Initialize best F1 score
    best_model_wts_path = PT_MODEL_SAVE_PATH # Save path
    epochs_no_improve = 0

    # Use Focal Loss
    print(f"PT Training uses Focal Loss (gamma={FOCAL_LOSS_GAMMA}) and MixUp (alpha={MIXUP_ALPHA})")
    criterion = FocalLoss(gamma=FOCAL_LOSS_GAMMA, reduction='mean')

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_optimize, lr=PT_LEARNING_RATE, weight_decay=1e-4)
    # Scheduler reduces LR if validation Macro F1 stagnates
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3, verbose=True)

    print(f"Starting PT Training ({PT_EPOCHS} epochs, stopping early based on Val Macro F1)...")
    model.to(DEVICE)

    # --- Training Loop ---
    for epoch in range(PT_EPOCHS):
        model.train() # Set model to training mode
        running_loss, total_samples = 0.0, 0

        for inputs, targets in train_loader:
            # Filter out samples with label -1 (loading errors)
            valid_idx = (targets != -1)
            inputs, targets = inputs[valid_idx], targets[valid_idx]
            if inputs.size(0) == 0: continue

            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Apply MixUp
            mixed_inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, MIXUP_ALPHA, DEVICE)

            optimizer.zero_grad()
            outputs = model(mixed_inputs)

            # Calculate loss using MixUp criterion
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

        epoch_train_loss = running_loss / total_samples if total_samples > 0 else 0

        # --- Validation Step ---
        model.eval() # Set model to evaluation mode
        val_loss_sum, val_total = 0.0, 0
        all_val_labels = []
        all_val_preds = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                valid_idx = (labels != -1)
                inputs, labels = inputs[valid_idx], labels[valid_idx]
                if inputs.size(0) == 0: continue

                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)

                # Calculate validation loss (use original criterion on non-mixed data)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss_sum += loss.item() * inputs.size(0)
                val_total += labels.size(0)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())

        epoch_val_loss = val_loss_sum / val_total if val_total > 0 else 0

        # Calculate Validation Macro F1 Score
        current_val_macro_f1 = 0.0
        if val_total > 0:
            current_val_macro_f1 = f1_score(all_val_labels, all_val_preds, average='macro', zero_division=0)

        print(f"Epoch {epoch+1}/{PT_EPOCHS}: Train Loss={epoch_train_loss:.4f} | Val Loss={epoch_val_loss:.4f}, Val Macro F1={current_val_macro_f1:.4f}")

        scheduler.step(current_val_macro_f1) # Step scheduler based on Macro F1

        # Check for improvement based on Macro F1
        if current_val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = current_val_macro_f1
            epochs_no_improve = 0
            # Save the best model weights
            try:
                torch.save(model.state_dict(), best_model_wts_path)
                print(f"  -> Best model saved (Macro F1: {best_val_macro_f1:.4f})")
            except Exception as e:
                print(f"Error saving PT model checkpoint: {e}")
        else:
            epochs_no_improve += 1
            print(f"  (No improvement in Val Macro F1 for {epochs_no_improve} epoch(s))")

        # Early Stopping
        if epochs_no_improve >= PT_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs based on Val Macro F1.")
            break

    elapsed = time.time() - start_time
    print(f"\nFinished PT training in {elapsed/60:.1f} min. Best Val Macro F1={best_val_macro_f1:.4f}")
    print(f"Best PT model weights saved to: {best_model_wts_path}")

    # Load the best weights found during training before returning
    if os.path.exists(best_model_wts_path):
        try:
            model.load_state_dict(torch.load(best_model_wts_path, map_location=DEVICE))
            print("Loaded best model weights for returning.")
        except Exception as e:
             print(f"Warning: Could not load best PT model weights after training: {e}")
    else:
        print("Warning: Best model file not found after training.")

    return model # Return the model object with best weights loaded

# ===============================
# YOLO MODEL TRAINING
# ===============================
def train_yolo_model(dataset_root_path):
    """Trains the YOLOv8 Large classification model."""
    print("\n--- Training YOLOv8 Classification Model ---")
    try:
        model = YOLO('yolov8l-cls.pt') # Use large classification model
        print("Starting YOLO training...")
        start_time = time.time()
        # Using 'auto' optimizer, default settings otherwise
        results = model.train(
            data=dataset_root_path,
            epochs=YOLO_EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            patience=YOLO_PATIENCE,
            optimizer='auto',
            project=YOLO_MODEL_DIR,
            name='classify/train',
            exist_ok=True,
            device=DEVICE.index if DEVICE.type == 'cuda' else 'cpu',
            verbose=True, # Show YOLO's progress
            workers=NUM_WORKERS,
            seed=RANDOM_SEED,
            deterministic=True
        )
        elapsed = time.time() - start_time
        print(f"YOLO training completed in {elapsed/60:.1f} min.")

        # Verify best model path exists
        best_path = YOLO_BEST_MODEL_PATH
        if not os.path.exists(best_path):
            # Fallback search if standard path isn't there (less likely now)
             possible_paths = list(Path(YOLO_MODEL_DIR).rglob('**/weights/best.pt'))
             if possible_paths:
                 best_path = str(max(possible_paths, key=os.path.getctime))
                 print(f"Found best YOLO model via search: {best_path}")
             else:
                 print("Error: No best.pt model found for YOLO.")
                 return None, None

        # Load the best model to get its class map
        print(f"Loading best YOLO model to get class map: {best_path}")
        yolo_model = YOLO(best_path)
        yolo_classes = yolo_model.names  # {idx: class_name}
        if not yolo_classes:
             print("Error: Loaded YOLO model has no class names.")
             return best_path, None # Return path but no map
        yolo_map = {name: idx for idx, name in yolo_classes.items()} # {name: idx} map

        print(f"YOLO model path: {best_path}")
        return best_path, yolo_map

    except Exception as e:
        print(f"An error occurred during YOLO training: {e}")
        traceback.print_exc()
        return None, None

# =========================================
# INDIVIDUAL MODEL EVALUATION FUNCTION
# =========================================
def evaluate_individual_model_for_report(model_ref, model_type, loader, class_names, yolo_map=None):
    """
    Evaluates a single model (PT object or YOLO path) on a loader
    and returns the classification report string.
    """
    if model_ref is None: return "Model reference is None."

    y_true = []
    y_pred = []
    eval_model = None
    pytorch_tta_model = None

    # Prepare model instance
    try:
        if model_type == 'pytorch':
            if not isinstance(model_ref, nn.Module): raise TypeError("PyTorch model must be nn.Module")
            model_ref.eval().to(DEVICE)
            tta_transforms = tta.Compose([tta.HorizontalFlip()]) # Use TTA for PT eval
            eval_model = tta.ClassificationTTAWrapper(model_ref, tta_transforms, merge_mode='mean')
        elif model_type == 'yolo':
            if not isinstance(model_ref, str) or not os.path.exists(model_ref): raise ValueError("YOLO model ref must be valid path string")
            if yolo_map is None: raise ValueError("yolo_map is required for YOLO evaluation")
            eval_model = YOLO(model_ref) # Load YOLO model
        else:
            return f"Unknown model type: {model_type}"
    except Exception as e:
        return f"Error preparing {model_type} model for evaluation: {e}"

    # Evaluation loop
    with torch.no_grad() if model_type == 'pytorch' else contextlib.nullcontext():
        for inputs, labels in loader:
            valid_idx = (labels != -1); inputs, labels = inputs[valid_idx], labels[valid_idx]
            if inputs.size(0) == 0: continue

            current_batch_true = labels.numpy()
            current_batch_pred = [-1] * inputs.size(0)

            for i in range(inputs.size(0)):
                img_tensor = inputs[i]
                pred_idx = -1
                try:
                    if model_type == 'pytorch':
                        probs_tensor = eval_model(img_tensor.unsqueeze(0).to(DEVICE))
                        probs = probs_tensor.squeeze().cpu().numpy()
                        if probs.shape == (len(class_names),): pred_idx = np.argmax(probs)
                    elif model_type == 'yolo':
                        bgr_img = _to_bgr_numpy(img_tensor) # Helper defined below
                        results = eval_model.predict(bgr_img, verbose=False)
                        if results and results[0].probs is not None:
                            yolo_probs = results[0].probs.data.cpu().numpy()
                            remapped_probs = np.zeros(len(class_names))
                            master_map = {name: idx for idx, name in enumerate(class_names)}
                            for class_name, master_idx in master_map.items():
                                if class_name in yolo_map:
                                    yolo_idx = yolo_map[class_name]
                                    if 0 <= yolo_idx < len(yolo_probs): remapped_probs[master_idx] = yolo_probs[yolo_idx]
                            if remapped_probs.shape == (len(class_names),): pred_idx = np.argmax(remapped_probs)
                except Exception as e: pred_idx = -1 # Error during prediction
                current_batch_pred[i] = pred_idx

            y_true.extend(current_batch_true)
            y_pred.extend(current_batch_pred)

    # Generate Classification Report
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if len(y_true) == 0: return "No valid samples found in loader for evaluation."

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        labels=list(range(len(class_names))),
        zero_division=0, digits=3
    )
    return report

# --- Helper for BGR conversion needed in evaluation ---
def _to_bgr_numpy(pt_img_tensor):
    """Converts a PyTorch tensor (CHW, normalized) to BGR numpy (HWC, 0-255)."""
    img = denormalize(pt_img_tensor.cpu()).numpy()
    img = np.transpose(img, (1, 2, 0)) # CHW -> HWC
    img = np.clip(img * 255, 0, 255).astype(np.uint8) # [0,1] -> [0,255]
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # RGB -> BGR
    return img_bgr

# ===============================
# ENSEMBLE MODEL (PT + YOLO)
# ===============================
class VehicleEnsemble:
    """Combines PyTorch and YOLO models for ensemble predictions."""
    def __init__(self, pt_class_names):
        self.models = []
        self.master_class_names = pt_class_names
        self.num_classes = len(pt_class_names)
        self.master_class_to_idx = {name: i for i, name in enumerate(pt_class_names)}
        self.master_idx_to_class = {i: name for i, name in enumerate(pt_class_names)}
        self.tta_transforms = tta.Compose([tta.HorizontalFlip()]) # TTA for PT model in ensemble

    def add_pytorch_model(self, model, weight):
        """Adds a trained PyTorch model object to the ensemble."""
        if not isinstance(model, nn.Module):
            print("Error: Invalid PyTorch model object provided.")
            self.models.append({'type': 'pytorch', 'status': 'failed_load', 'weight': weight})
            return
        try:
            model.eval().to(DEVICE)
            # Wrap the PT model in TTA for consistent ensemble prediction
            tta_model = tta.ClassificationTTAWrapper(model, self.tta_transforms, merge_mode='mean')
            self.models.append({'type': 'pytorch', 'model': tta_model, 'weight': weight, 'map': self.master_class_to_idx, 'status': 'active'})
            print(f"Added PyTorch model to ensemble (Weight: {weight:.2f})")
        except Exception as e:
            print(f"Error preparing PT model for ensemble: {e}")
            self.models.append({'type': 'pytorch', 'status': 'failed_load', 'weight': weight})

    def add_yolo_model(self, model_path, yolo_map, weight):
        """Adds a trained YOLO model (via path) to the ensemble."""
        if not model_path or not os.path.exists(model_path):
            print(f"Error: YOLO model path not found: {model_path}")
            self.models.append({'type': 'yolo', 'status': 'failed_load', 'weight': weight})
            return
        if yolo_map is None:
             print("Error: YOLO class map is missing.")
             self.models.append({'type': 'yolo', 'status': 'failed_load', 'weight': weight})
             return
        try:
            # Load the YOLO model instance here
            yolo_model_instance = YOLO(model_path)
            # Check if necessary classes are present (optional but good practice)
            missing_classes = set(self.master_class_names) - set(yolo_map.keys())
            if missing_classes:
                 print(f"Warning: YOLO model map missing classes: {missing_classes}")

            self.models.append({'type': 'yolo', 'model': yolo_model_instance, 'weight': weight, 'map': yolo_map, 'status': 'active'})
            print(f"Added YOLO model to ensemble (Weight: {weight:.2f}), Path: {model_path}")
        except Exception as e:
            print(f"Error loading YOLO model for ensemble: {e}")
            self.models.append({'type': 'yolo', 'status': 'failed_load', 'weight': weight})

    def predict_single_model(self, img_tensor, model_entry):
        """Generates predictions from one model within the ensemble."""
        model_type = model_entry['type']
        model_ref = model_entry['model'] # PT TTA wrapper or YOLO instance
        model_map = model_entry['map'] # PT uses master map, YOLO uses its own

        try:
            if model_type == 'pytorch':
                # PT model is already TTA wrapped, predict directly
                with torch.no_grad():
                    img_batch = img_tensor.unsqueeze(0).to(DEVICE)
                    probs = model_ref(img_batch) # TTA wrapper handles augmentation/merging
                return probs.squeeze().cpu().numpy()

            elif model_type == 'yolo':
                bgr_img = _to_bgr_numpy(img_tensor) # Use helper function
                results = model_ref.predict(bgr_img, verbose=False) # Predict using YOLO instance
                if results and results[0].probs is not None:
                    yolo_probs = results[0].probs.data.cpu().numpy()
                    # Remap YOLO output to master class indices
                    remapped = np.zeros(self.num_classes)
                    for class_name, master_idx in self.master_class_to_idx.items():
                        if class_name in model_map: # Use the model's specific map
                            yolo_idx = model_map[class_name]
                            if 0 <= yolo_idx < len(yolo_probs):
                                remapped[master_idx] = yolo_probs[yolo_idx]
                    return remapped
                else:
                    return np.zeros(self.num_classes) # Return zeros if no YOLO probs
            else:
                return np.zeros(self.num_classes)
        except Exception as e:
             print(f"Error during single model prediction ({model_type}): {e}")
             return np.zeros(self.num_classes) # Return zeros on error

    def predict_ensemble(self, img_tensor):
        """Generates weighted average predictions from all active models."""
        active_models = [m for m in self.models if m['status'] == 'active']
        if not active_models: return np.zeros(self.num_classes)

        all_weighted_probs = []
        total_weight = 0.0

        for m_entry in active_models:
            # Get prediction from the single model function
            probs = self.predict_single_model(img_tensor, m_entry)
            # Check if prediction is valid before adding
            if probs is not None and probs.shape == (self.num_classes,) and np.any(probs > 1e-9):
                weight = m_entry['weight']
                all_weighted_probs.append(probs * weight)
                total_weight += weight

        if not all_weighted_probs: return np.zeros(self.num_classes) # No valid predictions

        # Calculate weighted average
        ensemble_sum = np.sum(all_weighted_probs, axis=0)
        if total_weight > 0:
            ensemble_probs = ensemble_sum / total_weight
        else: # Fallback if all weights are zero (unlikely)
             ensemble_probs = np.mean(all_weighted_probs, axis=0)

        # Ensure probabilities sum to 1 (optional, good practice)
        # ensemble_probs = ensemble_probs / np.sum(ensemble_probs)
        return ensemble_probs

    def evaluate(self, loader, split_name="Val", cm_path=None):
        """Evaluates the full ensemble performance on a dataloader."""
        print(f"\n--- Evaluating ENSEMBLE on {split_name} set ---")
        y_true = []
        y_pred_ens = []
        start = time.time()

        for inputs, labels in loader:
            valid_idx = (labels != -1); inputs, labels = inputs[valid_idx], labels[valid_idx]
            if inputs.size(0) == 0: continue

            for i in range(inputs.size(0)):
                img_tensor = inputs[i]
                true_label = labels[i].item()
                y_true.append(true_label)

                # Get ensemble prediction
                probs = self.predict_ensemble(img_tensor)
                if probs is not None and probs.shape == (self.num_classes,) and np.any(probs > 1e-9):
                    y_pred = np.argmax(probs)
                else:
                    y_pred = -1 # Indicate prediction failure
                y_pred_ens.append(y_pred)

        end = time.time()
        print(f"Ensemble evaluation took {(end-start):.1f}s for {len(y_true)} samples.")

        # Calculate and print metrics
        y_true = np.array(y_true)
        y_pred_ens = np.array(y_pred_ens)
        if len(y_true) == 0:
             print("No samples evaluated.")
             return 0, 0

        # --- Report Generation ---
        print(f"--- Final ENSEMBLE Metrics ({split_name}) ---")
        # Note: Includes samples where prediction might have failed (pred=-1)
        report = classification_report(
            y_true, y_pred_ens,
            target_names=self.master_class_names,
            labels=list(range(self.num_classes)),
            zero_division=0, digits=3
        )
        print(report)

        # --- Confusion Matrix ---
        if cm_path:
            try:
                cm = confusion_matrix(y_true, y_pred_ens, labels=list(range(self.num_classes)))
                plt.figure(figsize=(max(8, self.num_classes*0.6), max(6, self.num_classes*0.5)))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=self.master_class_names,
                            yticklabels=self.master_class_names,
                            annot_kws={"size": 8})
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")
                plt.title(f"ENSEMBLE Confusion Matrix ({split_name} Set)")
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.yticks(rotation=0, fontsize=8)
                plt.tight_layout()
                plt.savefig(cm_path)
                print(f"Ensemble Confusion Matrix saved: {cm_path}")
                plt.close()
            except Exception as e:
                print(f"Warning: Could not save ENSEMBLE confusion matrix: {e}")

        # Return overall accuracy and weighted F1 (or macro F1 if preferred)
        acc = accuracy_score(y_true, y_pred_ens)
        f1_w = f1_score(y_true, y_pred_ens, average='weighted', zero_division=0)
        return acc, f1_w

# ===============================
# VISUALIZATION
# ===============================
def visualize_predictions(ensemble, loader, num_images=9):
    """Displays sample images with their true and predicted labels from the ENSEMBLE."""
    print("\n--- Visualizing Sample Ensemble Predictions ---")
    if not ensemble or not ensemble.models:
         print("Ensemble is empty, cannot visualize.")
         return
    if not hasattr(loader, '__iter__'):
         print("Invalid loader for visualization.")
         return

    from math import ceil, sqrt
    fig = plt.figure(figsize=(12, 12))
    shown = 0
    try:
        data_iter = iter(loader)
        while shown < num_images:
                inputs, labels = next(data_iter)
                valid_idx = (labels != -1); inputs, labels = inputs[valid_idx], labels[valid_idx]
                if inputs.size(0) == 0: continue

                for i in range(inputs.size(0)):
                    if shown >= num_images: break

                    img_tensor = inputs[i]
                    true_label_idx = labels[i].item()
                    true_name = ensemble.master_idx_to_class.get(true_label_idx, "Unknown")

                    # Get prediction from the full ensemble
                    probs = ensemble.predict_ensemble(img_tensor)

                    if probs is not None and probs.shape == (ensemble.num_classes,) and np.any(probs > 1e-9):
                        pred_idx = np.argmax(probs)
                        pred_name = ensemble.master_idx_to_class.get(pred_idx, "Error")
                        conf = probs[pred_idx]
                        color = "green" if pred_idx == true_label_idx else "red"
                    else:
                        pred_name = "FAIL"
                        conf = 0.0
                        color = "orange" # Indicate prediction failure

                    # Denormalize image for display
                    img_np = denormalize(img_tensor.cpu()).numpy().transpose(1, 2, 0)
                    img_np = np.clip(img_np, 0, 1)

                    # Add subplot
                    ax = fig.add_subplot(ceil(sqrt(num_images)), ceil(sqrt(num_images)), shown + 1)
                    ax.imshow(img_np)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f"Pred: {pred_name}\nTrue: {true_name}\nConf: {conf:.2f}", color=color, fontsize=9)
                    shown += 1
    except StopIteration:
         print("Reached end of data loader before showing desired number of images.")
    except Exception as e:
         print(f"Error during visualization: {e}")
         traceback.print_exc()

    if shown > 0:
        plt.suptitle("Sample Ensemble Predictions", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(PREDICTIONS_PLOT_PATH)
        print(f"Prediction visualization saved: {PREDICTIONS_PLOT_PATH}")
        plt.show() # Display the plot inline in Colab/Jupyter
    else:
         print("No images were visualized.")

# ===============================
# MAIN EXECUTION BLOCK
# ===============================
def main():
    """Main function to orchestrate data loading, training, evaluation."""
    script_start_time = time.time()
    set_seeds(RANDOM_SEED)

    print("="*70)
    print(" Vehicle Classification - PyTorch + YOLO (Final Enhancement Attempt)")
    print(f" PT_EPOCHS={PT_EPOCHS} (Early Stop: Val Macro F1), YOLO_EPOCHS={YOLO_EPOCHS}")
    print(f" PT Loss: Focal Loss (gamma={FOCAL_LOSS_GAMMA}), PT Aug: MixUp (alpha={MIXUP_ALPHA})")
    print("="*70)
    print(f"Device: {DEVICE} | Workers: {NUM_WORKERS}\n")

    # --- Path Checks ---
    if not os.path.isdir(DATASET_PATH): print(f"FATAL: Dataset root not found: {DATASET_PATH}"); return
    if not all(os.path.isdir(os.path.join(DATASET_PATH, s)) for s in ['train','val','test']): print(f"FATAL: Missing train/val/test subdir in {DATASET_PATH}"); return

    # --- Load Data ---
    print("--- Loading Data ---")
    try:
        train_transform = make_train_transform()
        val_test_transform = make_val_or_test_transform()
        train_ds = FastDataset(DATASET_PATH, 'train', transform=train_transform)
        class_names = train_ds.classes
        class_to_idx = train_ds.class_to_idx
        global NUM_CLASSES; num_classes = NUM_CLASSES
        if num_classes == 0: raise ValueError("Number of classes not determined.")
        print(f"Found {num_classes} classes: {class_names}")

        val_ds = FastDataset(DATASET_PATH, 'val', transform=val_test_transform, reference_classes=class_names, reference_class_to_idx=class_to_idx)
        test_ds = FastDataset(DATASET_PATH, 'test', transform=val_test_transform, reference_classes=class_names, reference_class_to_idx=class_to_idx)

        # Use WeightedRandomSampler for training data
        sampler = make_weighted_sampler(train_ds)
        if sampler is None: print("Warning: Weighted sampler creation failed, using standard shuffling.")
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, shuffle=(sampler is None), num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        print("DataLoaders created.")
    except Exception as e:
        print(f"FATAL ERROR during Data Loading: {e}"); traceback.print_exc(); return

    # --- Train PyTorch Model ---
    print("\n--- Training PyTorch Model ---")
    pt_model_obj = None
    best_pt_model = None # Will hold the trained model object with best weights
    try:
        pt_model_obj = EfficientNetModel(num_classes, freeze_layers=False)
        # Train function loads best weights back into the model object it returns
        best_pt_model = train_efficient_model(pt_model_obj, train_loader, val_loader)
        if not os.path.exists(PT_MODEL_SAVE_PATH):
             print("Warning: PyTorch best model file was not saved.")
             # Use the returned model object directly, hoping it has the best weights
    except Exception as e:
        print(f"ERROR during PyTorch training: {e}"); traceback.print_exc()
        best_pt_model = None # Ensure model is None if training fails

    # --- Train YOLO Model ---
    print("\n--- Training YOLO Model ---")
    best_yolo_model_path, yolo_cls_map = None, None
    try:
        best_yolo_model_path, yolo_cls_map = train_yolo_model(DATASET_PATH)
        if best_yolo_model_path is None:
            print("YOLO training seems to have failed to produce a model.")
    except Exception as e:
        print(f"ERROR during YOLO training: {e}"); traceback.print_exc()

    # --- INDIVIDUAL MODEL VALIDATION REPORTS (FOR WEIGHT TUNING) ---
    print("\n" + "="*70)
    print("--- INDIVIDUAL MODEL VALIDATION REPORTS (FOR WEIGHT TUNING) ---")
    print("Objective: Analyze these reports (esp. Recall/F1 for weak classes)")
    print("           to decide how to adjust PT_MODEL_WEIGHT/YOLO_MODEL_WEIGHT")
    print("           in the script config for a final *tuned* ensemble run.")
    print("           (This current run uses weights 1.0 each as placeholders).")
    print("="*70)

    pt_report_val = "PyTorch model training failed or model not available."
    if best_pt_model is not None:
        print("\n--- PyTorch Validation Report ---")
        pt_report_val = evaluate_individual_model_for_report(best_pt_model, 'pytorch', val_loader, class_names)
        print(pt_report_val)

    yolo_report_val = "YOLO model training failed or model path not found."
    if best_yolo_model_path and yolo_cls_map:
        print("\n--- YOLO Validation Report ---")
        yolo_report_val = evaluate_individual_model_for_report(best_yolo_model_path, 'yolo', val_loader, class_names, yolo_map=yolo_cls_map)
        print(yolo_report_val)
    print("="*70 + "\n")


    # --- Create and Evaluate Ensemble (Using PLACEHOLDER weights) ---
    print("\n--- Creating & Evaluating Ensemble (using initial weights) ---")
    ensemble = VehicleEnsemble(class_names)
    models_added = 0

    # Add PyTorch model object (which should have best weights loaded)
    if best_pt_model is not None:
        ensemble.add_pytorch_model(best_pt_model, PT_MODEL_WEIGHT)
        models_added +=1
    else: print("Skipping PyTorch model in ensemble (training failed/unavailable).")

    # Add YOLO model via path
    if best_yolo_model_path and yolo_cls_map:
        ensemble.add_yolo_model(best_yolo_model_path, yolo_cls_map, YOLO_MODEL_WEIGHT)
        models_added +=1
    else: print("Skipping YOLO model in ensemble (training failed/path unavailable).")

    # Evaluate ensemble if at least one model was added
    if models_added > 0:
        ensemble.evaluate(val_loader, "Validation", ENS_VAL_CM_PATH)
        ensemble.evaluate(test_loader, "Test", ENS_TEST_CM_PATH)
        # Visualize ensemble predictions
        visualize_predictions(ensemble, val_loader, num_images=9)
    else:
        print("\nNo models successfully trained or loaded. Skipping ensemble evaluation and visualization.")

    # --- End Script ---
    script_end_time = time.time()
    total_duration = script_end_time - script_start_time
    print("\n" + "="*70)
    print(f" Script completed in {total_duration / 60:.1f} minutes.")
    print("="*70)


if __name__ == "__main__":
    import multiprocessing
    # Set multiprocessing start method for compatibility, especially outside Linux
    try:
        if multiprocessing.get_start_method(allow_none=True) is None or multiprocessing.get_start_method() != 'spawn':
             multiprocessing.set_start_method('spawn', force=True)
        multiprocessing.freeze_support()
    except RuntimeError as e: pass # Ignore if already set or not possible
    except Exception as e: pass # Ignore other potential errors

    main()
