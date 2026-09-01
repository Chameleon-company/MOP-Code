"""
Vehicle Accident Detection Model
Uses MobileNetV2 backbone to classify vehicle images as accident/crash or normal
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import kagglehub
import random
from tqdm import tqdm
from PIL import Image

# Configuration
EPOCHS = 20
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
RANDOM_SEED = 42
MODEL_PATH = 'models/accident_detection_model.pt'
RESULTS_DIR = 'results/accident_detection'
DISPLAY_PLOTS = True  # Set to True to display plots

def set_seeds(seed=42):
    """Set random seeds for reproducibility across runs"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class AccidentImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform if transform else transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']
        label = row['label']

        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.transform:
                img = self.transform(img)

            return img, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a zero tensor as fallback
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), torch.tensor(label)

    def get_original_image(self, idx):
        """Get original image for visualization"""
        row = self.df.iloc[idx]
        img_path = row['path']
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img, row['label'], os.path.basename(img_path)
        except:
            return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8), row['label'], "error.jpg"

def get_train_transform():
    """Augmentation pipeline to prevent overfitting and improve generalization"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_test_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def setup_dataset(dataset_path):
    """Parse dataset with robust fallback mechanisms for inconsistent datasets"""
    print(f"Setting up dataset from: {dataset_path}")

    # Find the Excel annotation file
    xlsx_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith('.xlsx'):
                xlsx_files.append(os.path.join(root, file))

    if not xlsx_files:
        raise ValueError("Could not find annotation xlsx file in the dataset")

    # Use the first xlsx file found
    annotation_file = xlsx_files[0]
    print(f"Using annotation file: {annotation_file}")

    # Read the annotations
    try:
        annotations = pd.read_excel(annotation_file)
        print(f"Annotation file structure:\n{annotations.head()}")

        # Check for "Compressed" folder
        compressed_dir = os.path.join(dataset_path, 'Compressed')
        if os.path.exists(compressed_dir):
            print(f"Found Compressed directory: {compressed_dir}")
            image_dir = compressed_dir
        else:
            print("Compressed directory not found, using dataset root")
            image_dir = dataset_path

        # Create dataframe with image paths and labels
        data = []

        # Identify column names in the Excel file
        print("Column names in annotation file:", list(annotations.columns))

        # Find relevant columns (image filename and label)
        file_col = None
        label_col = None

        for col in annotations.columns:
            if 'file' in col.lower() or 'image' in col.lower() or 'name' in col.lower() or 'subject' in col.lower():
                file_col = col
            elif 'label' in col.lower() or 'class' in col.lower() or 'cat' in col.lower() or 'collision' in col.lower():
                label_col = col

        if file_col is None or label_col is None:
            print("Could not identify filename and label columns automatically.")
            print("Using first column as filename and second column as label.")
            file_col = annotations.columns[0]
            label_col = annotations.columns[1]

        print(f"Using '{file_col}' as filename column and '{label_col}' as label column")

        # Process each annotation
        missing_files = 0
        for _, row in tqdm(annotations.iterrows(), total=len(annotations), desc="Processing annotations"):
            filename = str(row[file_col])
            label = str(row[label_col]).lower()

            # Convert label to binary (1 for crash/collision, 0 for normal)
            binary_label = 1 if label in ['y', 'yes', '1', 'true', 'crash', 'collision'] else 0

            # Find image path
            img_path = None
            for root, _, files in os.walk(image_dir):
                for file in files:
                    if file.lower() == filename.lower() or file.lower() == f"{filename.lower()}.jpg":
                        img_path = os.path.join(root, file)
                        break
                if img_path:
                    break

            # Alternative: try direct path
            if not img_path:
                direct_path = os.path.join(image_dir, filename)
                if os.path.exists(direct_path):
                    img_path = direct_path
                elif os.path.exists(direct_path + '.jpg'):
                    img_path = direct_path + '.jpg'

            # Add to dataset if file exists
            if img_path and os.path.exists(img_path):
                data.append({'path': img_path, 'label': binary_label})
            else:
                missing_files += 1

        if missing_files > 0:
            print(f"Warning: {missing_files} image files referenced in annotations were not found")

        if not data:
            raise ValueError("No valid image files found that match the annotations")

        # Create DataFrame
        df = pd.DataFrame(data)

        # Print statistics
        print(f"Final dataset has {len(df)} images")
        print(f"Crash images: {df['label'].sum()}")
        print(f"Normal images: {len(df) - df['label'].sum()}")

        return df

    except Exception as e:
        print(f"Error processing annotation file: {e}")

        # Fallback: scan for images and try to infer labels from filenames
        print("Falling back to scanning image directory structure...")

        crash_images = []
        normal_images = []

        # Check if there are class-specific folders
        for root, dirs, files in os.walk(dataset_path):
            folder_name = os.path.basename(root).lower()
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    file_lower = file.lower()

                    # Try to infer class from folder name or filename
                    if ('y' in folder_name or 'yes' in folder_name or
                        'crash' in folder_name or 'collision' in folder_name or
                        'accident' in folder_name or
                        any(marker in file_lower for marker in ['crash', 'collision', 'accident', '_y_', '_y.', '_yes_'])):
                        crash_images.append(full_path)
                    elif ('n' in folder_name or 'no' in folder_name or
                          'normal' in folder_name or 'non' in folder_name or
                          any(marker in file_lower for marker in ['normal', 'non', '_n_', '_n.', '_no_'])):
                        normal_images.append(full_path)

        print(f"Found {len(crash_images)} crash images and {len(normal_images)} normal images by scanning directories")

        # Create DataFrame
        data = []
        for path in crash_images:
            data.append({'path': path, 'label': 1})  # 1 for crash

        for path in normal_images:
            data.append({'path': path, 'label': 0})  # 0 for normal

        if len(data) == 0:
            raise ValueError("Could not find any valid images in the dataset")

        df = pd.DataFrame(data)
        df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)  # Shuffle

        print(f"Final dataset has {len(df)} images")
        print(f"Crash images: {df['label'].sum()}")
        print(f"Normal images: {len(df) - df['label'].sum()}")

        return df

def visualize_dataset_samples(dataset, num_samples=10, filename='dataset_samples.png'):
    class_names = ['Normal', 'Crash']

    # Separate indices by class
    class_0_indices = [i for i, label in enumerate(dataset.df['label']) if label == 0]
    class_1_indices = [i for i, label in enumerate(dataset.df['label']) if label == 1]

    # Sample from each class
    indices = []
    if class_0_indices:
        indices.extend(random.sample(class_0_indices, min(num_samples//2, len(class_0_indices))))
    if class_1_indices:
        indices.extend(random.sample(class_1_indices, min(num_samples//2, len(class_1_indices))))

    # Create figure
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(indices):
        if i < num_samples:
            img, label, filename_img = dataset.get_original_image(idx)

            plt.subplot(2, 5, i+1)
            plt.imshow(img)
            plt.title(f"{class_names[label]}: {filename_img}")
            plt.axis('off')

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    
    if DISPLAY_PLOTS:
        plt.show()
    else:
        plt.close()
        
    print(f"Dataset samples visualization saved to {os.path.join(RESULTS_DIR, filename)}")

class AccidentDetectionModel(nn.Module):
    """MobileNetV2-based model with heavy regularization to prevent overfitting"""
    def __init__(self, num_classes=2):
        super(AccidentDetectionModel, self).__init__()

        # Use MobileNetV2 as a lightweight backbone
        self.backbone = models.mobilenet_v2(weights='DEFAULT')
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        # Classifier with dropout and batch normalization for regularization
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def train_model(train_loader, val_loader, model, epochs=EPOCHS):
    """Train with class weighting and early stopping to handle imbalanced data"""
    print("\nTraining model...")

    # Calculate class weights to handle imbalanced data
    train_labels = torch.tensor([label for _, label in train_loader.dataset])
    class_counts = torch.bincount(train_labels)
    total = class_counts.sum().float()
    class_weights = total / (class_counts.float() * len(class_counts))
    print(f"Class weights: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)

    best_val_f1, best_epoch = 0.0, 0
    patience, patience_counter = 7, 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_preds.extend(predicted.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())

            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # Calculate training metrics
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average='binary', zero_division=0)
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='binary', zero_division=0)
        val_loss /= len(val_loader)

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        print(f'\nEpoch {epoch+1}/{epochs} Results:')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        print(f'Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}')
        print(f'Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}')

        # Save best model based on F1 score (better for imbalanced data)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f'New best model saved with val_f1: {best_val_f1:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping after {epoch+1} epochs (no improvement in F1 score)')
                break

    print(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch+1}")
    return best_val_f1

def test_model(model, test_loader, test_dataset):
    print("\nTesting model...")

    model.eval()
    test_preds, test_targets = [], []
    test_probs = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)
            test_preds.extend(predicted.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
            test_probs.extend(probs[:, 1].cpu().numpy())

    # Calculate metrics
    acc = accuracy_score(test_targets, test_preds)
    f1 = f1_score(test_targets, test_preds, average='binary', zero_division=0)

    print(f"\nTest Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(test_targets, test_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Crash'],
                yticklabels=['Normal', 'Crash'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    
    if DISPLAY_PLOTS:
        plt.show()
    else:
        plt.close()

    # Visualize example predictions
    visualize_example_predictions(test_dataset, test_preds, test_targets)

    return acc, f1

def visualize_example_predictions(test_dataset, predictions, targets, num_samples=5):
    """Visualize model performance with examples of correct and incorrect predictions"""
    # Find correct and incorrect predictions
    correct_indices = [i for i, (p, t) in enumerate(zip(predictions, targets)) if p == t]
    incorrect_indices = [i for i, (p, t) in enumerate(zip(predictions, targets)) if p != t]

    # Create a combined figure for both correct and incorrect predictions
    plt.figure(figsize=(15, 10))

    # Samples from correct predictions
    samples_correct = min(num_samples, len(correct_indices))
    if samples_correct > 0:
        plt.suptitle("Example Predictions", fontsize=16)
        indices = random.sample(correct_indices, samples_correct)

        for i, idx in enumerate(indices):
            img, label, filename = test_dataset.get_original_image(idx)
            pred = predictions[idx]

            plt.subplot(2, num_samples, i+1)
            plt.imshow(img)
            plt.title(f"Correct\nTrue: {'Crash' if label==1 else 'Normal'}")
            plt.axis('off')

    # Samples from incorrect predictions
    samples_incorrect = min(num_samples, len(incorrect_indices))
    if samples_incorrect > 0:
        indices = random.sample(incorrect_indices, samples_incorrect)

        for i, idx in enumerate(indices):
            img, label, filename = test_dataset.get_original_image(idx)
            pred = predictions[idx]

            plt.subplot(2, num_samples, num_samples+i+1)
            plt.imshow(img)
            plt.title(f"Incorrect\nTrue: {'Crash' if label==1 else 'Normal'}\nPred: {'Crash' if pred==1 else 'Normal'}")
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'example_predictions.png'))
    
    if DISPLAY_PLOTS:
        plt.show()
    else:
        plt.close()

def load_model(model_path=MODEL_PATH):
    model = AccidentDetectionModel().to(DEVICE)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}. Using untrained model.")
    
    return model

def predict(model, image_path):
    """Make a prediction on a single image with confidence scores"""
    transform = get_test_transform()
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")
    
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(outputs, 1)
    
    is_crash = predicted.item() == 1
    crash_prob = probabilities[1].item()
    
    return {
        'is_crash': is_crash,
        'crash_probability': crash_prob,
        'normal_probability': probabilities[0].item(),
        'image': image
    }

def display_prediction(result):
    plt.figure(figsize=(8, 6))
    plt.imshow(result['image'])
    plt.title(f"Prediction: {'Crash' if result['is_crash'] else 'Normal'} ({result['crash_probability']:.2%})")
    plt.axis('off')
    
    prediction_text = (
        f"Crash: {result['crash_probability']:.2%}\n"
        f"Normal: {result['normal_probability']:.2%}"
    )
    plt.figtext(0.02, 0.02, prediction_text, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    if DISPLAY_PLOTS:
        plt.show()
    else:
        plt.close()

def prepare_dataset():
    """Download and prepare dataset with stratified splits to preserve class balance"""
    print("Downloading Car Crash or Collision Prediction dataset...")
    path = kagglehub.dataset_download("mdfahimbinamin/car-crash-or-collision-prediction-dataset")
    print("Path to dataset files:", path)
    
    df = setup_dataset(path)
    
    if df is None or len(df) == 0:
        print("Failed to create dataset. Exiting.")
        return None, None, None
    
    # Split dataset with stratification to maintain class balance
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, stratify=df['label'])
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=RANDOM_SEED, stratify=train_df['label'])
    
    print(f"Dataset splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df

def main():
    set_seeds()
    
    print("="*50)
    print("VEHICLE ACCIDENT DETECTION MODEL")
    print("="*50)
    print(f"Using device: {DEVICE}")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if os.path.exists(MODEL_PATH):
        print(f"Found existing model at {MODEL_PATH}")
        model = load_model()
    else:
        train_df, val_df, test_df = prepare_dataset()
        
        if train_df is None:
            print("Dataset preparation failed. Exiting.")
            return
            
        train_dataset = AccidentImageDataset(train_df, transform=get_train_transform())
        val_dataset = AccidentImageDataset(val_df, transform=get_test_transform())
        test_dataset = AccidentImageDataset(test_df, transform=get_test_transform())
        
        visualize_dataset_samples(train_dataset, filename='train_samples.png')
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        model = AccidentDetectionModel().to(DEVICE)
        
        train_model(train_loader, val_loader, model)
        
        model = load_model()
        test_acc, test_f1 = test_model(model, test_loader, test_dataset)
        
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
    
    print("\n" + "="*50)
    print("ACCIDENT DETECTION SETUP COMPLETE")
    print("="*50)

if __name__ == "__main__":
    main()