"""
Vehicle Classifier using EfficientNet-B0
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configuration
DATASET_PATH = 'datasets/'
MODEL_SAVE_PATH = 'models/vehicle_classifier.pth'
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class FocalLoss(nn.Module):
    """Focal Loss down-weights well-classified examples, focusing training on hard examples"""
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', class_weights=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.class_weights)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class VehicleDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = sorted([d for d in os.listdir(self.root_dir)
                             if os.path.isdir(os.path.join(self.root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms(split='train'):
    if split == 'train':
        """Apply data augmentation to improve model generalization and prevent overfitting"""
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def create_weighted_sampler(dataset):
    """Create sampler that oversamples minority classes to balance training distribution"""
    class_counts = Counter(dataset.labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    weights = [class_weights[label] for label in dataset.labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)

def calculate_class_weights(dataset):
    """Calculate weights inversely proportional to class frequency for loss function"""
    class_counts = Counter(dataset.labels)
    total_samples = sum(class_counts.values())
    class_weights = []

    for i in range(len(dataset.classes)):
        weight = total_samples / (len(dataset.classes) * class_counts[i])
        class_weights.append(weight)

    return torch.FloatTensor(class_weights).to(DEVICE)

class VehicleClassifier(nn.Module):
    """Transfer learning model using pretrained EfficientNet-B0 with custom classifier head"""
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True)
        self.dropout = nn.Dropout(0.3)  # Regularization to prevent overfitting
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.model.global_pool(x)
        x = self.dropout(x)
        x = self.model.classifier(x)
        return x

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_labels, all_predictions

def visualize_samples(dataset, num_samples=9, title="Sample Images"):
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.ravel()

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for i, idx in enumerate(indices):
        image, label = dataset[idx]

        # Convert from normalized tensor to displayable image
        image = image.permute(1, 2, 0).numpy()
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)

        axes[i].imshow(image)
        axes[i].set_title(f'{dataset.classes[label]}')
        axes[i].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def visualize_results(model, dataset, num_samples=9):
    """Visualize model predictions to identify error patterns and confusion between classes"""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    correct_samples = []
    incorrect_samples = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            if preds[0] == labels[0]:
                if len(correct_samples) < num_samples:
                    correct_samples.append((inputs[0].cpu(), labels[0].item(), preds[0].item()))
            else:
                if len(incorrect_samples) < num_samples:
                    incorrect_samples.append((inputs[0].cpu(), labels[0].item(), preds[0].item()))

            if len(correct_samples) >= num_samples and len(incorrect_samples) >= num_samples:
                break

    # Display correct predictions
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.ravel()

    for i, (image, true_label, pred_label) in enumerate(correct_samples[:num_samples]):
        image = image.permute(1, 2, 0).numpy()
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)

        axes[i].imshow(image)
        axes[i].set_title(f'True: {dataset.classes[true_label]}\nPred: {dataset.classes[pred_label]}')
        axes[i].axis('off')

    plt.suptitle('Correctly Classified Samples', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

    # Display incorrect predictions
    if len(incorrect_samples) > 0:
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        axes = axes.ravel()

        for i, (image, true_label, pred_label) in enumerate(incorrect_samples[:min(num_samples, len(incorrect_samples))]):
            image = image.permute(1, 2, 0).numpy()
            image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            image = np.clip(image, 0, 1)

            axes[i].imshow(image)
            axes[i].set_title(f'True: {dataset.classes[true_label]}\nPred: {dataset.classes[pred_label]}')
            axes[i].axis('off')

        for j in range(len(incorrect_samples), len(axes)):
            axes[j].axis('off')

        plt.suptitle('Incorrectly Classified Samples', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()
    else:
        print("No incorrect classifications found in the samples!")

def train_model(dataset_path=DATASET_PATH, model_save_path=MODEL_SAVE_PATH, epochs=EPOCHS, batch_size=BATCH_SIZE):
    set_seeds()
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    print("="*50)
    print("EfficientNet-B0 VEHICLE CLASSIFICATION MODEL")
    print("="*50)
    print(f"Device: {DEVICE}")
    print("="*50)

    train_dataset = VehicleDataset(dataset_path, 'train', get_transforms('train'))
    val_dataset = VehicleDataset(dataset_path, 'val', get_transforms('val'))
    test_dataset = VehicleDataset(dataset_path, 'test', get_transforms('test'))

    print("Displaying sample images...")
    display_dataset = VehicleDataset(dataset_path, 'train', get_transforms('val'))
    visualize_samples(display_dataset, title="Sample Training Images")

    class_weights = calculate_class_weights(train_dataset)
    print("\nClass weights:", class_weights)

    train_sampler = create_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    num_classes = len(train_dataset.classes)
    model = VehicleClassifier(num_classes).to(DEVICE)

    criterion = FocalLoss(gamma=2.0, class_weights=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Track best model for early stopping
    best_val_acc = 0
    best_model_state = None

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_labels, val_preds = validate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f'New best model saved! Val Acc: {val_acc:.4f}')

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, model_save_path)
        print(f"\nBest model saved to {model_save_path}")

    test_loss, test_acc, test_labels, test_preds = validate(model, test_loader, criterion, DEVICE)
    print(f'\nTest Accuracy: {test_acc:.4f}')

    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=train_dataset.classes))

    cm = confusion_matrix(test_labels, test_preds)
    plot_confusion_matrix(cm, train_dataset.classes)

    print("\nPer-class accuracy:")
    for i, class_name in enumerate(train_dataset.classes):
        class_mask = np.array(test_labels) == i
        class_correct = np.sum(np.array(test_preds)[class_mask] == i)
        class_total = np.sum(class_mask)
        class_acc = class_correct / class_total if class_total > 0 else 0
        print(f"{class_name}: {class_acc:.4f} ({class_correct}/{class_total})")

    print("\nDisplaying test results...")
    visualize_results(model, test_dataset)
    
    return model, train_dataset.classes

def load_model(model_path=MODEL_SAVE_PATH, dataset_path=DATASET_PATH):
    try:
        train_dir = os.path.join(dataset_path, 'train')
        classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    except:
        print("Warning: Could not determine class names from dataset. Using generic names.")
        classes = [f"Class_{i}" for i in range(10)]
    
    model = VehicleClassifier(len(classes)).to(DEVICE)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model file not found at {model_path}. Using untrained model.")
    
    return model, classes

def predict(model, image_path, classes):
    transform = get_transforms('test')
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    pred_class = classes[predicted.item()]
    pred_prob = probabilities[predicted.item()].item()
    
    top_probs, top_indices = torch.topk(probabilities, 3)
    top_predictions = [(classes[idx.item()], prob.item()) for idx, prob in zip(top_indices, top_probs)]
    
    return pred_class, pred_prob, top_predictions, image

def display_prediction(image, pred_class, pred_prob, top_predictions):
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(f"Prediction: {pred_class} ({pred_prob:.2%})")
    plt.axis('off')
    
    predictions_text = "\n".join([f"{cls}: {prob:.2%}" for cls, prob in top_predictions])
    plt.figtext(0.02, 0.02, f"Top predictions:\n{predictions_text}", bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(MODEL_SAVE_PATH):
        print("No trained model found. Training a new model...")
        model, classes = train_model()
    else:
        print("Loading trained model...")
        model, classes = load_model()
        
    print(f"Model ready with {len(classes)} classes: {classes}")