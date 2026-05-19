# Car damage detection model using YOLO
import kagglehub
import shutil
from ultralytics import YOLO
import yaml
import cv2
import json
from pycocotools.coco import COCO
from tqdm import tqdm
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Configuration
DAMAGE_EPOCHS = 50
DAMAGE_BATCH_SIZE = 32
DAMAGE_IMAGE_SIZE = 640
DAMAGE_MODEL_PATH = 'damage_detection_model_proper.pt'

print("="*50)
print("VEHICLE DAMAGE DETECTION MODEL (YOLO)")
print("="*50)

# Download dataset
print("Downloading Car Damage Assessment dataset...")
path = kagglehub.dataset_download("hamzamanssor/car-damage-assessment")
print("Path to dataset files:", path)

# Prepare dataset structure for YOLO
damage_dataset_path = 'damage_detection_dataset_proper'
if os.path.exists(damage_dataset_path):
    shutil.rmtree(damage_dataset_path)

# Create YOLO-compatible directory structure
os.makedirs(f'{damage_dataset_path}/train/images', exist_ok=True)
os.makedirs(f'{damage_dataset_path}/train/labels', exist_ok=True)
os.makedirs(f'{damage_dataset_path}/val/images', exist_ok=True)
os.makedirs(f'{damage_dataset_path}/val/labels', exist_ok=True)

# Function to convert COCO bbox to YOLO format
def coco_to_yolo(bbox, img_width, img_height):
    """Convert COCO bbox format [x,y,width,height] to YOLO format [x_center,y_center,width,height] (normalized)"""
    x, y, w, h = bbox
    x_center = (x + w/2) / img_width
    y_center = (y + h/2) / img_height
    width = w / img_width
    height = h / img_height
    return [x_center, y_center, width, height]

# Parse COCO annotations and convert to YOLO format
def convert_coco_to_yolo(coco_json_path, image_dir, output_label_dir, split_name):
    print(f"\nProcessing {split_name} annotations...")

    # Load COCO annotations
    try:
        coco = COCO(coco_json_path)
    except:
        # If COCO tools fail, parse manually
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        # Create category mapping
        categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}

        # Process images and annotations
        image_anns = {}
        for ann in coco_data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in image_anns:
                image_anns[img_id] = []
            image_anns[img_id].append(ann)

        # Get image info
        image_info = {img['id']: img for img in coco_data.get('images', [])}
    else:
        categories = {cat['id']: cat['name'] for cat_id, cat in coco.cats.items()}
        image_info = coco.imgs
        image_anns = {}
        for img_id in coco.getImgIds():
            ann_ids = coco.getAnnIds(imgIds=img_id)
            image_anns[img_id] = coco.loadAnns(ann_ids)

    # Get unique category names
    category_names = sorted(list(set(categories.values())))
    category_to_idx = {name: idx for idx, name in enumerate(category_names)}

    print(f"Found categories: {category_names}")

    # Process each image
    processed_count = 0
    for img_id, img_data in tqdm(image_info.items(), desc=f"Converting {split_name}"):
        # Get image filename
        if isinstance(img_data, dict):
            filename = img_data.get('file_name', '')
            width = img_data.get('width', 0)
            height = img_data.get('height', 0)
        else:
            continue

        if not filename or not width or not height:
            continue

        # Check if image exists
        img_path = os.path.join(image_dir, filename)
        if not os.path.exists(img_path):
            # Try without extension changes
            base_name = os.path.splitext(filename)[0]
            for ext in ['.jpg', '.jpeg', '.png']:
                alt_path = os.path.join(image_dir, base_name + ext)
                if os.path.exists(alt_path):
                    img_path = alt_path
                    break

        if not os.path.exists(img_path):
            continue

        # Copy image to destination
        dest_img_path = os.path.join(output_label_dir.replace('labels', 'images'), filename)
        shutil.copy2(img_path, dest_img_path)

        # Create YOLO label file
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(output_label_dir, label_filename)

        with open(label_path, 'w') as f:
            annotations = image_anns.get(img_id, [])
            for ann in annotations:
                if 'bbox' in ann and 'category_id' in ann:
                    bbox = ann['bbox']
                    cat_id = ann['category_id']

                    if cat_id in categories:
                        category_name = categories[cat_id]
                        class_idx = category_to_idx[category_name]

                        # Convert to YOLO format
                        yolo_bbox = coco_to_yolo(bbox, width, height)

                        # Write to file
                        f.write(f"{class_idx} {' '.join(map(str, yolo_bbox))}\n")

        processed_count += 1

    print(f"Processed {processed_count} images for {split_name}")
    return category_names

# Process train and validation data
train_categories = None
val_categories = None

# Find and process JSON files
json_files = {}
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.json'):
            if 'train' in file.lower():
                json_files['train'] = os.path.join(root, file)
            elif 'val' in file.lower() and 'mul' not in file.lower():
                json_files['val'] = os.path.join(root, file)

print(f"Found JSON files: {json_files}")

# Process JSON files if available
if json_files:
    # Process training data
    if 'train' in json_files:
        train_categories = convert_coco_to_yolo(
            json_files['train'],
            os.path.join(path, 'train'),
            f'{damage_dataset_path}/train/labels',
            'train'
        )

    # Process validation data
    if 'val' in json_files:
        val_categories = convert_coco_to_yolo(
            json_files['val'],
            os.path.join(path, 'val'),
            f'{damage_dataset_path}/val/labels',
            'val'
        )

# Fallback to direct folder processing if no valid JSON files
if not json_files or (train_categories is None and val_categories is None):
    print("\nNo valid COCO annotations found. Processing images directly...")

    # Function to process images without annotations
    def process_images_from_directory(base_path, is_train=True):
        image_files = []
        for root, _, files in os.walk(base_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))

        if not image_files:
            return []

        print(f"Found {len(image_files)} images")

        # Get split name
        split_name = "train" if is_train else "val"
        dest_dir = f"{damage_dataset_path}/{split_name}"

        # Process each image
        for img_path in tqdm(image_files, desc=f"Processing {split_name} images"):
            # Copy image
            img_name = os.path.basename(img_path)
            dest_img_path = os.path.join(dest_dir, 'images', img_name)
            shutil.copy2(img_path, dest_img_path)

            # Create label file (assume damage covers center of image)
            img = cv2.imread(img_path)
            if img is not None:
                # Create a centered bounding box covering ~60% of the image
                x_center, y_center = 0.5, 0.5  # Center of image
                bbox_width, bbox_height = 0.6, 0.6  # 60% coverage

                # Write label file (class_id x_center y_center width height)
                label_path = os.path.join(dest_dir, 'labels', os.path.splitext(img_name)[0] + '.txt')
                with open(label_path, 'w') as f:
                    f.write(f"0 {x_center} {y_center} {bbox_width} {bbox_height}")

        return ['damage']  # Default class

    # Look for train and val directories
    train_dir = None
    val_dir = None

    # Check for standard train/val split
    for root, dirs, _ in os.walk(path):
        if 'train' in dirs:
            train_dir = os.path.join(root, 'train')
        if 'val' in dirs or 'valid' in dirs or 'validation' in dirs or 'test' in dirs:
            for val_folder in ['val', 'valid', 'validation', 'test']:
                if val_folder in dirs:
                    val_dir = os.path.join(root, val_folder)
                    break

    # If no explicit split, create a random split from all images
    if train_dir is None and val_dir is None:
        print("No explicit train/val split found. Creating a random 80/20 split...")
        all_images = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_images.append(os.path.join(root, file))

        if all_images:
            random.shuffle(all_images)
            split_idx = int(0.8 * len(all_images))
            train_images = all_images[:split_idx]
            val_images = all_images[split_idx:]

            print(f"Creating train split with {len(train_images)} images")
            for img_path in tqdm(train_images, desc="Processing train images"):
                img_name = os.path.basename(img_path)
                dest_img_path = os.path.join(f"{damage_dataset_path}/train/images", img_name)
                shutil.copy2(img_path, dest_img_path)

                # Create label file
                label_path = os.path.join(f"{damage_dataset_path}/train/labels", os.path.splitext(img_name)[0] + '.txt')
                with open(label_path, 'w') as f:
                    f.write(f"0 0.5 0.5 0.6 0.6")  # Default box

            print(f"Creating val split with {len(val_images)} images")
            for img_path in tqdm(val_images, desc="Processing val images"):
                img_name = os.path.basename(img_path)
                dest_img_path = os.path.join(f"{damage_dataset_path}/val/images", img_name)
                shutil.copy2(img_path, dest_img_path)

                # Create label file
                label_path = os.path.join(f"{damage_dataset_path}/val/labels", os.path.splitext(img_name)[0] + '.txt')
                with open(label_path, 'w') as f:
                    f.write(f"0 0.5 0.5 0.6 0.6")  # Default box

            train_categories = ['damage']
        else:
            print("No images found in the dataset")
    else:
        # Process train and val directories if they exist
        if train_dir:
            train_categories = process_images_from_directory(train_dir, is_train=True)

        if val_dir:
            val_categories = process_images_from_directory(val_dir, is_train=False)

# Determine final categories
if train_categories:
    final_categories = train_categories
elif val_categories:
    final_categories = val_categories
else:
    # Fallback to default categories
    final_categories = ['damage', 'dent', 'scratch', 'crack', 'broken', 'shatter']

print(f"\nFinal categories for training: {final_categories}")

# Create data.yaml for YOLO
data_yaml = {
    'path': os.path.abspath(damage_dataset_path),
    'train': 'train/images',
    'val': 'val/images',
    'nc': len(final_categories),
    'names': final_categories
}

yaml_path = f'{damage_dataset_path}/data.yaml'
with open(yaml_path, 'w') as f:
    yaml.dump(data_yaml, f)

print(f"\nCreated data.yaml at {yaml_path}")

# Check if we have data
train_images = len(os.listdir(f'{damage_dataset_path}/train/images')) if os.path.exists(f'{damage_dataset_path}/train/images') else 0
val_images = len(os.listdir(f'{damage_dataset_path}/val/images')) if os.path.exists(f'{damage_dataset_path}/val/images') else 0

print(f"\nDataset summary:")
print(f"Train images: {train_images}")
print(f"Val images: {val_images}")
print(f"Classes: {final_categories}")

# Function to visualize dataset samples
def visualize_damage_samples(num_samples=9):
    """Visualize random samples from the damage dataset"""
    train_img_dir = f'{damage_dataset_path}/train/images'
    train_label_dir = f'{damage_dataset_path}/train/labels'

    if os.path.exists(train_img_dir) and train_images > 0:
        # Get all image files with corresponding labels
        image_files = [f for f in os.listdir(train_img_dir) if os.path.exists(
            os.path.join(train_label_dir, os.path.splitext(f)[0] + '.txt'))]

        # Sample images
        if len(image_files) > num_samples:
            sampled_images = random.sample(image_files, num_samples)
        else:
            sampled_images = image_files

        # Set up the plot
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        axes = axes.ravel()

        for i, img_file in enumerate(sampled_images):
            if i >= len(axes):
                break

            # Load image
            img_path = os.path.join(train_img_dir, img_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Load corresponding labels
            label_path = os.path.join(train_label_dir, os.path.splitext(img_file)[0] + '.txt')
            height, width = img.shape[:2]

            # Draw bounding boxes
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * width
                        y_center = float(parts[2]) * height
                        bbox_width = float(parts[3]) * width
                        bbox_height = float(parts[4]) * height

                        # Calculate bbox corners
                        x1 = int(x_center - bbox_width / 2)
                        y1 = int(y_center - bbox_height / 2)
                        x2 = int(x_center + bbox_width / 2)
                        y2 = int(y_center + bbox_height / 2)

                        # Draw rectangle
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Add label
                        if class_id < len(final_categories):
                            label = final_categories[class_id]
                            cv2.putText(img, label, (x1, y1-5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Display the image
            axes[i].imshow(img)
            axes[i].set_title(f'Sample {i+1}')
            axes[i].axis('off')

        # Hide unused subplots
        for j in range(len(sampled_images), len(axes)):
            axes[j].axis('off')

        plt.suptitle("Vehicle Damage Dataset Samples", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()
    else:
        print("No training images available to visualize.")

# Display dataset samples
print("\nVisualizing dataset samples...")
visualize_damage_samples()

if train_images > 0:
    # Initialize YOLO model
    model = YOLO('yolov8m.pt')  # Using medium model for better accuracy

    # Train the model
    print("\nStarting proper damage detection training...")
    results = model.train(
        data=yaml_path,
        epochs=DAMAGE_EPOCHS,
        imgsz=DAMAGE_IMAGE_SIZE,
        batch=DAMAGE_BATCH_SIZE,
        device=0 if torch.cuda.is_available() else 'cpu',
        project='damage_detection_runs_proper',
        name='train',
        exist_ok=True,
        verbose=True,
        patience=10,
        save=True,
        plots=True,
        amp=True,
        workers=8,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0
    )

    # Save the trained model
    model.save(DAMAGE_MODEL_PATH)
    print(f"\nModel saved to {DAMAGE_MODEL_PATH}")

    # Validate the model
    print("\nValidating damage detection model...")
    val_results = model.val()

    # Display results
    print("\nValidation Results:")
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")

    # Per-class results
    print("\nPer-class results:")
    for i, class_name in enumerate(final_categories):
        if i < len(val_results.box.ap50):
            print(f"{class_name}: AP50={val_results.box.ap50[i]:.4f}")

    # Visualize predictions
    def visualize_predictions(num_samples=3):
        """Visualize predictions on validation images"""
        val_img_dir = f'{damage_dataset_path}/val/images'
        if os.path.exists(val_img_dir):
            val_images = os.listdir(val_img_dir)
            if len(val_images) > 0:
                if len(val_images) > num_samples:
                    val_images = random.sample(val_images, num_samples)

                plt.figure(figsize=(12, 4*len(val_images)))
                for i, img_name in enumerate(val_images):
                    img_path = os.path.join(val_img_dir, img_name)

                    # Run prediction
                    results = model(img_path, conf=0.25)

                    # Plot results
                    plt.subplot(len(val_images), 1, i+1)
                    img_with_boxes = results[0].plot()
                    plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                    plt.title(f'Predictions for {img_name}')

                plt.tight_layout()
                plt.show()
            else:
                print("No validation images available for visualization.")
        else:
            print("Validation image directory not found.")

    # Show predictions
    print("\nVisualizing model predictions...")
    visualize_predictions()

else:
    print("\nNo training data found. Please check the dataset structure.")

# Function to test on new images
def test_damage_detection(image_path):
    """Test the damage detection model on a single image"""
    if os.path.exists(DAMAGE_MODEL_PATH):
        # Load the trained model
        damage_model = YOLO(DAMAGE_MODEL_PATH)

        # Run inference
        results = damage_model(image_path, conf=0.25)

        # Visualize results
        for result in results:
            img = result.plot()
            plt.figure(figsize=(8, 8))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title('Damage Detection Results')
            plt.show()

            # Print detections
            if len(result.boxes) > 0:
                print("\nDetections:")
                for box in result.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    class_name = result.names[class_id]
                    bbox = box.xyxy[0].cpu().numpy()
                    print(f"- {class_name}: confidence={confidence:.2f}, bbox={bbox}")
            else:
                print("No damage detected in the image.")
    else:
        print("No trained model found. Train the model first.")

print("\n" + "="*50)
print("Training complete!")
print("="*50)