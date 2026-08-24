import kagglehub
import os
import json
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm
import yaml
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import gc

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

CONFIG = {
    'epochs': 20,  
    'batch_size': 16,
    'image_size': 640,
    'model_path': 'car_damage_detection_model.pt',
    'dataset_path': 'car_damage_dataset',
    'yolo_model': 'yolov8l.pt',  
    'patience': 15,
    'train_ratio': 0.8,
    'visualize_samples': 5,  
    'class_weights': True,
}

gc.collect()
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64,expandable_segments:True'

print("="*50)
print("VEHICLE DAMAGE DETECTION MODEL - ADVANCED VERSION")
print("="*50)

print("Downloading VehiDE Dataset...")
path = kagglehub.dataset_download("hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection")
print(f"Path to dataset files: {path}")

if os.path.exists(CONFIG['dataset_path']):
    shutil.rmtree(CONFIG['dataset_path'])

for split in ['train', 'val']:
    for folder in ['images', 'labels']:
        os.makedirs(os.path.join(CONFIG['dataset_path'], split, folder), exist_ok=True)

def find_files(base_path, extensions):
    result_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                result_files.append(os.path.join(root, file))
    return result_files

all_images = find_files(path, ['.jpg', '.jpeg', '.png'])
json_annotations = find_files(path, ['.json'])
print(f"Found {len(all_images)} images")
print(f"Found {len(json_annotations)} annotation files")

DAMAGE_CLASSES = ['rach', 'mop_lom', 'tray_son', 'mat_bo_phan']
damage_categories = DAMAGE_CLASSES
category_to_idx = {cat: idx for idx, cat in enumerate(damage_categories)}
print(f"Using damage categories: {damage_categories}")

def process_json_annotations(annotation_files, all_images):
    processed_data = []
    img_path_map = {os.path.basename(img): img for img in all_images}
    class_distribution = Counter()
    
    skipped_count = 0
    no_match_count = 0
    annotation_count = 0
    
    for ann_file in tqdm(annotation_files, desc="Processing JSON annotations"):
        try:
            with open(ann_file, 'r') as f:
                data = json.load(f)
                
            if isinstance(data, dict):
                for img_key, img_data in data.items():
                    if not isinstance(img_data, dict) or "name" not in img_data or "regions" not in img_data:
                        continue
                    
                    img_name = img_data["name"]
                    img_path = img_path_map.get(img_name)
                    if not img_path:
                        no_match_count += 1
                        continue
                    
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            skipped_count += 1
                            continue
                        img_height, img_width = img.shape[:2]
                    except Exception as e:
                        skipped_count += 1
                        continue
                    
                    objects = []
                    for region in img_data["regions"]:
                        if "class" not in region or "all_x" not in region or "all_y" not in region:
                            continue
                            
                        class_name = region["class"]
                        if class_name not in category_to_idx:
                            continue
                            
                        x_points = region["all_x"]
                        y_points = region["all_y"]
                        
                        if len(x_points) < 3 or len(y_points) < 3:
                            continue
                            
                        x_min, x_max = min(x_points), max(x_points)
                        y_min, y_max = min(y_points), max(y_points)
                        
                        width = x_max - x_min
                        height = y_max - y_min
                        
                        if width <= 0 or height <= 0:
                            continue
                            
                        x_center = x_min + width / 2
                        y_center = y_min + height / 2
                        
                        x_center /= img_width
                        y_center /= img_height
                        width /= img_width
                        height /= img_height
                        
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                            continue
                            
                        class_idx = category_to_idx[class_name]
                        objects.append((class_idx, x_center, y_center, width, height))
                        class_distribution[class_name] += 1
                        annotation_count += 1
                        
                    if objects:
                        processed_data.append({
                            'image_path': img_path,
                            'width': img_width,
                            'height': img_height,
                            'objects': objects
                        })
                        
        except Exception as e:
            skipped_count += 1
            continue
    
    print(f"Processed {len(processed_data)} images with {annotation_count} annotations")
    
    return processed_data, class_distribution

print("\nProcessing JSON annotations...")
processed_data, class_distribution = process_json_annotations(json_annotations, all_images)

if CONFIG['class_weights'] and damage_categories:
    total_instances = sum(class_distribution.values())
    inverse_weights = {cat: total_instances / (class_distribution.get(cat, 0) + 1) for cat in damage_categories}
    
    mean_weight = sum(inverse_weights.values()) / len(inverse_weights) if inverse_weights else 1.0
    if mean_weight > 0:
        normalized_weights = {cat: w / mean_weight for cat, w in inverse_weights.items()}
        print("\nClass weights for handling imbalance:")
        for cat, weight in normalized_weights.items():
            print(f"  {cat}: {weight:.2f}")
    else:
        normalized_weights = {cat: 1.0 for cat in damage_categories}
else:
    normalized_weights = {cat: 1.0 for cat in damage_categories}

def visualize_samples(data_list, num_samples=5, title="Sample Images"):
    if num_samples <= 0:
        return
    
    samples = random.sample(data_list, min(num_samples, len(data_list)))
    
    plt.figure(figsize=(15, num_samples * 5))
    for i, sample in enumerate(samples):
        img = cv2.imread(sample['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        for obj in sample['objects']:
            class_idx, x_center, y_center, width, height = obj
            class_name = damage_categories[class_idx]
            
            x_min = int((x_center - width/2) * sample['width'])
            y_min = int((y_center - height/2) * sample['height'])
            x_max = int((x_center + width/2) * sample['width'])
            y_max = int((y_center + height/2) * sample['height'])
            
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(img, class_name, (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        plt.subplot(num_samples, 1, i+1)
        plt.imshow(img)
        plt.title(f"Sample {i+1}: {os.path.basename(sample['image_path'])}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.close()

if CONFIG['visualize_samples'] > 0:
    print("\nVisualizing sample images before training...")
    visualize_samples(processed_data, CONFIG['visualize_samples'], "Pre-Training Samples")

if len(processed_data) > 10:
    class_samples = {cat: [] for cat in damage_categories}
    
    for idx, item in enumerate(processed_data):
        if not item['objects']:
            continue
            
        class_counts = Counter([obj[0] for obj in item['objects']])
        if not class_counts:
            continue
            
        predominant_class_idx = max(class_counts.items(), key=lambda x: x[1])[0]
        if predominant_class_idx < len(damage_categories):
            predominant_class = damage_categories[predominant_class_idx]
            class_samples[predominant_class].append(idx)
    
    train_indices = []
    val_indices = []
    
    for cat, indices in class_samples.items():
        if not indices:
            continue
            
        random.shuffle(indices)
        split_idx = int(CONFIG['train_ratio'] * len(indices))
        train_indices.extend(indices[:split_idx])
        val_indices.extend(indices[split_idx:])
    
    if not train_indices or not val_indices:
        all_indices = list(range(len(processed_data)))
        random.shuffle(all_indices)
        split_idx = int(CONFIG['train_ratio'] * len(all_indices))
        train_indices = all_indices[:split_idx]
        val_indices = all_indices[split_idx:]
    
    train_data = [processed_data[i] for i in train_indices]
    val_data = [processed_data[i] for i in val_indices]

    print(f"Training images: {len(train_data)}")
    print(f"Validation images: {len(val_data)}")
    
    for split_name, split_data in [('train', train_data), ('val', val_data)]:
        print(f"Creating {split_name} dataset...")
        for idx, item in enumerate(tqdm(split_data, desc=f"Preparing {split_name} data")):
            image_filename = os.path.basename(item['image_path'])
            dest_image_path = os.path.join(f"{CONFIG['dataset_path']}/{split_name}/images", image_filename)
            
            if os.path.exists(dest_image_path):
                base, ext = os.path.splitext(image_filename)
                dest_image_path = os.path.join(f"{CONFIG['dataset_path']}/{split_name}/images", f"{base}_{idx}{ext}")
            
            try:
                shutil.copy2(item['image_path'], dest_image_path)
            except Exception:
                continue
            
            base_filename = os.path.basename(dest_image_path)
            label_filename = os.path.splitext(base_filename)[0] + '.txt'
            label_path = os.path.join(f"{CONFIG['dataset_path']}/{split_name}/labels", label_filename)
            
            with open(label_path, 'w') as f:
                for obj in item['objects']:
                    class_idx, x_center, y_center, width, height = obj
                    f.write(f"{class_idx} {x_center} {y_center} {width} {height}\n")
else:
    print(f"Warning: Only {len(processed_data)} images with annotations found. This may not be enough for training.")
    exit()

class_instances = {c: 0 for c in damage_categories}
for item in processed_data:
    for obj in item['objects']:
        class_idx = obj[0]
        if class_idx < len(damage_categories):
            class_name = damage_categories[class_idx]
            class_instances[class_name] += 1

print("\nClass distribution:")
for cls, count in sorted(class_instances.items(), key=lambda x: x[1], reverse=True):
    print(f"  {cls}: {count} instances")

data_yaml = {
    'path': os.path.abspath(CONFIG['dataset_path']),
    'train': 'train/images',
    'val': 'val/images',
    'nc': len(damage_categories),
    'names': damage_categories
}

yaml_path = f'{CONFIG["dataset_path"]}/data.yaml'
with open(yaml_path, 'w') as f:
    yaml.dump(data_yaml, f)

train_images_dir = f'{CONFIG["dataset_path"]}/train/images'
train_images = len(os.listdir(train_images_dir)) if os.path.exists(train_images_dir) else 0
val_images_dir = f'{CONFIG["dataset_path"]}/val/images'
val_images = len(os.listdir(val_images_dir)) if os.path.exists(val_images_dir) else 0

print(f"\nDataset summary: {train_images} train images, {val_images} validation images")

if train_images > 10:
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\nStarting YOLO model training...")
    model = YOLO(CONFIG['yolo_model'])
    
    if CONFIG['class_weights'] and damage_categories:
        class_weights_list = [normalized_weights.get(cat, 1.0) for cat in damage_categories]
        print(f"Using class weights: {class_weights_list}")
        
        with open(yaml_path, 'r') as f:
            yaml_content = yaml.safe_load(f)
        
        yaml_content['class_weights'] = class_weights_list
        
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f)
    
    if torch.cuda.is_available():
        free_mem = torch.cuda.get_device_properties(0).total_memory
        free_mem_gb = free_mem / (1024**3)
        print(f"GPU memory: {free_mem_gb:.2f} GB")
        
        if free_mem_gb < 8:
            CONFIG['batch_size'] = max(1, CONFIG['batch_size'] // 2)
            print(f"Adjusted batch size to {CONFIG['batch_size']} due to limited GPU memory")
    
    try:
        results = model.train(
            data=yaml_path,
            epochs=CONFIG['epochs'],
            imgsz=CONFIG['image_size'],
            batch=CONFIG['batch_size'],
            device=0 if torch.cuda.is_available() else 'cpu',
            project='car_damage_detection',
            name='train',
            exist_ok=True,
            verbose=False,
            patience=CONFIG['patience'],
            save=True,
            plots=True,
            
            cache=False,
            half=True,
            rect=False,
            overlap_mask=False,
            single_cls=False,
            workers=2,
            
            augment=True,
            mixup=0.1,
            mosaic=0.7,
            copy_paste=0.1,
            
            cos_lr=True,
            lr0=0.0005,
            lrf=0.05,
            warmup_epochs=2.0,
            
            box=4.5,
            cls=0.7,
            dfl=1.5,
            
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.2,
            scale=0.5,
            fliplr=0.5,
            perspective=0.001
        )
        
        model.save(CONFIG['model_path'])
        print(f"\nModel saved to {CONFIG['model_path']}")
        
        print("\nValidating model...")
        val_results = model.val()
        
        print("\nValidation Results:")
        print(f"mAP50: {val_results.box.map50:.4f}")
        print(f"mAP50-95: {val_results.box.map:.4f}")
        
        if CONFIG['visualize_samples'] > 0:
            print("\nVisualizing sample predictions after training...")
            
            val_img_dir = f"{CONFIG['dataset_path']}/val/images"
            val_images = [os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            if val_images:
                samples = random.sample(val_images, min(CONFIG['visualize_samples'], len(val_images)))
                
                plt.figure(figsize=(15, len(samples) * 5))
                for i, img_path in enumerate(samples):
                    results = model.predict(img_path, conf=0.25)
                    
                    fig = results[0].plot()
                    img = cv2.cvtColor(fig, cv2.COLOR_BGR2RGB)
                    
                    plt.subplot(len(samples), 1, i+1)
                    plt.imshow(img)
                    plt.title(f"Prediction {i+1}: {os.path.basename(img_path)}")
                    plt.axis('off')
                
                plt.tight_layout()
                plt.savefig("post_training_predictions.png")
                plt.close()
        
        print("\n" + "="*50)
        print("Training complete!")
        print("="*50)
    except Exception as e:
        print(f"Error during training: {str(e)}")
else:
    print("\nInsufficient data for training. Need at least 10 images.")
