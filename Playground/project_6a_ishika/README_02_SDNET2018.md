# 02_sdnet2018_data_exploration.ipynb

## Overview

This notebook performs comprehensive **data exploration, analysis, and preparation** for the SDNET2018 dataset—a collection of structural surface images (deck, pavement, and wall) labeled for crack detection. The notebook profiles class distribution, creates balanced data splits, and previews augmentation techniques for training crack-detection models.

## Dataset Structure

The SDNET2018 dataset follows a hierarchical directory structure:

```
dataset/
├── D/           # Deck images
│   ├── CD/      # Cracked deck
│   └── UD/      # Uncracked deck
├── P/           # Pavement images
│   ├── CP/      # Cracked pavement
│   └── UP/      # Uncracked pavement
└── W/           # Wall images
    ├── CW/      # Cracked wall
    └── UW/      # Uncracked wall
```

## Key Features

### 1. **Data Discovery & Manifest Creation**
   - Scans the dataset folder structure
   - Builds a pandas DataFrame containing file paths, surface types, and crack labels
   - Exports manifests for downstream use

### 2. **Class Imbalance Analysis**
   - Visualizes the distribution of samples across:
     - Surface types (Deck, Pavement, Wall)
     - Crack status (Cracked vs. Uncracked)
   - Identifies imbalanced classes and suggests balancing strategies

### 3. **Data Splitting**
   - Creates stratified train/validation/test splits to maintain class distribution
   - Generates separate manifests for:
     - Training set (with balancing options)
     - Validation set
     - Test set

### 4. **Data Balancing**
   - Implements undersampling and oversampling strategies
   - Creates balanced manifests suitable for training models with class weighting or sampling techniques

### 5. **Augmentation Preview**
   - Uses **Albumentations** library for image transformations
   - Previews augmentations including:
     - Rotations, flips, and shifts
     - Brightness and contrast adjustments
     - Blur and noise injection
   - Helps identify effective augmentations for crack-detection tasks

## Outputs

The notebook generates:
- **`artifacts/manifests/`** — CSV files containing train/validation/test splits
- **`artifacts/plots/`** — Visualizations of class distribution and augmentation effects

## Dependencies

- `pandas` — Data manipulation
- `numpy` — Numerical operations
- `matplotlib` & `seaborn` — Plotting
- `albumentations` — Image augmentation
- `opencv-python` (cv2) — Image I/O
- `scikit-learn` — Train/test splitting

## Usage

1. Ensure the SDNET2018 dataset is located in `dataset/` (relative to the notebook directory)
2. Run the notebook sequentially to generate manifests and visualizations
3. Use the generated manifest CSVs for training downstream models

## Notes

- The notebook uses a random seed (42) for reproducibility
- All paths are configurable via the Configuration section
- Albumentations are previewed as plots for manual inspection before use in training pipelines
