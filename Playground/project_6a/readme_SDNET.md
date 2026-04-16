# 02 SDNET2018 Data Exploration

## Use Case

**Title:** SDNET2018 Data Exploration for Crack Detection  
**Author:** Ishika  
**Duration:** 360 minutes  
**Level:** Intermediate  
**Pre-requisite skills:** Python, OpenCV, computer vision, data exploration, Jupyter notebooks

## Notebook

- Notebook path: `02_sdnet2018_data_exploration.ipynb`
- Project: `Project 6a: Crack Detection`
- Dataset: SDNET2018 concrete crack images
- Task: exploratory data analysis for downstream crack-detection modelling

## Purpose

This notebook explores SDNET2018 before modelling. It builds an image-level manifest, checks dataset quality, analyses class balance, profiles image statistics, validates train/validation/test splits, prepares balanced training manifests, previews augmentation choices, and records dataset metadata with MLflow when available.

## Main Sections

- Use-case template and project context
- API v2.1 dataset access and local cache resolution
- Manifest creation from SDNET2018 folder structure
- Data quality audit for missing, unreadable, duplicate, and repeated images
- Dataset overview and image geometry summary
- Class balance and surface-wise crack-rate analysis
- Pixel intensity, texture, and edge-density exploration
- Representative sample and outlier image grids
- Stratified train/validation/test split creation
- Split validation and leakage checks
- Training-only balancing options
- Manifest export for downstream modelling
- Optional MLflow dataset tracking
- Augmentation preview
- Key findings and references

## Dataset Access

The notebook expects SDNET2018 to be available through an approved API v2.1 archive endpoint.

Environment variables:

- `SDNET2018_API_V21_URL`: approved SDNET2018 API v2.1 archive URL
- `SDNET2018_API_V21_TOKEN`: optional bearer token for the API endpoint
- `SDNET2018_DATASET_ROOT`: optional override for the local dataset cache path

Default dataset cache path:

- `dataset/`

If `dataset/` already contains the expected SDNET2018 structure, the notebook uses the cached copy. If the cache is missing, it downloads and extracts the API v2.1 archive into the configured dataset root.

Expected SDNET2018 folders:

- `dataset/D/CD`
- `dataset/D/UD`
- `dataset/P/CP`
- `dataset/P/UP`
- `dataset/W/CW`
- `dataset/W/UW`

## Confirmed Paths

These paths were checked from the repository root.

| Path | Status | Notes |
| --- | --- | --- |
| `02_sdnet2018_data_exploration.ipynb` | Exists | Main notebook |
| `dataset/` | Exists | Local SDNET2018 cache |
| `artifacts/` | Exists | Artifact root |
| `artifacts/manifests/` | Exists | Manifest export directory |
| `artifacts/plots/` | Exists | Plot export directory |
| `artifacts/downloads/` | Generated when needed | Created only when the API archive is downloaded |
| `artifacts/mlruns/` | Generated when needed | Created only when the MLflow cell runs with MLflow installed |

## Generated Manifests

| Path | Status | Purpose |
| --- | --- | --- |
| `artifacts/manifests/sdnet2018_manifest.csv` | Exists | Full image-level manifest |
| `artifacts/manifests/sdnet2018_split_manifest.csv` | Exists | Stratified train/validation/test split manifest |
| `artifacts/manifests/sdnet2018_train_balanced_downsample.csv` | Exists | Training split balanced by downsampling |
| `artifacts/manifests/sdnet2018_train_balanced_oversample.csv` | Exists | Training split balanced by oversampling |
| `artifacts/manifests/sdnet2018_train_binary_balanced.csv` | Exists | Training split balanced by binary crack label |

## Generated Plots

| Path | Status | Purpose |
| --- | --- | --- |
| `artifacts/plots/sdnet2018_class_balance.png` | Exists | Class and surface balance | 
| `artifacts/plots/sdnet2018_surface_crack_rate.png` | Exists | Crack percentage by surface |
| `artifacts/plots/sdnet2018_pixel_statistics.png` | Exists | Pixel intensity profiles |
| `artifacts/plots/sdnet2018_texture_edge_features.png` | Exists | Texture and edge-density features |
| `artifacts/plots/sdnet2018_outlier_review.png` | Exists | Outlier image review grid |
| `artifacts/plots/sdnet2018_sample_grid.png` | Exists | Representative sample grid |
| `artifacts/plots/sdnet2018_balancing_options.png` | Exists | Balancing comparison |
| `artifacts/plots/sdnet2018_augmentation_preview.png` | Exists | Augmentation preview |

## How To Run

1. Install project dependencies with `uv sync`.
2. Set `SDNET2018_API_V21_URL` if the dataset cache is not already available.
3. Open `02_sdnet2018_data_exploration.ipynb`.
4. Run cells from top to bottom.
5. Install MLflow separately if dataset tracking is required:

```powershell
uv add mlflow
```

The MLflow cell is optional. If MLflow is not installed, it prints an installation message and the rest of the notebook remains usable.

## Notes

- Validation and test distributions are kept untouched.
- Balancing is applied only to training manifests.
- The notebook uses Australian English in prose and helper naming.
- The notebook is for data exploration and dataset preparation, not model training.
