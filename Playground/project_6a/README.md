# Crack Detection Capstone

This repository contains notebook-based experiments for concrete crack detection on SDNET2018. The notes below focus on the two workflows currently captured in the repo:

- `01_crack_segmentation_benchmark.ipynb`
- `03_yolo11_sdnet2018_still_image_classification.ipynb`

## Setup

- Python: `>=3.12`
- Install dependencies: `uv sync`
- Place SDNET2018 under `DEPENDENCIES/SDNET2018` (preferred for review reproducibility). The notebooks still fall back to `dataset/` for local compatibility.
- Open and run the notebooks from the repository root so relative artifact paths resolve correctly.

The notebooks are written to use CUDA when available, but they can still run on CPU with slower training and evaluation.

## 01_crack

Notebook: `01_crack_segmentation_benchmark.ipynb`

### What it does

This notebook is designed to benchmark either:

- crack segmentation, if paired image and mask files exist under `data/crack_segmentation`
- cracked-vs-uncracked image classification, if only the raw SDNET2018 folder is available under `DEPENDENCIES/SDNET2018` (or `dataset/` fallback)

In the current repo state, the notebook detects raw SDNET2018 and automatically switches to classification mode because segmentation masks are not present.

### Current recorded run

- Dataset mode: classification fallback
- Raw class counts:
  - cracked: `8,484`
  - uncracked: `47,608`
- Stratified split:
  - train: `39,264`
  - val: `8,414`
  - test: `8,414`
- Positive rate stays close to `15.1%` across all splits
- Classification models benchmarked:
  - `resnet18`
  - `efficientnet_b0`
- Main training config:
  - input size: `224 x 224`
  - batch size: `16`
  - epochs: `8`
  - learning rate: `1e-4`
  - patience: `3`
  - mixed precision: enabled on CUDA

### Benchmark results

| Model | Accuracy | Precision | Recall | F1 | Best Val Metric | Train Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b0` | `94.60%` | `83.63%` | `79.95%` | `81.75%` | `82.50%` | `40.02 min` |
| `resnet18` | `94.46%` | `89.82%` | `71.46%` | `79.60%` | `80.07%` | `22.96 min` |

`efficientnet_b0` is the strongest recorded model in this run by test F1.

### Saved artifacts

- Summary table:
  - `artifacts/metrics/classification_benchmark_summary.csv`
  - `artifacts/metrics/classification_benchmark_summary.json`
- Per-model checkpoints:
  - `artifacts/checkpoints/classification_efficientnet_b0_best.pt`
  - `artifacts/checkpoints/classification_resnet18_best.pt`
- Per-model metrics:
  - `artifacts/metrics/classification_efficientnet_b0_test_metrics.json`
  - `artifacts/metrics/classification_resnet18_test_metrics.json`
- Plots:
  - `artifacts/plots/classification_efficientnet_b0_training_curves.png`
  - `artifacts/plots/classification_resnet18_training_curves.png`
  - `artifacts/plots/classification_efficientnet_b0_test_predictions.png`
  - `artifacts/plots/classification_resnet18_test_predictions.png`

### If you want true segmentation mode

Populate the expected segmentation dataset structure before running the notebook:

- `data/crack_segmentation/images/train`
- `data/crack_segmentation/images/val`
- `data/crack_segmentation/images/test`
- `data/crack_segmentation/masks/train`
- `data/crack_segmentation/masks/val`
- `data/crack_segmentation/masks/test`

When those image-mask pairs are available, the notebook is set up to benchmark:

- `unet_resnet34`
- `deeplabv3plus_resnet50`

## 03_yolo11

Notebook: `03_yolo11_sdnet2018_still_image_classification.ipynb`

### What it does

This notebook builds a binary crack-classification workflow with Ultralytics YOLO11:

- reads SDNET2018 from `DEPENDENCIES/SDNET2018` with fallback to `dataset/`
- converts the original surface/subclass folders into a binary `cracked` vs `uncracked` manifest
- creates a YOLO classification directory layout under `artifacts/yolo11_sdnet2018_binary_cls_data`
- trains `yolo11n-cls.pt`, which Ultralytics downloads automatically from the official assets release on first use
- includes helper cells for held-out test evaluation, confusion matrices, and still-image inference

### Current recorded run

- Binary class counts:
  - cracked: `8,484`
  - uncracked: `47,608`
- Stratified split:
  - train: `39,264`
  - val: `8,414`
  - test: `8,414`
- Training config:
  - model: `yolo11n-cls.pt`
  - image size: `224`
  - batch size: `64`
  - epochs: `20`
  - patience: `8`
  - seed: `42`
  - workers: `2`
  - device: CUDA if available, otherwise CPU
- Recorded validation top-1 accuracy at epoch 20: `94.26%`

### Pretrained weights

The checked-in weight files are excluded from the repository and should be fetched at runtime from the official Ultralytics assets release:

- `yolo11n-cls.pt`: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-cls.pt
- `yolo26n.pt`: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt

Ultralytics downloads these automatically the first time the relevant model is loaded.

### Saved artifacts

- Binary split manifest:
  - `artifacts/manifests/yolo11_sdnet2018_binary_split_manifest.csv`
- Prepared classification dataset:
  - `artifacts/yolo11_sdnet2018_binary_cls_data/`
- Dataset summary plots:
  - `artifacts/plots/yolo11_sdnet2018_binary_distribution.png`
  - `artifacts/plots/yolo11_sdnet2018_training_samples.png`
- Ultralytics training run:
  - `runs/classify/artifacts/yolo11_runs/yolo11n_sdnet2018_binary_cls/`
- Trained weights:
  - `runs/classify/artifacts/yolo11_runs/yolo11n_sdnet2018_binary_cls/weights/best.pt`
  - `runs/classify/artifacts/yolo11_runs/yolo11n_sdnet2018_binary_cls/weights/last.pt`
- Training metrics:
  - `runs/classify/artifacts/yolo11_runs/yolo11n_sdnet2018_binary_cls/results.csv`
  - `runs/classify/artifacts/yolo11_runs/yolo11n_sdnet2018_binary_cls/results.png`

### Reviewer note

The YOLO notebook now auto-resolves trained weights from either:

- `artifacts/yolo11_runs/yolo11n_sdnet2018_binary_cls/weights/best.pt`
- `runs/classify/artifacts/yolo11_runs/yolo11n_sdnet2018_binary_cls/weights/best.pt`

so evaluation, still-image inference, and confusion-matrix cells run end-to-end without manual `trained_weights_path` edits.

## Quick Start

1. Install dependencies with `uv sync`.
2. Place SDNET2018 inside `DEPENDENCIES/SDNET2018`.
3. Open either notebook and run from top to bottom.
4. Use the references section links for the external SDNET2018 source and Ultralytics pretrained weights.
