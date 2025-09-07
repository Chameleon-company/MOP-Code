# UC5 — DAWN preprocessing → YOLOv8 training → features (vehicle density & haze)

## What this folder contains
- **Scripts:** `prepare_dawn.py`, `count_density.py`, `haze_score_dcp.py`, `merge_features.py`, `analyze_uc5.py`
- **Data config:** `data/dawn.yaml`
- **Results:** `results/uc5_features.csv`, `results/haze_vs_density.png`, `results/SUMMARY.md`, `results/predict_samples/`

## How to reproduce
```powershell
# 1) Activate environment
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1

# 2) Prepare + feature merge + analysis
python .\datascience\air-quality\uc5_dawn\src\merge_features.py
python .\datascience\air-quality\uc5_dawn\src\analyze_uc5.py
