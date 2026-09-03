# UC5: Vehicle Density & Haze (DAWN)

## Pipeline
1. Prepare dataset → `src/prepare_dawn.py`
2. Train YOLOv8 → `src/train_uc5.py`
3. Count vehicles → `src/count_density.py`
4. Compute haze (DCP) → `src/haze_score_dcp.py`
5. Merge features → `src/merge_features.py`
6. Analyze → `src/analyze_uc5.py`
7. Demo inference → `src/uc5_demo.py`

## Reproduce
```bash
python src/prepare_dawn.py
python src/train_uc5.py
python src/count_density.py
python src/haze_score_dcp.py
python src/merge_features.py
python src/analyze_uc5.py
python src/uc5_demo.py

