# UC5: DAWN preprocessing → YOLOv8 training → features (vehicle density & haze)

## What’s here
- src/*.py — prepare_dawn, count_density, haze_score_dcp, merge_features, analyze_uc5
- data/dawn.yaml — YOLO data config (update train/val/test paths for your machine)
- results/ — optional, generated locally (not tracked)

## Reproduce (from repo root, within venv)
python .\artificial-intelligence\AI_IOT_Team\UC5_DAWN\src\prepare_dawn.py
yolo task=detect mode=train data=artificial-intelligence/AI_IOT_Team/UC5_DAWN/data/dawn.yaml epochs=20 imgsz=640 batch=16
yolo task=detect mode=val   model=runs/detect/train/weights/best.pt data=artificial-intelligence/AI_IOT_Team/UC5_DAWN/data/dawn.yaml
python .\artificial-intelligence\AI_IOT_Team\UC5_DAWN\src\count_density.py
python .\artificial-intelligence\AI_IOT_Team\UC5_DAWN\src\haze_score_dcp.py
python .\artificial-intelligence\AI_IOT_Team\UC5_DAWN\src\merge_features.py
python .\artificial-intelligence\AI_IOT_Team\UC5_DAWN\src\analyze_uc5.py
