import os

# Train YOLOv8 with strong aug + longer schedule
os.system(
    "python -m ultralytics yolo task=detect mode=train "
    "model=yolov8m.pt "
    "data=artificial-intelligence/AI_IOT_Team/UC5_DAWN/data/dawn.yaml "
    "hyp=artificial-intelligence/AI_IOT_Team/UC5_DAWN/hyp/uc5_strong_aug.yaml "
    "epochs=120 imgsz=896 batch=16 cos_lr=True patience=30 "
    "project=runs/detect name=uc5_aug_train"
)

# Validate best weights
os.system(
    "python -m ultralytics yolo task=detect mode=val "
    "model=runs/detect/uc5_aug_train/weights/best.pt "
    "data=artificial-intelligence/AI_IOT_Team/UC5_DAWN/data/dawn.yaml"
)
