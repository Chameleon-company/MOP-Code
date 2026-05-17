# UC5: Air Quality from Traffic Feeds (DAWN)
- Dataset: DAWN (Fog/Rain/Snow/Sand) → VOC XML converted to YOLO (`prepare_dawn.py`).
- Training: YOLOv8n `epochs=20 imgsz=640` on vehicle class. Metrics in `runs/detect/train` + `val`.
- Inference: `yolo predict` saved examples in `runs/detect/predict`.  
- Features: `vehicle_density.csv` (count txt labels), `haze_dcp.csv` (Dark Channel Prior).  
- Combined table: `results/uc5_features.csv`; scatter `results/haze_vs_density.png`.  
- Reproduce: commands in repo.  
- Docs: YOLO CLI train/val/predict; DCP paper cited in code.  

