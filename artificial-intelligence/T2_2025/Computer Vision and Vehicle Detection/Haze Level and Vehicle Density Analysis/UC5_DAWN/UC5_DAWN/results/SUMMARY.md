## UC5 DAWN – Validation Results (latest)
## Weather Condition Predictions

- **Fog:** results/pred_fog (123 images processed)
- **Rain:** results/pred_rain (145 images processed)
- **Sand:** results/pred_sand (130 images processed)
- **Snow:** results/pred_snow (110 images processed)

All runs used the same YOLOv8 model (`best.pt`) trained on DAWN dataset.
Outputs include bounding-box predictions, confidence scores, and per-image labels.


**Dataset:** see plots  
**Images:** 923  
**Classes:** 1 (vehicle)

- Precision (P): 0.892
- Recall (R):    0.803
- mAP@0.50:      0.900
- mAP@0.50–0.95: 0.623

Artifacts: PR/F1 curves, confusion matrices, and sample predictions copied into 
results/.
