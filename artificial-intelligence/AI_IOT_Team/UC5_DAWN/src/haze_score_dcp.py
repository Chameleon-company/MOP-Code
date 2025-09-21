from pathlib import Path
import cv2, numpy as np, csv

IMG_ROOT = Path("C:/Users/ramad/Downloads/dawn_yolo/images")
OUT = Path("artificial-intelligence/AI_IOT_Team/UC5_DAWN/results/haze_dcp.csv")
OUT.parent.mkdir(exist_ok=True, parents=True)

def dark_channel_prior(img, patch=15):
    min_channel = cv2.min(cv2.min(img[:,:,0], img[:,:,1]), img[:,:,2])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(min_channel, kernel).mean()

rows = []
for split in ["train","val","test"]:
    for p in (IMG_ROOT/split).glob("*.*"):
        im = cv2.imread(str(p))
        if im is not None:
            score = dark_channel_prior(im)
            rows.append([split, p.stem, score])

with open(OUT,"w",newline="",encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["split","image","haze_score_dcp"]); w.writerows(rows)

print("Saved:", OUT)
