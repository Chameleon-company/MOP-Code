from pathlib import Path
import cv2, numpy as np, csv

IMG_ROOT = Path(r"C:\Users\ramad\Downloads\dawn_yolo\images")
OUT = Path(r"C:\Users\ramad\Downloads\dawn_yolo\haze_dcp.csv")

def dark_channel(img, r=7):
    m = cv2.min(cv2.min(img[:,:,0], img[:,:,1]), img[:,:,2])
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2*r+1, 2*r+1))
    return cv2.erode(m, k)

def haze_score(img):
    # lower dark-channel mean ⇒ heavier haze
    d = dark_channel(img, r=7)
    return 1.0 - (d.mean()/255.0)

rows=[]
for split in ["train","val","test"]:
    for p in (IMG_ROOT/split).glob("*.*"):
        im = cv2.imread(str(p))
        if im is None:   # OpenCV returns None/empty if unreadable
            continue
        rows.append([split, p.as_posix(), round(float(haze_score(im)),4)])

with OUT.open("w", newline="", encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["split","image","haze_score_dcp"]); w.writerows(rows)

print("Wrote:", OUT)
