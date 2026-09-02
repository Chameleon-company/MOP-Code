from pathlib import Path
import cv2, numpy as np, csv

IMG_ROOT = Path(r"C:\Users\ramad\Downloads\dawn_yolo\images")
OUT = Path(r"C:\Users\ramad\Downloads\dawn_yolo\haze_alt.csv")

def variance_of_laplacian(img):
    # Higher = sharper (less haze/blur)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def clahe_contrast(img, clip=2.0, tiles=(8,8)):
    # Higher ≈ more local contrast (can correlate with less haze)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    l2 = clahe.apply(l)
    # use std-dev of equalized L-channel as a simple contrast proxy
    return float(np.std(l2))

rows=[]
for split in ["train","val","test"]:
    for p in (IMG_ROOT/split).glob("*.*"):
        im = cv2.imread(str(p))
        if im is None: 
            continue
        rows.append([split, p.as_posix(),
                     round(variance_of_laplacian(im), 4),
                     round(clahe_contrast(im), 4)])

with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["split","image","lap_var","clahe_contrast"]); w.writerows(rows)
print("Wrote:", OUT, "rows:", len(rows))
