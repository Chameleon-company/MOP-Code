from pathlib import Path
import shutil

# Prepare DAWN dataset folders for YOLO
root = Path("C:/Users/ramad/Downloads/dawn_yolo")
img_root, lbl_root = root / "images", root / "labels"

# Ensure expected structure exists
for split in ["train", "val", "test"]:
    (img_root / split).mkdir(parents=True, exist_ok=True)
    (lbl_root / split).mkdir(parents=True, exist_ok=True)

print("DAWN dataset structure ready:", img_root)
