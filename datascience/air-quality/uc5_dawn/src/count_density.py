from pathlib import Path
import csv

ROOT = Path(r"C:\Users\ramad\Downloads\dawn_yolo")
out = ROOT / "vehicle_density.csv"

rows=[]
for split in ["train","val","test"]:
    lbl_dir = ROOT/f"labels/{split}"
    for lbl in lbl_dir.glob("*.txt"):
        n = sum(1 for line in lbl.read_text().splitlines() if line.strip())
        img = (ROOT/f"images/{split}/{lbl.stem}.jpg")
        if not img.exists():
            png = img.with_suffix(".png")
            img = png if png.exists() else img
        rows.append([split, img.as_posix(), n])

with out.open("w", newline="", encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["split","image","vehicle_count"]); w.writerows(rows)

print("Wrote:", out)
