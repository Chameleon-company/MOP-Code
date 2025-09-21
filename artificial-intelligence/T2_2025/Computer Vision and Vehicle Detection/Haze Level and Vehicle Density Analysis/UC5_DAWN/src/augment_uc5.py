# augment_uc5.py
# UC5: create extra training images with haze/blur/illumination while keeping YOLO bboxes valid.
# Works with Albumentations >= 1.4 (no deprecated args like filter_lost_elements).

from pathlib import Path
import argparse
import csv
import random
import sys
import cv2
import albumentations as A

# ---------- IO helpers ----------
def _valid_img(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png"}

def yolo_read(lbl_path: Path):
    """
    Read YOLO .txt -> (boxes, classes).
    Each line: <class> <x> <y> <w> <h>, all normalized to [0,1] (YOLO center format).
    """
    boxes, classes = [], []
    if not lbl_path.exists():
        return boxes, classes
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue  # skip malformed lines
        try:
            c, x, y, w, h = int(parts[0]), *map(float, parts[1:5])
        except ValueError:
            continue
        # basic validity check (tolerate tiny numeric noise)
        if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
            classes.append(c)
            boxes.append([x, y, w, h])
    return boxes, classes

def yolo_write(lbl_path: Path, boxes, classes):
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lbl_path, "w", newline="") as f:
        w = csv.writer(f, delimiter=" ")
        for b, c in zip(boxes, classes):
            w.writerow([int(c)] + [f"{v:.6f}" for v in b])

# ---------- augmentation pipeline ----------
def build_transform():
    """
    Albumentations pipeline focused on haze/blur/illumination + light geometry.
    BboxParams(format="yolo") tells Albumentations to transform YOLO (x_c, y_c, w, h) boxes. 
    """
    return A.Compose(
        [
            # photometric
            A.RandomBrightnessContrast(p=0.6),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=40, val_shift_limit=30, p=0.5),

            # haze / blur / noise
            A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.7, alpha_coef=0.08, p=0.35),
            A.MotionBlur(blur_limit=(3, 9), p=0.35),
            A.GaussNoise(var_limit=(10.0, 60.0), p=0.30),
            A.Defocus(radius=(2, 5), p=0.25),
            A.Downscale(scale_min=0.6, scale_max=0.9, p=0.25),

            # light geometry
            A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.05),
                     rotate=(-5, 5), shear=(-2, 2), p=0.5),
            A.HorizontalFlip(p=0.5),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.2,   # drop boxes that become tiny/hidden after transforms
        ),
    )

def clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="UC5 augmentation for YOLO (haze/blur/illumination)")
    ap.add_argument("--root", required=True, help="DAWN dataset root (contains images/ and labels/)")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"], help="Split to augment")
    ap.add_argument("--n", type=int, default=1, help="Augmented variants per image")
    ap.add_argument("--out_suffix", default="_aug", help="Suffix for augmented file stems")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    random.seed(args.seed)
    root = Path(args.root)

    img_in  = root / "images" / args.split
    lbl_in  = root / "labels" / args.split
    img_out = root / "images" / f"{args.split}_aug"
    lbl_out = root / "labels" / f"{args.split}_aug"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    if not img_in.exists() or not lbl_in.exists():
        print(f"[ERR] Missing split folders:\n  {img_in}\n  {lbl_in}", file=sys.stderr)
        sys.exit(1)

    images = sorted([p for p in img_in.iterdir() if p.is_file() and _valid_img(p)])
    if not images:
        print(f"[WARN] No images in {img_in}")
        return

    tfm = build_transform()
    written = 0
    skipped_imgs = 0
    skipped_all_boxes = 0

    for i, img_p in enumerate(images, 1):
        lbl_p = lbl_in / f"{img_p.stem}.txt"
        bboxes, classes = yolo_read(lbl_p)
        if not bboxes:
            skipped_imgs += 1
            continue

        img = cv2.imread(str(img_p))
        if img is None:
            skipped_imgs += 1
            continue

        for k in range(args.n):
            try:
                data = tfm(image=img, bboxes=bboxes, class_labels=classes)
            except Exception:
                # very rare numerical edge cases; safely skip this variant
                continue

            # clamp coords to [0,1] and drop invalid zero-area boxes
            new_boxes, new_cls = [], []
            for (x, y, w, h), c in zip(data["bboxes"], data["class_labels"]):
                x, y, w, h = clamp01(x), clamp01(y), clamp01(w), clamp01(h)
                if w > 0 and h > 0:
                    new_boxes.append([x, y, w, h])
                    new_cls.append(c)

            if not new_boxes:
                skipped_all_boxes += 1
                continue

            out_img = img_out / f"{img_p.stem}{args.out_suffix}{k}.jpg"
            out_lbl = lbl_out / f"{img_p.stem}{args.out_suffix}{k}.txt"
            cv2.imwrite(str(out_img), data["image"], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            yolo_write(out_lbl, new_boxes, new_cls)
            written += 1

        if i % 100 == 0:
            print(f"[{i}/{len(images)}] written={written}  skipped_imgs={skipped_imgs}  no_boxes={skipped_all_boxes}")

    print(f"Done. Augmented files written: {written}")
    print(f"Images skipped (no/invalid labels or unreadable): {skipped_imgs}")
    print(f"Aug variants dropped (all boxes vanished): {skipped_all_boxes}")
    print(f"Output images -> {img_out}")
    print(f"Output labels -> {lbl_out}")

if __name__ == "__main__":
    main()
