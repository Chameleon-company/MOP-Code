from pathlib import Path
import os, re, random, shutil
import cv2, pandas as pd
import xml.etree.ElementTree as ET

# === YOUR PATHS (leave quotes; your path has a space) ===
DAWN_RAW = Path(r"C:\Users\ramad\Downloads\dawn dataset\DAWN")
OUT_ROOT = Path(r"C:\Users\ramad\Downloads\dawn_yolo")
# =======================================================

for p in ["images/train","images/val","images/test","labels/train","labels/val","labels/test"]:
    (OUT_ROOT/p).mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg",".jpeg",".png",".bmp"}

def good_image(p: Path) -> bool:
    im = cv2.imread(str(p))               # returns empty/None if unreadable
    return im is not None and im.size>0   # simple validity check

def voc_to_yolo(xmin,ymin,xmax,ymax,W,H):
    cx=((xmin+xmax)/2)/W; cy=((ymin+ymax)/2)/H
    w=(xmax-xmin)/W; h=(ymax-ymin)/H
    return cx,cy,w,h

def find_image_for_xml(xml_path: Path, fname_text: str) -> Path | None:
    """
    Find the image referenced by a VOC XML.
    Try: (1) the XML folder, (2) sibling 'JPEGImages' or 'images',
         (3) parent folder, (4) global search under DAWN_RAW by stem.
    """
    # filename may be 'foggy-001.jpg' or 'foggy-001' (rare)
    fname = Path(fname_text)
    stem = fname.stem
    # 1) direct path as written in XML (relative to the xml folder)
    direct = (xml_path.parent / fname_text)
    if direct.exists():
        return direct

    # 2) common VOC layouts near the xml
    for cand_dir in [xml_path.parent,            # same folder
                     xml_path.parent / "JPEGImages",
                     xml_path.parent / "images",
                     xml_path.parent.parent / "JPEGImages",
                     xml_path.parent.parent / "images"]:
        if cand_dir and cand_dir.exists():
            # try exact filename first
            if fname.suffix:
                p = cand_dir / fname.name
                if p.exists(): return p
            # then try by stem with common extensions
            for ext in IMG_EXTS:
                p = cand_dir / f"{stem}{ext}"
                if p.exists(): return p

    # 3) fall back: global search in DAWN_RAW by stem
    for p in DAWN_RAW.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS and p.stem == stem:
            return p

    return None

# ---- detect annotation format (XML preferred for DAWN) ----
xml_list = list(DAWN_RAW.rglob("*.xml"))
csv_list = list(DAWN_RAW.glob("*.csv"))
pairs = []

if xml_list:
    found, missing = 0, 0
    for xml in xml_list:
        try:
            root = ET.parse(xml).getroot()
        except Exception:
            missing += 1
            continue

        fname_text = (root.findtext("filename") or "").strip()
        # some VOC files also include an absolute <path>; prefer filename logic
        img = find_image_for_xml(xml, fname_text) if fname_text else None
        if img is None or not good_image(img):
            missing += 1
            continue

        size = root.find("size")
        W = int(size.findtext("width")); H = int(size.findtext("height"))
        lines=[]
        for obj in root.findall("object"):
            bb=obj.find("bndbox")
            xmin=int(float(bb.findtext("xmin"))); ymin=int(float(bb.findtext("ymin")))
            xmax=int(float(bb.findtext("xmax"))); ymax=int(float(bb.findtext("ymax")))
            cx,cy,w,h = voc_to_yolo(xmin,ymin,xmax,ymax,W,H)
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")  # map all classes -> 'vehicle' (id 0)
        if lines:
            pairs.append((img, lines))
            found += 1
        else:
            missing += 1
    print(f"[INFO] XMLs processed: found={found}, missing={missing}")

elif csv_list:
    # If your DAWN happened to ship with a CSV of boxes
    csv_path = csv_list[0]
    df = pd.read_csv(csv_path)
    df['file'] = df['file'].astype(str)

    from collections import defaultdict
    groups = defaultdict(list)
    for _,r in df.iterrows():
        groups[r['file']].append(r)

    all_imgs = [p for p in DAWN_RAW.rglob("*") if p.suffix.lower() in IMG_EXTS]
    def find_by_rel(rel):
        rel = rel.replace("\\","/").lstrip("./")
        for p in all_imgs:
            if p.as_posix().endswith(rel): 
                return p
        return None

    for rel, rows in groups.items():
        img = find_by_rel(rel)
        if img is None or not good_image(img): 
            continue
        W,H = int(rows[0]["width"]), int(rows[0]["height"])
        lines=[]
        for r in rows:
            cx,cy,w,h = voc_to_yolo(r["xmin"],r["ymin"],r["xmax"],r["ymax"],W,H)
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        pairs.append((img, lines))
else:
    raise SystemExit("No .xml or .csv found under DAWN_RAW. Check your unzip path.")

# ---- split 75/15/10 and write ----
random.seed(42); random.shuffle(pairs)
n=len(pairs); tr,va,te = pairs[:int(.75*n)], pairs[int(.75*n):int(.90*n)], pairs[int(.90*n):]

def safe_name(p: Path)->str:
    return re.sub(r"[^0-9A-Za-z_-]","_", p.stem)+p.suffix.lower()

def write_one(p: Path, lines, split: str):
    img_dst = OUT_ROOT/f"images/{split}/{safe_name(p)}"
    lbl_dst = OUT_ROOT/f"labels/{split}/{img_dst.stem}.txt"
    shutil.copy2(p, img_dst)
    with open(lbl_dst,"w",encoding="utf-8") as f: f.write("\n".join(lines))

for group,split in [(tr,"train"),(va,"val"),(te,"test")]:
    for p,lines in group: write_one(p, lines, split)

print(f"Done. train={len(tr)} val={len(va)} test={len(te)}")
print("Output:", OUT_ROOT)
