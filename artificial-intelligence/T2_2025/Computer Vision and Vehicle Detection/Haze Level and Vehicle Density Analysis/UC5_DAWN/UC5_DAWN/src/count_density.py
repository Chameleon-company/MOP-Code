from pathlib import Path
import pandas as pd

ROOT = Path("C:/Users/ramad/Downloads/dawn_yolo/labels")
OUT = Path("artificial-intelligence/AI_IOT_Team/UC5_DAWN/results/vehicle_counts.csv")
OUT.parent.mkdir(exist_ok=True, parents=True)

rows = []
for split in ["train","val","test"]:
    for p in (ROOT/split).glob("*.txt"):
        with open(p) as f:
            count = sum(1 for _ in f)
        rows.append([split, p.stem, count])

pd.DataFrame(rows, columns=["split","image","vehicle_count"]).to_csv(OUT, index=False)
print("Saved:", OUT)
