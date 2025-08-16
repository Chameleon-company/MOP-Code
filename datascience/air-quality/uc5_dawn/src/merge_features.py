from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\ramad\Downloads\dawn_yolo")
out_dir = Path(r"C:\Users\ramad\Documents\MOP-Code\datascience\air-quality\uc5_dawn\results")
out_dir.mkdir(parents=True, exist_ok=True)

density = pd.read_csv(ROOT / "vehicle_density.csv")
haze = pd.read_csv(ROOT / "haze_dcp.csv")

df = density.merge(haze, on=["split","image"], how="inner")

# tidy columns and save
cols = ["split","image","vehicle_count","haze_score_dcp"]
df = df[cols].sort_values(["split","image"])
dst = out_dir / "uc5_features.csv"
df.to_csv(dst, index=False)
print("Wrote:", dst, "rows:", len(df))
