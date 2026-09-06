from pathlib import Path
import pandas as pd

RES = Path("artificial-intelligence/AI_IOT_Team/UC5_DAWN/results")
counts = pd.read_csv(RES/"vehicle_counts.csv")
haze   = pd.read_csv(RES/"haze_dcp.csv")

df = counts.merge(haze, on=["split","image"])
df.to_csv(RES/"uc5_features.csv", index=False)
print("Merged features saved:", RES/"uc5_features.csv")

# Balance check
counts = df.groupby("split")["vehicle_count"].count()
print("Images per split:\n", counts)

min_count = counts.min()
balanced = df.groupby("split").apply(lambda g: g.sample(min_count, random_state=42)).reset_index(drop=True)
balanced.to_csv(RES/"uc5_features_balanced.csv", index=False)
print("Balanced features saved:", RES/"uc5_features_balanced.csv")
