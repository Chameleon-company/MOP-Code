from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RES = Path(r"C:\Users\ramad\Documents\MOP-Code\datascience\air-quality\uc5_dawn\results")
df = pd.read_csv(RES / "uc5_features.csv")

# Summary by split
summary = df.groupby("split").agg(
    images=("image","count"),
    mean_vehicles=("vehicle_count","mean"),
    mean_haze=("haze_score_dcp","mean"),
    mean_lap=("lap_var","mean"),
    mean_clahe=("clahe_contrast","mean")
).round(3)
print(summary)

# Correlations
corrs = df[["vehicle_count","haze_score_dcp","lap_var","clahe_contrast"]].corr().round(3)
print("\nCorrelations:\n", corrs)

# 1) Scatter: haze vs vehicles
plt.figure(figsize=(7,5))
plt.scatter(df["haze_score_dcp"], df["vehicle_count"], s=6)
plt.xlabel("Haze score (DCP ↑ = hazier)")
plt.ylabel("Vehicle count")
plt.title("UC5: Vehicle density vs haze (DAWN)")
plt.tight_layout()
plt.savefig(RES / "haze_vs_density.png", dpi=160)

# 2) Histograms
for col in ["vehicle_count","haze_score_dcp","lap_var","clahe_contrast"]:
    plt.figure(figsize=(6,4))
    df[col].hist(bins=40)
    plt.title(f"Distribution: {col}")
    plt.xlabel(col); plt.ylabel("Freq")
    plt.tight_layout()
    plt.savefig(RES / f"hist_{col}.png", dpi=160)

# 3) Pairwise quick look (vehicles vs new features)
for col in ["lap_var","clahe_contrast"]:
    plt.figure(figsize=(6,4))
    plt.scatter(df[col], df["vehicle_count"], s=5)
    plt.xlabel(col); plt.ylabel("Vehicle count")
    plt.title(f"{col} vs vehicles")
    plt.tight_layout()
    plt.savefig(RES / f"{col}_vs_vehicles.png", dpi=160)

print("Saved plots in:", RES)
