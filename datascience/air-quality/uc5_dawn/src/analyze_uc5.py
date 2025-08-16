from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RES = Path(r"C:\Users\ramad\Documents\MOP-Code\datascience\air-quality\uc5_dawn\results")
df = pd.read_csv(RES / "uc5_features.csv")

summary = (
    df.groupby("split")
      .agg(images=("image","count"),
           mean_vehicles=("vehicle_count","mean"),
           mean_haze=("haze_score_dcp","mean"))
      .reset_index()
)
print(summary)

# overall correlation (simple baseline)
corr = df["vehicle_count"].corr(df["haze_score_dcp"])
print(f"Pearson corr (vehicles vs haze): {corr:.3f}")

# scatter for the report
plt.figure(figsize=(7,5))
plt.scatter(df["haze_score_dcp"], df["vehicle_count"], s=6)
plt.xlabel("Haze score (Dark Channel Prior ↑ = hazier)")
plt.ylabel("Vehicle count")
plt.title("UC5: Vehicle density vs. haze (DAWN validation set)")
plt.tight_layout()
png = RES / "haze_vs_density.png"
plt.savefig(png, dpi=160)
print("Saved plot:", png)

# save a small markdown summary
md = RES / "SUMMARY.md"
md.write_text(
    f"# UC5 quick results\n\n"
    f"- Rows: {len(df)}\n"
    f"- Per-split summary:\n\n{summary.to_markdown(index=False)}\n\n"
    f"- Pearson corr (vehicles vs haze): {corr:.3f}\n"
)
print("Wrote:", md)
