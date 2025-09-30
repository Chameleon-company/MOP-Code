from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RES = Path("artificial-intelligence/AI_IOT_Team/UC5_DAWN/results")
df = pd.read_csv(RES/"uc5_features.csv")

print("Summary stats:\n", df.describe())

plt.scatter(df["haze_score_dcp"], df["vehicle_count"], s=6)
plt.xlabel("Haze score (DCP)")
plt.ylabel("Vehicle count")
plt.title("UC5: Haze vs Vehicle density")
plt.savefig(RES/"haze_vs_density.png", dpi=160)
print("Plot saved:", RES/"haze_vs_density.png")
