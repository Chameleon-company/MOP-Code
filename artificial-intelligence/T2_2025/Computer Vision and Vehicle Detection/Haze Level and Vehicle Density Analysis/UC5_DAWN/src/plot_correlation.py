# src/plot_correlation.py
"""
Plot correlation between Vehicle Density and Haze Score, color-coded by Condition.

- Reads merged features CSV with weather/condition labels
- Creates a scatter plot
- Saves the plot as PNG into results/
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------- Paths ----------
RESULTS_DIR = "artificial-intelligence/AI_IOT_Team/UC5_DAWN/results"
CSV_PATH = os.path.join(RESULTS_DIR, "uc5_features_with_conditions.csv")
PLOT_PATH = os.path.join(RESULTS_DIR, "haze_vs_density.png")

# ---------- Load Data ----------
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find {CSV_PATH}. Make sure you ran merge_features.py before this.")

# ---------- Validate Required Columns ----------
required_cols = {"VehicleDensity", "HazeScore", "Condition"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain columns {required_cols}, but got {list(df.columns)}")

# ---------- Plot ----------
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    df["VehicleDensity"],
    df["HazeScore"],
    c=df["Condition"].astype("category").cat.codes,
    cmap="tab10",
    alpha=0.7,
    edgecolor="k",
    s=60
)

plt.xlabel("Vehicle Density", fontsize=12)
plt.ylabel("Haze Score", fontsize=12)
plt.title("Correlation between Vehicle Density and Haze (by Condition)", fontsize=14)

# Add colorbar with condition labels
cbar = plt.colorbar(scatter)
cbar.set_ticks(range(len(df["Condition"].astype("category").cat.categories)))
cbar.set_ticklabels(df["Condition"].astype("category").cat.categories)
cbar.set_label("Condition")

plt.grid(alpha=0.3)

# ---------- Save ----------
os.makedirs(RESULTS_DIR, exist_ok=True)
plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ Correlation plot saved to {PLOT_PATH}")
# src/plot_correlation.py
"""
Plot correlation between Vehicle Density and Haze Score, color-coded by Condition.

- Reads merged features CSV with weather/condition labels
- Creates a scatter plot
- Saves the plot as PNG into results/
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------- Paths ----------
RESULTS_DIR = "artificial-intelligence/AI_IOT_Team/UC5_DAWN/results"
CSV_PATH = os.path.join(RESULTS_DIR, "uc5_features_with_conditions.csv")
PLOT_PATH = os.path.join(RESULTS_DIR, "haze_vs_density.png")

# ---------- Load Data ----------
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find {CSV_PATH}. Make sure you ran merge_features.py before this.")

# ---------- Validate Required Columns ----------
required_cols = {"VehicleDensity", "HazeScore", "Condition"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain columns {required_cols}, but got {list(df.columns)}")

# ---------- Plot ----------
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    df["VehicleDensity"],
    df["HazeScore"],
    c=df["Condition"].astype("category").cat.codes,
    cmap="tab10",
    alpha=0.7,
    edgecolor="k",
    s=60
)

plt.xlabel("Vehicle Density", fontsize=12)
plt.ylabel("Haze Score", fontsize=12)
plt.title("Correlation between Vehicle Density and Haze (by Condition)", fontsize=14)

# Add colorbar with condition labels
cbar = plt.colorbar(scatter)
cbar.set_ticks(range(len(df["Condition"].astype("category").cat.categories)))
cbar.set_ticklabels(df["Condition"].astype("category").cat.categories)
cbar.set_label("Condition")

plt.grid(alpha=0.3)

# ---------- Save ----------
os.makedirs(RESULTS_DIR, exist_ok=True)
plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ Correlation plot saved to {PLOT_PATH}")
