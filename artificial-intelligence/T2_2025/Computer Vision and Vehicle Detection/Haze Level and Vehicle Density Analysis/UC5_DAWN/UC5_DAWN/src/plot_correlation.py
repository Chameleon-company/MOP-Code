import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("artificial-intelligence/AI_IOT_Team/UC5_DAWN/results/uc5_features_with_conditions.csv")

plt.scatter(df["VehicleDensity"], df["HazeScore"], c=df["Condition"].astype('category').cat.codes)
plt.xlabel("Vehicle Density")
plt.ylabel("Haze Score")
plt.title("Correlation between Vehicle Density and Haze (by Condition)")
plt.colorbar()
plt.show()
