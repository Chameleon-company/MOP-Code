import pandas as pd
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv("../data/cleaned_parking_with_area.csv")


# Select Area
areas = sorted(df["area"].dropna().unique())

print("Available Areas:")
for i, area in enumerate(areas, start=1):
    print(f"{i}. {area}")

choice = int(input("\nSelect an area by number: "))

if choice < 1 or choice > len(areas):
    print("Invalid selection.")
    exit()

selected_area = areas[choice - 1]


# Filter area
area_df = df[df["area"] == selected_area]


# Plot
plt.figure(figsize=(10, 8))

plt.scatter(
    area_df["longitude"],
    area_df["latitude"],
    s=80,
    alpha=0.7
)

plt.title(f"Parking Bays in {selected_area}")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)

# Annotate bay IDs
for _, row in area_df.iterrows():
    plt.text(
        row["longitude"],
        row["latitude"],
        str(row["bay_id"]),
        fontsize=6
    )

plt.show()