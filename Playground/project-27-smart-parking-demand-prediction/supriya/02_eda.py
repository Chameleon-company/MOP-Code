# %%
import os
import pandas as pd
import matplotlib.pyplot as plt

# %%
# paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)

processed_dir = os.path.join(project_dir, "data", "processed")
output_dir = os.path.join(current_dir, "outputs")
figures_dir = os.path.join(output_dir, "figures")
tables_dir = os.path.join(output_dir, "tables")

os.makedirs(figures_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

merged_file = os.path.join(processed_dir, "merged_bay_sensor_data.csv")
demand_file = os.path.join(processed_dir, "cleaned_demand_data.csv")

print("Processed folder:", processed_dir)

# %%
# load cleaned data
merged_df = pd.read_csv(merged_file)
demand_df = pd.read_csv(demand_file)

print("Files loaded")
print("merged_df shape:", merged_df.shape)
print("demand_df shape:", demand_df.shape)

# %%
# preview
merged_df.head()

# %%
demand_df.head()

# %%
# basic checking
print("Missing values in merged_df")
print(merged_df.isnull().sum())

print("\nMissing values in demand_df")
print(demand_df.isnull().sum())

# %%
# average demand by hour
# this helps identify the busiest parking times in a day
hourly_demand = demand_df.groupby("status_hour", as_index=False)["average_occupancy"].mean()
hourly_demand

# %%
plt.figure(figsize=(10, 5))
plt.plot(hourly_demand["status_hour"], hourly_demand["average_occupancy"], marker="o")
plt.title("Average Parking Demand by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Occupancy")
plt.xticks(range(0, 24))
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# save chart
plt.figure(figsize=(10, 5))
plt.plot(hourly_demand["status_hour"], hourly_demand["average_occupancy"], marker="o")
plt.title("Average Parking Demand by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Occupancy")
plt.xticks(range(0, 24))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "01_average_demand_by_hour.png"))
plt.show()

# %%
# demand by day of the week
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

daily_demand = demand_df.groupby("status_day", as_index=False)["average_occupancy"].mean()
daily_demand["status_day"] = pd.Categorical(daily_demand["status_day"], categories=day_order, ordered=True)
daily_demand = daily_demand.sort_values("status_day")

daily_demand

# %%
plt.figure(figsize=(10, 5))
plt.bar(daily_demand["status_day"], daily_demand["average_occupancy"])
plt.title("Average Parking Demand by Day of Week")
plt.xlabel("Day")
plt.ylabel("Average Occupancy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# distribution of low, medium and high demand
demand_counts = demand_df["demand_level"].value_counts()
demand_counts

# %%
plt.figure(figsize=(8, 5))
plt.bar(demand_counts.index, demand_counts.values)
plt.title("Distribution of Demand Levels")
plt.xlabel("Demand Level")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# %%
# busiest zones
zone_summary = (
    merged_df.groupby("zone_number")
    .agg(
        average_occupancy=("occupied", "mean"),
        observation_count=("occupied", "count")
    )
    .reset_index()
    .sort_values("average_occupancy", ascending=False)
)

zone_summary.head(10)

# %%
top_zones = zone_summary.head(10)

plt.figure(figsize=(12, 5))
plt.bar(top_zones["zone_number"].astype(str), top_zones["average_occupancy"])
plt.title("Top 10 Parking Zones by Average Occupancy")
plt.xlabel("Zone Number")
plt.ylabel("Average Occupancy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# least used zones
underutilized_zones = zone_summary.sort_values("average_occupancy", ascending=True).head(10)
underutilized_zones

# %%
plt.figure(figsize=(12, 5))
plt.bar(underutilized_zones["zone_number"].astype(str), underutilized_zones["average_occupancy"])
plt.title("Top 10 Underutilized Parking Zones")
plt.xlabel("Zone Number")
plt.ylabel("Average Occupancy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# occupancy by restriction type
if "restriction_display" in merged_df.columns:
    restriction_demand = (
        merged_df.groupby("restriction_display", as_index=False)["occupied"]
        .mean()
        .sort_values("occupied", ascending=False)
    )
    restriction_demand.head(10)

# %%
if "restriction_display" in merged_df.columns:
    top_restrictions = restriction_demand.head(10)

    plt.figure(figsize=(12, 5))
    plt.bar(top_restrictions["restriction_display"], top_restrictions["occupied"])
    plt.title("Average Occupancy by Restriction Type")
    plt.xlabel("Restriction Display")
    plt.ylabel("Average Occupancy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# %%
# weekday vs weekend
weekend_comparison = merged_df.groupby("is_weekend", as_index=False)["occupied"].mean()
weekend_comparison

# %%
plt.figure(figsize=(6, 4))
plt.bar(weekend_comparison["is_weekend"].astype(str), weekend_comparison["occupied"])
plt.title("Weekday vs Weekend Occupancy")
plt.xlabel("Is Weekend")
plt.ylabel("Average Occupancy")
plt.tight_layout()
plt.show()

# %%
# hourly demand split by weekday/weekend
hour_weekend_demand = (
    demand_df.groupby(["status_hour", "is_weekend"], as_index=False)["average_occupancy"].mean()
)
hour_weekend_demand.head(20)

# %%
plt.figure(figsize=(10, 5))
for weekend_value in sorted(hour_weekend_demand["is_weekend"].unique()):
    subset = hour_weekend_demand[hour_weekend_demand["is_weekend"] == weekend_value]
    label = "Weekend" if weekend_value else "Weekday"
    plt.plot(subset["status_hour"], subset["average_occupancy"], marker="o", label=label)

plt.title("Hourly Demand: Weekday vs Weekend")
plt.xlabel("Hour of Day")
plt.ylabel("Average Occupancy")
plt.xticks(range(0, 24))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# busiest day-hour combinations
peak_periods = (
    demand_df.groupby(["status_day", "status_hour"], as_index=False)["average_occupancy"]
    .mean()
    .sort_values("average_occupancy", ascending=False)
    .head(10)
)

peak_periods

# %%
# quick statistics
demand_df.describe(include="all")

# %%
# simple correlation check
numeric_cols = [col for col in ["zone_number", "status_hour", "is_weekend", "average_occupancy"] if col in demand_df.columns]
demand_df[numeric_cols].corr(numeric_only=True)

# %%
# save useful tables
hourly_demand.to_csv(os.path.join(tables_dir, "hourly_demand.csv"), index=False)
daily_demand.to_csv(os.path.join(tables_dir, "daily_demand.csv"), index=False)
zone_summary.to_csv(os.path.join(tables_dir, "zone_summary.csv"), index=False)
peak_periods.to_csv(os.path.join(tables_dir, "peak_periods.csv"), index=False)

print("EDA tables saved")

# %%
# save a few final charts
plt.figure(figsize=(10, 5))
plt.plot(hourly_demand["status_hour"], hourly_demand["average_occupancy"], marker="o")
plt.title("Average Parking Demand by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Occupancy")
plt.xticks(range(0, 24))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "01_average_demand_by_hour.png"))
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(daily_demand["status_day"], daily_demand["average_occupancy"])
plt.title("Average Parking Demand by Day of Week")
plt.xlabel("Day")
plt.ylabel("Average Occupancy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "02_average_demand_by_day.png"))
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(demand_counts.index, demand_counts.values)
plt.title("Distribution of Demand Levels")
plt.xlabel("Demand Level")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "03_demand_level_distribution.png"))
plt.show()

print("Charts saved in:", figures_dir)