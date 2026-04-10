# %%
import os
import pandas as pd

# %%
# setting up paths
# this data cleaning code is inside the supriya folder, so we move one level up to the main project folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)

data_dir = os.path.join(project_dir, "data")
raw_dir = os.path.join(data_dir, "raw")
processed_dir = os.path.join(data_dir, "processed")

os.makedirs(processed_dir, exist_ok=True)

print("Project folder:", project_dir)
print("Raw data folder:", raw_dir)
print("Processed data folder:", processed_dir)

# %%
# file paths
sensor_file = os.path.join(raw_dir, "On-street_Car_Parking_Sensor_Data_-_2014.csv")
bay_sensor_file = os.path.join(raw_dir, "on-street-parking-bay-sensors.csv")
arrivals_file = os.path.join(raw_dir, "Parking_bay_arrivals_and_departures_2014.csv")
sign_plate_file = os.path.join(raw_dir, "sign-plates-located-in-each-parking-zone.csv")

print(sensor_file)
print(bay_sensor_file)
print(arrivals_file)
print(sign_plate_file)

# %%
# loading datasets
# I am limiting rows for the larger files so the notebook runs faster
sensor_df = pd.read_csv(sensor_file, nrows=300000)
bay_sensor_df = pd.read_csv(bay_sensor_file)
arrivals_df = pd.read_csv(arrivals_file, nrows=300000)
sign_df = pd.read_csv(sign_plate_file)

print("Datasets loaded successfully")

# %%
# quick preview of each dataset
print("sensor_df shape:", sensor_df.shape)
sensor_df.head()

# %%
print("bay_sensor_df shape:", bay_sensor_df.shape)
bay_sensor_df.head()

# %%
print("arrivals_df shape:", arrivals_df.shape)
arrivals_df.head()

# %%
print("sign_df shape:", sign_df.shape)
sign_df.head()

# %%
# checking columns
print(sensor_df.columns)
print(bay_sensor_df.columns)
print(arrivals_df.columns)
print(sign_df.columns)

# %%
# checking missing values before cleaning
print("Missing values in sensor_df")
print(sensor_df.isnull().sum())

print("\nMissing values in bay_sensor_df")
print(bay_sensor_df.isnull().sum())

print("\nMissing values in arrivals_df")
print(arrivals_df.isnull().sum())

print("\nMissing values in sign_df")
print(sign_df.isnull().sum())

# %%
# making column names easier to work with
def clean_columns(df):
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df

sensor_df = clean_columns(sensor_df)
bay_sensor_df = clean_columns(bay_sensor_df)
arrivals_df = clean_columns(arrivals_df)
sign_df = clean_columns(sign_df)

print("Column names cleaned")
print(sensor_df.columns)

# %%
# removing duplicate rows
print("Before removing duplicates:")
print("sensor_df:", sensor_df.shape)
print("bay_sensor_df:", bay_sensor_df.shape)
print("arrivals_df:", arrivals_df.shape)
print("sign_df:", sign_df.shape)

sensor_df = sensor_df.drop_duplicates()
bay_sensor_df = bay_sensor_df.drop_duplicates()
arrivals_df = arrivals_df.drop_duplicates()
sign_df = sign_df.drop_duplicates()

print("\nAfter removing duplicates:")
print("sensor_df:", sensor_df.shape)
print("bay_sensor_df:", bay_sensor_df.shape)
print("arrivals_df:", arrivals_df.shape)
print("sign_df:", sign_df.shape)

# %%
# converting number-like columns
for col in ["deviceid", "streetid", "durationseconds"]:
    if col in sensor_df.columns:
        sensor_df[col] = sensor_df[col].astype(str).str.replace(",", "", regex=False)
        sensor_df[col] = pd.to_numeric(sensor_df[col], errors="coerce")

for col in ["parkingeventid", "deviceid", "durationseconds", "bayid"]:
    if col in arrivals_df.columns:
        arrivals_df[col] = arrivals_df[col].astype(str).str.replace(",", "", regex=False)
        arrivals_df[col] = pd.to_numeric(arrivals_df[col], errors="coerce")

print("Numeric columns converted")

# %%
# fixing date and time columns
sensor_df["arrivaltime"] = pd.to_datetime(sensor_df["arrivaltime"], errors="coerce")
sensor_df["departuretime"] = pd.to_datetime(sensor_df["departuretime"], errors="coerce")

bay_sensor_df["lastupdated"] = pd.to_datetime(bay_sensor_df["lastupdated"], errors="coerce", utc=True)
bay_sensor_df["status_timestamp"] = pd.to_datetime(bay_sensor_df["status_timestamp"], errors="coerce", utc=True)

arrivals_df["arrivaltime"] = pd.to_datetime(arrivals_df["arrivaltime"], errors="coerce")
arrivals_df["departuretime"] = pd.to_datetime(arrivals_df["departuretime"], errors="coerce")

print("Datetime conversion finished")

# %%
# checking data types after conversion
sensor_df.info()

# %%
bay_sensor_df.info()

# %%
arrivals_df.info()

# %%
# dropping rows where important values are missing
sensor_df = sensor_df.dropna(subset=["deviceid", "arrivaltime", "departuretime"])
bay_sensor_df = bay_sensor_df.dropna(subset=["status_timestamp", "status_description", "kerbsideid"])
arrivals_df = arrivals_df.dropna(subset=["deviceid", "arrivaltime", "departuretime", "bayid"])
sign_df = sign_df.dropna(subset=["parkingzone"])

print("Important missing rows removed")
print(sensor_df.shape)
print(bay_sensor_df.shape)
print(arrivals_df.shape)
print(sign_df.shape)

# %%
# creating time-based features
sensor_df["arrival_hour"] = sensor_df["arrivaltime"].dt.hour
sensor_df["arrival_day"] = sensor_df["arrivaltime"].dt.day_name()
sensor_df["arrival_month"] = sensor_df["arrivaltime"].dt.month
sensor_df["is_weekend"] = sensor_df["arrivaltime"].dt.dayofweek >= 5

bay_sensor_df["status_hour"] = bay_sensor_df["status_timestamp"].dt.hour
bay_sensor_df["status_day"] = bay_sensor_df["status_timestamp"].dt.day_name()
bay_sensor_df["status_month"] = bay_sensor_df["status_timestamp"].dt.month
bay_sensor_df["is_weekend"] = bay_sensor_df["status_timestamp"].dt.dayofweek >= 5

arrivals_df["arrival_hour"] = arrivals_df["arrivaltime"].dt.hour
arrivals_df["arrival_day"] = arrivals_df["arrivaltime"].dt.day_name()
arrivals_df["arrival_month"] = arrivals_df["arrivaltime"].dt.month
arrivals_df["is_weekend"] = arrivals_df["arrivaltime"].dt.dayofweek >= 5

print("Time features added")

# %%
# removing timezone from the live bay sensor timestamps
bay_sensor_df["lastupdated"] = bay_sensor_df["lastupdated"].dt.tz_convert(None)
bay_sensor_df["status_timestamp"] = bay_sensor_df["status_timestamp"].dt.tz_convert(None)

bay_sensor_df[["lastupdated", "status_timestamp"]].head()

# %%
# converting parking status into a simple occupied flag
# Present = occupied, anything else = not occupied
bay_sensor_df["occupied"] = bay_sensor_df["status_description"].apply(
    lambda x: 1 if x == "Present" else 0
)

print(bay_sensor_df["occupied"].value_counts())

# %%
# renaming parkingzone so it matches the other table
sign_df = sign_df.rename(columns={"parkingzone": "zone_number"})
sign_df.head()

# %%
# merging bay sensor data with restriction/sign plate data
merged_df = bay_sensor_df.merge(sign_df, on="zone_number", how="left")

print("Merged dataset shape:", merged_df.shape)
merged_df.head()

# %%
# checking missing values after merge
merged_df.isnull().sum()

# %%
# filling a few useful columns after merge
merged_df = merged_df.dropna(subset=["zone_number"])

for col in ["restriction_days", "restriction_display", "time_restrictions_start", "time_restrictions_finish"]:
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].fillna("Unknown")

print("Post-merge cleaning done")

# %%
# now I create a smaller demand table
# average occupancy by zone, day, hour and weekend flag
demand_df = (
    merged_df.groupby(["zone_number", "status_day", "status_hour", "is_weekend"], as_index=False)["occupied"]
    .mean()
)

demand_df = demand_df.rename(columns={"occupied": "average_occupancy"})
demand_df.head()

# %%
# classifying demand levels into low, medium and high
def demand_level(value):
    if value < 0.33:
        return "Low"
    elif value < 0.66:
        return "Medium"
    else:
        return "High"

demand_df["demand_level"] = demand_df["average_occupancy"].apply(demand_level)

demand_df.head()

# %%
# final checking
print("Merged dataset shape:", merged_df.shape)
print("Demand dataset shape:", demand_df.shape)

print("\nMissing values in demand_df")
print(demand_df.isnull().sum())

print("\nDemand level distribution")
print(demand_df["demand_level"].value_counts())

# %%
# saving cleaned outputs for EDA
merged_output = os.path.join(processed_dir, "merged_bay_sensor_data.csv")
demand_output = os.path.join(processed_dir, "cleaned_demand_data.csv")

merged_df.to_csv(merged_output, index=False)
demand_df.to_csv(demand_output, index=False)

print("Saved files:")
print(merged_output)
print(demand_output)