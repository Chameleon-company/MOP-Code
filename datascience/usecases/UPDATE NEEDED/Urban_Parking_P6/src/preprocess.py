
#%% STEP 1: Import Libraries
#Import Libraries

import pandas as pd

print("Step 1: Libraries imported")


# %%STEP 2: Load dataset
# Load Dataset

file_path = "../data/melbourne_parking.csv"
df = pd.read_csv(file_path)

print("Dataset loaded:", df.shape)

# Show sample timestamps
sample = pd.read_csv(file_path, nrows=5)
print("\nSample timestamps:")
print(sample[['Status_Timestamp']])


# %%STEP 3: Standardize and rename columns
#Rename Columns

df.columns = df.columns.str.strip()

df = df.rename(columns={
    'KerbsideID': 'bay_id',
    'Status_Timestamp': 'timestamp',
    'Status_Description': 'occupancy',
    'Location': 'location'
})

print("\nColumns after rename:")
print(df.columns.tolist())


# %%STEP 4: Keep only needed columns
#Keep only needed columns

df = df[['bay_id', 'timestamp', 'occupancy', 'location']]


# %%STEP 5: Convert timestamp
#Convert timestamp

df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')


# %%STEP 6: Clean occupancy values
#Clean occupancy values

df['occupancy'] = df['occupancy'].astype(str).str.strip().str.lower()

df['occupancy'] = df['occupancy'].map({
    'present': 1,
    'unoccupied': 0,
    'occupied': 1,
    'vacant': 0
})


# %%STEP 7: Extract latitude and longitude
#Distinguish latitude and longitude

df['location'] = df['location'].astype(str).str.replace('[()]', '', regex=True)

lat_long = df['location'].str.split(',', expand=True)

df['latitude'] = pd.to_numeric(lat_long[0].str.strip(), errors='coerce')
df['longitude'] = pd.to_numeric(lat_long[1].str.strip(), errors='coerce')


# %%STEP 8: Drop rows with missing required values
#Drop rows with missing required values

df = df.dropna(subset=['bay_id', 'timestamp', 'latitude', 'longitude', 'occupancy'])


# %%STEP 9: Convert data types
#Conversion of data types

df['bay_id'] = df['bay_id'].astype(int)
df['occupancy'] = df['occupancy'].astype(int)


# %%STEP 10: Final column order
#Final Order

df = df[['bay_id', 'timestamp', 'latitude', 'longitude', 'occupancy']]

print("\nFinal dataset shape:", df.shape)
print("\nFirst 5 rows of cleaned data:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())


# %%STEP 11: Save cleaned dataset
#Saving the Cleaned Dataset

output_path = "../data/final_cleaned_parking.csv"
df.to_csv(output_path, index=False)

print("\n✅ Cleaning complete. File saved at:", output_path)

