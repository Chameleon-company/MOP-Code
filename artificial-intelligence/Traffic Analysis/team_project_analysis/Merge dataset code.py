import pandas as pd

# Load datasets
data_traffic_count = pd.read_csv('traffic_count_cleaned_data.csv')
data_road_corridors = pd.read_csv('CleanData-road-corridors.csv')

# Merge datasets on the road segment ID
merged_data = pd.merge(data_traffic_count, data_road_corridors, left_on="road_segment", right_on="seg_id")

# Save merged data to a new CSV file
merged_data.to_csv('merged_traffic_data.csv', index=False)

# Print the first few rows of the merged data to verify
print(merged_data.head())