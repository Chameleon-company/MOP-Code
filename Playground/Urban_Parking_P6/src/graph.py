import pandas as pd
import networkx as nx
from math import radians, sin, cos, sqrt, atan2

# Loading the dataset
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "data", "cleaned_parking.csv")

df = pd.read_csv(file_path)

df = df.sample(n=50)

# Create graph
G = nx.Graph()

# Add nodes (unique parking bays)
for bay in df['bay_id'].unique():
    G.add_node(bay)

# Function to calculate distance (Haversine)
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius (km)

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

# Create edges (connect nearby bays)
threshold = 50  # km (100 meters)

bays = df[['bay_id', 'latitude', 'longitude']].drop_duplicates()

for i in range(len(bays)):
    for j in range(i+1, len(bays)):
        lat1, lon1 = bays.iloc[i][['latitude', 'longitude']]
        lat2, lon2 = bays.iloc[j][['latitude', 'longitude']]

        dist = calculate_distance(lat1, lon1, lat2, lon2)

        if dist < threshold:
            G.add_edge(bays.iloc[i]['bay_id'], bays.iloc[j]['bay_id'])

print("Graph created!")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# Add node features (latest record per bay)
latest_data = df.sort_values('timestamp').groupby('bay_id').last()

for bay in G.nodes():
    if bay in latest_data.index:
        G.nodes[bay]['occupancy'] = latest_data.loc[bay]['occupancy']