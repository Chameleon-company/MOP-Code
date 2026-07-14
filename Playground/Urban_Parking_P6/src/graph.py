import pandas as pd
import networkx as nx
from math import radians, sin, cos, sqrt, atan2
import os


# Load dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "data", "cleaned_parking_with_area.csv")

print("Loading dataset...")
df = pd.read_csv(file_path)
print(f"Dataset loaded: {df.shape[0]} rows")

# Keep only one record per parking bay
bays = df[["bay_id", "latitude", "longitude"]].drop_duplicates()
# Limit graph size for faster processing and clearer visualization
bays = bays.head(500)

print(f"Unique parking bays: {len(bays)}")

# Optional: limit to first 500 bays for faster graph creation
# Uncomment this if graph creation is too slow
# bays = bays.head(500)
# print(f"Using first {len(bays)} bays for graph construction")



# Create graph
print("Creating graph...")
G = nx.Graph()

# Add nodes
for bay in bays["bay_id"]:
    G.add_node(bay)

print(f"Added {G.number_of_nodes()} nodes")



# Haversine distance function
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1))
        * cos(radians(lat2))
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c



# Create edges between nearby bays
threshold = 0.5  # km = 500 meters

print("Creating edges...")

for i in range(len(bays)):
    # Progress indicator
    if i % 50 == 0:
        print(f"Processing node {i}/{len(bays)}")

    lat1 = bays.iloc[i]["latitude"]
    lon1 = bays.iloc[i]["longitude"]
    bay1 = bays.iloc[i]["bay_id"]

    for j in range(i + 1, len(bays)):
        lat2 = bays.iloc[j]["latitude"]
        lon2 = bays.iloc[j]["longitude"]
        bay2 = bays.iloc[j]["bay_id"]

        dist = calculate_distance(lat1, lon1, lat2, lon2)

        if dist < threshold:
            G.add_edge(bay1, bay2)

print("Graph created!")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())



# Add occupancy attribute to nodes
print("Adding occupancy attributes...")

latest_data = df.sort_values("timestamp").groupby("bay_id").last()

for bay in G.nodes():
    if bay in latest_data.index:
        G.nodes[bay]["occupancy"] = latest_data.loc[bay]["occupancy"]

print("Graph setup complete.")