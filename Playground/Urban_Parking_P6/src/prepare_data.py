import pandas as pd
from graph import G
from features import map_nodes, prepare_features, create_edge_index

# Load cleaned dataset
df = pd.read_csv("../data/cleaned_parking.csv")

# Create node mapping
node_map = map_nodes(G)

# Prepare features and labels
X, y = prepare_features(df, node_map)

# Prepare edge index
edge_index = create_edge_index(G, node_map)

print("Features shape:", X.shape)
print("Labels shape:", y.shape)
print("Edges shape:", edge_index.shape)

print("\nSample features:")
print(X[:5])

print("\nSample labels:")
print(y[:5])

print("\nSample edge_index:")
print(edge_index[:, :10])