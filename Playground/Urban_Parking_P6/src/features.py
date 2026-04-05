import pandas as pd
import torch


def map_nodes(G):
    """
    Map original graph node IDs to 0...N-1
    """
    nodes = list(G.nodes())
    node_map = {node: i for i, node in enumerate(nodes)}
    return node_map


def prepare_features(df, node_map):
    """
    Prepare node features X and labels y
    based on the latest record for each parking bay.
    """

    df = df.copy()

    # Convert timestamp safely
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Drop rows with missing essential values
    df = df.dropna(subset=['bay_id', 'timestamp', 'occupancy'])

    # Keep only bays that exist in the graph
    df = df[df['bay_id'].isin(node_map.keys())]

    # Extract time features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.dayofweek

    # Sort by timestamp and keep latest record per bay
    latest = df.sort_values('timestamp').groupby('bay_id').last()

    # Reindex in the exact order of graph nodes
    ordered_nodes = list(node_map.keys())
    latest = latest.reindex(ordered_nodes)

    # Fill any missing values if a node has no matching row
    latest['hour'] = latest['hour'].fillna(0)
    latest['day'] = latest['day'].fillna(0)
    latest['occupancy'] = latest['occupancy'].fillna(0)

    # Features: hour and day
    X = latest[['hour', 'day']].values

    # Labels: occupancy
    y = latest['occupancy'].values

    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)

    return X, y


def create_edge_index(G, node_map):
    """
    Convert graph edges into PyTorch Geometric style edge_index
    """
    edges = list(G.edges())
    edge_index = []

    for u, v in edges:
        if u in node_map and v in node_map:
            edge_index.append([node_map[u], node_map[v]])
            edge_index.append([node_map[v], node_map[u]])  # undirected graph

    if len(edge_index) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index