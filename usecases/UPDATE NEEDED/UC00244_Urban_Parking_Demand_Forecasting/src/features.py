import pandas as pd
import torch


def map_nodes(G):
    nodes = list(G.nodes())
    node_map = {node: i for i, node in enumerate(nodes)}
    return node_map


def prepare_features(df, node_map):
    df = df.copy()


    # Basic cleaning
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['bay_id', 'timestamp', 'occupancy'])
    df = df[df['bay_id'].isin(node_map.keys())]


    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.dayofweek

    
    # Forecasting logic (t → t+1)
    df = df.sort_values(['bay_id', 'timestamp'])
    df['future_occupancy'] = df.groupby('bay_id')['occupancy'].shift(-1)

    # fill last missing with current occupancy (safe fallback)
    df['future_occupancy'] = df['future_occupancy'].fillna(df['occupancy'])

    
    # Latest snapshot per node
    latest = df.groupby('bay_id').last()

    ordered_nodes = list(node_map.keys())
    latest = latest.reindex(ordered_nodes)

    # Fill missing safely
    latest['hour'] = latest['hour'].fillna(0)
    latest['day'] = latest['day'].fillna(0)
    latest['future_occupancy'] = latest['future_occupancy'].fillna(0)
    latest['latitude'] = latest['latitude'].fillna(0)
    latest['longitude'] = latest['longitude'].fillna(0)

  
    # Feature matrix
    X = latest[['hour', 'day', 'latitude', 'longitude']].values

   
    # NORMALIZATION
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    # avoid division by zero
    std[std == 0] = 1

    X = (X - mean) / std

    
    # Labels (future occupancy)
    y = latest['future_occupancy'].values

    
    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)

    return X, y


def create_edge_index(G, node_map):
    edges = list(G.edges())
    edge_index = []

    for u, v in edges:
        if u in node_map and v in node_map:
            edge_index.append([node_map[u], node_map[v]])
            edge_index.append([node_map[v], node_map[u]])

    if len(edge_index) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index