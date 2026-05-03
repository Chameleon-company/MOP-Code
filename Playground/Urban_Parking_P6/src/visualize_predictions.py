import matplotlib.pyplot as plt
import networkx as nx
import torch
import matplotlib.patches as mpatches

from graph import G
from prepare_data import X, y, edge_index
from model import GCN


# -----------------------------
# 1. Load trained model
# -----------------------------
model = GCN(X.shape[1], 16, 2)
model.load_state_dict(torch.load("model.pth"))
model.eval()

out = model(X, edge_index)
pred = out.argmax(dim=1)


# -----------------------------
# 2. Select target node
# -----------------------------
nodes_list = list(G.nodes())
target_node = nodes_list[0]
node_map = {node: i for i, node in enumerate(nodes_list)}

target_idx = node_map[target_node]


# -----------------------------
# 3. Get subgraph
# -----------------------------
neighbors = list(G.neighbors(target_node))
sub_nodes = [target_node] + neighbors
subG = G.subgraph(sub_nodes)


# -----------------------------
# 4. Color + Label logic
# -----------------------------
colors = []
labels = {}

for node in subG.nodes():
    idx = node_map[node]

    pred_val = pred[idx].item()
    actual_val = y[idx].item()

    # Label shows both
    labels[node] = f"{node}\nP:{pred_val} A:{actual_val}"

    if node == target_node:
        colors.append("blue")
    else:
        if pred_val == 1:
            colors.append("red")
        else:
            colors.append("green")


# -----------------------------
# 5. Draw graph
# -----------------------------
pos = nx.spring_layout(subG, seed=42)

plt.figure(figsize=(8, 6))

nx.draw(
    subG,
    pos,
    labels=labels,
    node_color=colors,
    node_size=900,
    font_size=8
)

# -----------------------------
# 6. Legend
# -----------------------------
blue_patch = mpatches.Patch(color='blue', label='Target Node')
red_patch = mpatches.Patch(color='red', label='Predicted Occupied')
green_patch = mpatches.Patch(color='green', label='Predicted Free')

plt.legend(handles=[blue_patch, red_patch, green_patch])


# -----------------------------
# 7. INSIGHTS (NEW 🔥)
# -----------------------------
occupied_neighbors = sum(pred[node_map[n]].item() == 1 for n in neighbors)
free_neighbors = sum(pred[node_map[n]].item() == 0 for n in neighbors)

print("\n🔍 INSIGHTS:")
print(f"Target Node: {target_node}")
print(f"Predicted: {'Occupied' if pred[target_idx]==1 else 'Free'}")
print(f"Actual: {'Occupied' if y[target_idx]==1 else 'Free'}")
print(f"Occupied neighbors: {occupied_neighbors}")
print(f"Free neighbors: {free_neighbors}")

if occupied_neighbors > free_neighbors:
    print("👉 Insight: Surrounding bays are mostly occupied → higher chance target is occupied")
else:
    print("👉 Insight: Surrounding bays are mostly free → higher chance target is free")


# -----------------------------
plt.title("GNN Parking Prediction (Predicted vs Actual)")
plt.show()