import matplotlib.pyplot as plt
import networkx as nx
import torch
import matplotlib.patches as mpatches

from graph import G
from prepare_data import X, y, edge_index
from model import GCN



# Load model
model = GCN(X.shape[1], 16, 2)
model.load_state_dict(torch.load("model.pth"))
model.eval()

out = model(X, edge_index)
pred = out.argmax(dim=1)
probs = torch.softmax(out, dim=1)

nodes_list = list(G.nodes())
node_map = {node: i for i, node in enumerate(nodes_list)}



# REAL GEO POSITION
pos = {}

for node in G.nodes():
    idx = node_map[node]
    lat = X[idx][2].item()
    lon = X[idx][3].item()
    pos[node] = (lon, lat)



# COLOR LOGIC
node_colors = []
edge_colors = []

for node in G.nodes():
    idx = node_map[node]

    pred_val = pred[idx].item()
    actual_val = y[idx].item()

    # Fill color = ACTUAL VALUE
    if actual_val == 1:
        node_colors.append("red")     # Occupied
    else:
        node_colors.append("green")   # Free

    # Border = CORRECT / WRONG
    if pred_val == actual_val:
        edge_colors.append("black")   # correct
    else:
        edge_colors.append("yellow")  # wrong


# FIGURE SETUP
fig, ax = plt.subplots(figsize=(14, 10))



# DRAW FUNCTION
def draw_graph(selected_node=None):

    ax.clear()

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)

    # Node sizes
    sizes = []
    for node in G.nodes():
        if node == selected_node:
            sizes.append(650)
        else:
            sizes.append(260)

    # Copy colors
    node_colors_updated = node_colors.copy()
    edge_colors_updated = edge_colors.copy()

    # Highlight selected node
    if selected_node is not None:
        node_index = nodes_list.index(selected_node)
        node_colors_updated[node_index] = "blue"
        edge_colors_updated[node_index] = "black"

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors_updated,
        edgecolors=edge_colors_updated,
        linewidths=2,
        node_size=sizes,
        ax=ax
    )

    # Label selected node
    if selected_node is not None:
        nx.draw_networkx_labels(
            G,
            pos,
            labels={selected_node: str(selected_node)},
            font_size=9,
            font_color="black",
            ax=ax
        )


    # TITLE
    ax.set_title(
        "Urban Parking Prediction (Melbourne)\n"
        "Given: Time (Hour, Day) + Location + Neighbors → Predict: Next Time-Step Occupancy\n"
        "Click on a node to inspect prediction",
        fontsize=13
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)


    # LEGEND
    red_patch = mpatches.Patch(color='red', label='Actual Occupied')
    green_patch = mpatches.Patch(color='green', label='Actual Free')
    yellow_patch = mpatches.Patch(edgecolor='yellow', facecolor='white', label='Incorrect Prediction', linewidth=2)
    blue_patch = mpatches.Patch(color='blue', label='Selected Node')

    ax.legend(
        handles=[red_patch, green_patch, yellow_patch, blue_patch],
        loc='upper right'
    )


    # INFO PANEL
    if selected_node is not None:

        idx = node_map[selected_node]

        pred_val = pred[idx].item()
        actual_val = y[idx].item()
        confidence = probs[idx][pred_val].item()

        correct = pred_val == actual_val

        # Color-coded status
        status_text = "Correct" if correct else "Incorrect"
        status_color = "green" if correct else "red"

        info_text = (
            f"Selected Bay ID: {selected_node}\n\n"
            f"Prediction: {'Occupied' if pred_val==1 else 'Free'}\n"
            f"Actual: {'Occupied' if actual_val==1 else 'Free'}\n"
            f"Confidence: {confidence:.2f}\n"
        )

        ax.text(
            1.02, 0.6,
            info_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='center'
        )

        ax.text(
            1.02, 0.48,
            f"Status: {status_text}",
            transform=ax.transAxes,
            fontsize=12,
            color=status_color,
            weight='bold'
        )

        ax.text(
            1.02, 0.35,
            f"Latitude: {X[idx][2].item():.5f}\n"
            f"Longitude: {X[idx][3].item():.5f}",
            transform=ax.transAxes,
            fontsize=11
        )

    else:
        ax.text(
            1.02, 0.5,
            "Click on a node to see details",
            transform=ax.transAxes,
            fontsize=11
        )

    plt.draw()



# CLICK HANDLER
def on_click(event):

    if event.inaxes != ax:
        return

    x_click = event.xdata
    y_click = event.ydata

    closest_node = None
    min_dist = float('inf')

    for node, (x, y_pos) in pos.items():
        dist = (x - x_click)**2 + (y_pos - y_click)**2

        if dist < min_dist:
            min_dist = dist
            closest_node = node

    draw_graph(closest_node)



# INITIAL DRAW
draw_graph()

fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()