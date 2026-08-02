import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from model import GCN
from graph import G as full_graph
from features import map_nodes, create_edge_index



# 1. LOAD DATASET
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "data", "cleaned_parking_with_area.csv")

df = pd.read_csv(file_path)



# 2. USER INPUT: AREA
areas = sorted(df["area"].dropna().unique())

print("Available Areas:")
for i, area in enumerate(areas, start=1):
    print(f"{i}. {area}")

choice = int(input("\nSelect an area by number: "))

if choice < 1 or choice > len(areas):
    print("Invalid selection.")
    exit()

selected_area = areas[choice - 1]



# 3. USER INPUT: HOUR
hour = int(input("Enter hour (0-23): "))

if hour < 0 or hour > 23:
    print("Invalid hour.")
    exit()



# 4. USER INPUT: DAY
print("\nDay Mapping:")
print("0 = Monday")
print("1 = Tuesday")
print("2 = Wednesday")
print("3 = Thursday")
print("4 = Friday")
print("5 = Saturday")
print("6 = Sunday")

day = int(input("Enter day (0-6): "))

if day < 0 or day > 6:
    print("Invalid day.")
    exit()



# 5. FILTER DATA TO SELECTED AREA
area_df = df[df["area"] == selected_area].copy()

print(f"\nSelected Area: {selected_area}")
print(f"Rows: {len(area_df)}")
print(f"Unique Parking Bays: {area_df['bay_id'].nunique()}")



# 6. CREATE SUBGRAPH
selected_bays = area_df["bay_id"].unique()
selected_bays = [bay for bay in selected_bays if bay in full_graph.nodes()]

G = full_graph.subgraph(selected_bays).copy()

print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

if G.number_of_nodes() == 0:
    print("No nodes found for this area.")
    exit()



# 7. NODE MAPPING
node_map = map_nodes(G)
edge_index = create_edge_index(G, node_map)



# 8. PREPARE LATEST DATA
area_df["timestamp"] = pd.to_datetime(area_df["timestamp"], errors="coerce")
latest = area_df.sort_values("timestamp").groupby("bay_id").last()

ordered_nodes = list(node_map.keys())
latest = latest.reindex(ordered_nodes)

# Override time with user input
latest["hour"] = hour
latest["day"] = day



# 9. CREATE FEATURES
X = latest[["hour", "day", "latitude", "longitude"]].fillna(0)

# Normalize features
X = (X - X.mean()) / (X.std() + 1e-8)

X = torch.tensor(X.values, dtype=torch.float)

# Actual occupancy used only for comparison in visualization
y = torch.tensor(
    latest["occupancy"].fillna(0).astype(int).values,
    dtype=torch.long
)



# 10. LOAD TRAINED MODEL
model = GCN(X.shape[1], 16, 2)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()



# 11. PREDICT
with torch.no_grad():
    out = model(X, edge_index)
    probs = torch.softmax(out, dim=1)

    occupied_probs = probs[:, 1]
    threshold = 0.4
    pred = (occupied_probs > threshold).long()



# 12. POSITIONS
pos = {}

for node in G.nodes():
    lat = latest.loc[node, "latitude"]
    lon = latest.loc[node, "longitude"]
    pos[node] = (lon, lat)

nodes_list = list(G.nodes())



# 13. FIGURE SETUP
fig, ax = plt.subplots(figsize=(18, 10))



# 14. DRAW FUNCTION
def draw_graph(selected_node=None):
    ax.clear()

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.15,
        edge_color="gray",
        ax=ax
    )

    node_colors = []
    border_colors = []
    sizes = []

    for node in nodes_list:
        idx = node_map[node]

        actual = y[idx].item()
        predicted = pred[idx].item()

        # Fill color based on ACTUAL occupancy
        if actual == 1:
            fill_color = "red"      # Occupied
        else:
            fill_color = "green"    # Free

        # Yellow border if prediction is incorrect
        if actual == predicted:
            border_color = "black"
        else:
            border_color = "yellow"

        # Selected node
        if node == selected_node:
            fill_color = "blue"
            size = 800
        else:
            size = 350

        node_colors.append(fill_color)
        border_colors.append(border_color)
        sizes.append(size)

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        edgecolors=border_colors,
        linewidths=2,
        node_size=sizes,
        ax=ax
    )

    # Label only selected node
    if selected_node is not None:
        nx.draw_networkx_labels(
            G,
            pos,
            labels={selected_node: str(selected_node)},
            font_size=9,
            font_color="black",
            ax=ax
        )

    # Title
    ax.set_title(
        f"Urban Parking Forecasting – {selected_area}\n"
        f"Given: Hour={hour}, Day={day} | Predicting Next Time-Step Occupancy\n"
        f"Using Spatial + Temporal Features",
        fontsize=14
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)

    # Legend
    occupied_patch = mpatches.Patch(color="red", label="Actual Occupied")
    free_patch = mpatches.Patch(color="green", label="Actual Free")
    wrong_patch = mpatches.Patch(
        facecolor="white",
        edgecolor="yellow",
        label="Incorrect Prediction"
    )
    selected_patch = mpatches.Patch(color="blue", label="Selected Node")

    ax.legend(
        handles=[occupied_patch, free_patch, wrong_patch, selected_patch],
        loc="upper right"
    )

    # Information panel
    if selected_node is not None:
        idx = node_map[selected_node]

        predicted = pred[idx].item()
        actual = y[idx].item()
        confidence = probs[idx][predicted].item()

        pred_text = "Occupied" if predicted == 1 else "Free"
        actual_text = "Occupied" if actual == 1 else "Free"

        status = "Correct" if predicted == actual else "Incorrect"
        status_color = "green" if status == "Correct" else "red"

        info_x = 1.02
        y0 = 0.72

        ax.text(
            info_x,
            y0,
            (
                f"Selected Bay ID: {selected_node}\n\n"
                f"Prediction: {pred_text}\n"
                f"Actual: {actual_text}\n"
                f"Confidence: {confidence:.2f}\n"
            ),
            transform=ax.transAxes,
            fontsize=11,
            va="top"
        )

        ax.text(
            info_x,
            y0 - 0.18,
            f"Status: {status}",
            color=status_color,
            fontsize=12,
            fontweight="bold",
            transform=ax.transAxes,
            va="top"
        )

        ax.text(
            info_x,
            y0 - 0.30,
            (
                f"\nLatitude: {latest.loc[selected_node, 'latitude']:.5f}\n"
                f"Longitude: {latest.loc[selected_node, 'longitude']:.5f}"
            ),
            transform=ax.transAxes,
            fontsize=11,
            va="top"
        )
    else:
        ax.text(
            1.02,
            0.72,
            "Click on a node to inspect prediction",
            transform=ax.transAxes,
            fontsize=11,
            va="top"
        )

    plt.draw()



# 15. CLICK HANDLER
def on_click(event):
    if event.inaxes != ax:
        return

    if event.xdata is None or event.ydata is None:
        return

    x_click = event.xdata
    y_click = event.ydata

    closest_node = None
    min_dist = float("inf")

    for node, (x, y_pos) in pos.items():
        dist = (x - x_click) ** 2 + (y_pos - y_click) ** 2

        if dist < min_dist:
            min_dist = dist
            closest_node = node

    draw_graph(closest_node)



# 16. RUN
draw_graph()
fig.canvas.mpl_connect("button_press_event", on_click)
plt.show()