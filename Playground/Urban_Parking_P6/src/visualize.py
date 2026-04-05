import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


def generate_insights(G, target_node=None):
    """
    Generate short insights for presentation based on the graph.
    """
    insights = []

    occupied_nodes = [n for n in G.nodes() if G.nodes[n].get("occupancy", 0) == 1]
    free_nodes = [n for n in G.nodes() if G.nodes[n].get("occupancy", 0) == 0]

    # Insight 1: nearby occupied bays
    if target_node is not None:
        neighbors = list(G.neighbors(target_node))
        occupied_neighbors = [
            n for n in neighbors if G.nodes[n].get("occupancy", 0) == 1
        ]

        if len(occupied_neighbors) > 0:
            insights.append("Nearby occupied bays influence prediction.")
        else:
            insights.append("Most nearby bays around the target node are free.")

    # Insight 2: cluster visibility
    if len(occupied_nodes) > len(free_nodes):
        insights.append("Clusters of full parking are observed.")
    else:
        insights.append("A mix of free and occupied bays is observed.")

    # Insight 3: congestion pattern
    if len(occupied_nodes) >= max(1, len(G.nodes()) // 2):
        insights.append("Peak congestion zones are visible in this area.")
    else:
        insights.append("Congestion appears moderate in this selected area.")

    return insights


def visualize_graph(G, target_node=None):
    """
    Visualize parking graph with:
    - blue = target node
    - red = occupied
    - green = free
    - legend
    - explanation/insight text
    """

    # Stable layout
    pos = nx.spring_layout(G, seed=42)

    colors = []
    sizes = []

    for node in G.nodes():
        if node == target_node:
            colors.append("blue")
            sizes.append(1500)
        elif G.nodes[node].get("occupancy", 0) == 1:
            colors.append("red")
            sizes.append(900)
        else:
            colors.append("green")
            sizes.append(900)

    plt.figure(figsize=(12, 8))

    nx.draw(
        G,
        pos,
        node_color=colors,
        node_size=sizes,
        with_labels=True,
        font_size=8,
        font_weight="bold",
        edge_color="gray"
    )

    # Title
    plt.title(
        "Parking Occupancy Graph: Target Node and Nearby Bays",
        fontsize=14,
        fontweight="bold"
    )

    # Legend
    blue_patch = mpatches.Patch(color="blue", label="Target Node")
    red_patch = mpatches.Patch(color="red", label="Occupied")
    green_patch = mpatches.Patch(color="green", label="Free")

    plt.legend(handles=[blue_patch, red_patch, green_patch], loc="upper right")

    # Insights text
    insights = generate_insights(G, target_node)
    insight_text = "\n".join([f"• {insight}" for insight in insights])

    plt.figtext(
        0.02,
        0.02,
        "Graph Insights:\n" + insight_text,
        fontsize=10,
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5")
    )

    plt.tight_layout()
    plt.show()