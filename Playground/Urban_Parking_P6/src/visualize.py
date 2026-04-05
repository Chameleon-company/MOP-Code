import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import networkx as nx

def visualize_graph(G, target_node=None):
    pos = nx.spring_layout(G, seed=42)  # stable layout

    colors = []
    sizes = []

    for node in G.nodes():
        if node == target_node:
            colors.append('blue')
            sizes.append(1500)   # BIGGER target node
        elif G.nodes[node].get('occupancy', 0) == 1:
            colors.append('red')
            sizes.append(800)
        else:
            colors.append('green')
            sizes.append(800)

    nx.draw(G, pos,
            node_color=colors,
            node_size=sizes,
            with_labels=True,
            font_size=8)

    plt.title("Parking Graph (Target + Neighbors)")
    plt.show()