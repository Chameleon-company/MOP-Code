import matplotlib
matplotlib.use('TkAgg')   # for popup window

import matplotlib.pyplot as plt
import networkx as nx

def visualize_graph(G, target_node=None):
    pos = nx.spring_layout(G)

    colors = []
    for node in G.nodes():
        if node == target_node:
            colors.append('blue')
        elif G.nodes[node].get('occupancy', 0) == 1:
            colors.append('red')
        else:
            colors.append('green')

    nx.draw(G, pos,
            node_color=colors,
            with_labels=True,
            node_size=800)

    plt.title("Parking Graph Visualization")
    plt.show()