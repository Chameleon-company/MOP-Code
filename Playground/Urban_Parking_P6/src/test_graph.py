from graph import G
from visualize import visualize_graph

print("Running test_graph...")

# Pick a node
target = list(G.nodes())[0]

# Call visualization
visualize_graph(G, target_node=target)