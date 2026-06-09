from graph import G
from visualize import visualize_graph

print("Running test_graph...")

target = list(G.nodes())[0]

# Get neighbors
neighbors = list(G.neighbors(target))

# Create subgraph
sub_nodes = [target] + neighbors
subG = G.subgraph(sub_nodes)

visualize_graph(subG, target_node=target)