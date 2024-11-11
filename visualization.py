import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

# Assuming 'data' and 'model' are already loaded and the model is in eval mode
from data import data
from eval_BGCN import model

# Convert PyG graph to a NetworkX graph
G = to_networkx(data, to_undirected=True)

# Get model predictions
model.eval()
_, pred = model(data.x, data.edge_index).max(dim=1)
pred = pred.cpu().numpy()

# Define color map, one color for each class/label
color_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'cyan', 5: 'magenta', 6: 'black'}

# Calculate node positions using a layout algorithm
pos = nx.spring_layout(G)

# Draw nodes with colors according to the predictions and add a label for the color
nx.draw_networkx_nodes(G, pos, node_color=[color_map[int(y)] for y in pred], node_size=50, label={color: i for i, color in color_map.items()})
nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5)

# Create a legend with the color map
patch_list = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=color, markersize=10) for label, color in color_map.items()]
plt.legend(handles=patch_list, loc='upper center', ncol=len(color_map.keys()))

# Disable the axis
plt.axis('off')

# Show the plot
plt.show()
