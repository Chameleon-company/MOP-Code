import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()

        # First graph convolution layer
        self.conv1 = GCNConv(input_dim, hidden_dim)

        # Second graph convolution layer
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Second layer
        x = self.conv2(x, edge_index)

        return x