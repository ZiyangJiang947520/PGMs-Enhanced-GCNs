import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import add_self_loops, degree, dropout_adj

# Bayesian Graph Convolutional Layer
class BayesianGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, dropout=0.5, aggr='add'):
        super(BayesianGCNConv, self).__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.conv = GCNConv(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index):
        # Apply MC Dropout to edges
        edge_index, _ = dropout_adj(edge_index, p=self.dropout, force_undirected=True, num_nodes=x.size(0), training=self.training)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.conv(x, edge_index)

# GCN Model with Bayesian Layers
class BayesianGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(BayesianGCN, self).__init__()
        self.conv1 = BayesianGCNConv(input_dim, hidden_dim, dropout=dropout)
        self.conv2 = BayesianGCNConv(hidden_dim, output_dim, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
