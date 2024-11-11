import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import softmax


class GCN_with_CRF(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, alpha=0.1, beta=1.0):
        super(GCN_with_CRF, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)

        # CRF layer
        x = self.crf_layer(x, edge_index, edge_weight)

        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def crf_layer(self, x, edge_index, edge_weight=None):
        # 获取节点数
        num_nodes = x.size(0)

        # 初始化CRF更新后的节点特征
        x_crf = torch.zeros_like(x)

        # 如果没有提供边权重，我们假设所有边的权重相等
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), device=x.device)

        # 对于每一条边，更新相连的节点特征
        for i in range(edge_index.size(1)):
            src, dst = edge_index[:, i]
            weight = edge_weight[i]

            # 根据CRF更新规则来更新节点特征
            x_crf[src] += self.beta * weight * x[dst]
            x_crf[dst] += self.beta * weight * x[src]

        # 应用softmax来保证更新后的特征值分布在一个合理的范围内
        x_crf = softmax(x_crf, torch.arange(num_nodes, device=x.device))

        # 将CRF层的输出与原始特征结合
        x_combined = (1 - self.alpha) * x + self.alpha * x_crf
        return x_combined
