import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

from model.gcn import GCN


class GCN_MRF_MAP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, lambda_mrf):
        super(GCN_MRF_MAP, self).__init__()
        self.lambda_mrf = lambda_mrf
        self.gcn = GCN(input_dim, hidden_dim, output_dim)

    def forward(self, x, edge_index, labels=None):
        # GCN前向传播
        logits = self.gcn(x, edge_index)

        # 如果labels为None，说明我们处于评估模式，只需要返回logits
        if labels is None:
            return logits

        # 计算节点分类损失（负对数似然损失）
        classification_loss = F.nll_loss(F.log_softmax(logits, dim=1), labels)

        # 计算MRF约束项
        mrf_loss = self.compute_mrf_loss(logits, edge_index)

        # 综合考虑节点分类损失和MRF约束项
        total_loss = classification_loss + self.lambda_mrf * mrf_loss  # 使用加法代替减法

        return total_loss

    def compute_mrf_loss(self, logits, edge_index):
        # 使用简单的MRF约束项：对节点之间的相似性进行L2范数惩罚，并确保损失为正
        # 增加了一个小的正则项以避免数值不稳定
        mrf_loss = torch.sum((logits[edge_index[0]] - logits[edge_index[1]]).pow(2)) + 1e-8
        return mrf_loss