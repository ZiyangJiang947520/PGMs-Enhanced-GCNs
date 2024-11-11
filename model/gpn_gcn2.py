import torch
from torch import nn, softmax
from torch_geometric.nn import GCNConv, APPNP
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch.nn.functional as F

class BernoulliDensity(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(BernoulliDensity, self).__init__()
        self.logits = GCNConv(hidden_dim, output_dim)  # 注意参数的调整

    def forward(self, x, edge_index):
        logits = self.logits(x, edge_index)
        probs = torch.sigmoid(logits)
        return probs



class Evidence(nn.Module):
    def __init__(self, num_classes):
        super(Evidence, self).__init__()
        self.evidence = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        return F.relu(self.evidence(x))  # 确保证据分数非负


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

# 导入BayesianGCNConv
from model.BGCN import BayesianGCNConv


class GPN_GCN_with_CRF(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, alpha=0.1, beta=1.0):
        super(GPN_GCN_with_CRF, self).__init__()
        self.gcn1 = GCNConv(num_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)  # 第二个GCN层保持特征维度不变
        self.density = BernoulliDensity(hidden_dim, hidden_dim)  # 这里假设输出维度与输入维度相同
        self.classifier = nn.Linear(hidden_dim, num_classes)  # 将隐藏层特征转换为类别数
        self.evidence = Evidence(num_classes)  # Evidence层接收分类器的输出
        self.appnp = APPNP(K=10, alpha=0.1)
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.gcn1(x, edge_index))
        x = F.dropout(x, training=self.training)

        # CRF层可以直接在这里应用，因为它与隐藏层的维度兼容
        x = self.crf_layer(x, edge_index, edge_weight)

        x = F.relu(self.gcn2(x, edge_index))

        # 应用分类器
        logits = self.classifier(x)

        # 应用Evidence层
        evidence = self.evidence(logits)

        # 将证据通过APPNP层进行传播
        x = self.appnp(evidence, edge_index)

        return F.log_softmax(x, dim=1)

    def crf_layer(self, x, edge_index, edge_weight=None):
        num_nodes = x.size(0)
        x_crf = torch.zeros_like(x)
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), device=x.device)

        for i in range(edge_index.size(1)):
            src, dst = edge_index[:, i]
            weight = edge_weight[i]
            x_crf[src] += self.beta * weight * x[dst]
            x_crf[dst] += self.beta * weight * x[src]

        # 注意：这里不需要softmax，直接将加权后的特征进行组合即可
        x_combined = (1 - self.alpha) * x + self.alpha * x_crf
        return x_combined

# Evidence类定义保持不变

# 使用和训练模型的代码也保持不变






