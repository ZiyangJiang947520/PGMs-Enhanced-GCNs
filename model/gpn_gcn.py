import torch
from torch import nn
from torch_geometric.nn import GCNConv, APPNP
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch.nn.functional as F

class BernoulliDensity(nn.Module):
    def __init__(self, hidden_dim, num_features):
        super(BernoulliDensity, self).__init__()
        # 对于 Bernoulli 分布，我们只需要输出特征为 1 的概率
        self.logits = GCNConv(hidden_dim, num_features)

    def forward(self, x, edge_index):
        logits = self.logits(x, edge_index)
        probs = torch.sigmoid(logits)  # 将输出转换为概率
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

class GPN_GCN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
        super(GPN_GCN, self).__init__()
        # 使用BayesianGCNConv作为卷积层
        self.conv1 = BayesianGCNConv(num_features, hidden_dim, dropout=dropout)
        self.conv2 = BayesianGCNConv(hidden_dim, hidden_dim, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.evidence = Evidence(num_classes)  # 使用Evidence
        self.appnp = APPNP(K=10, alpha=0.1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)  # 这里的dropout实际上是多余的，因为已经在BayesianGCNConv中应用
        x = F.relu(self.conv2(x, edge_index))
        logits = self.classifier(x)
        evidence = self.evidence(logits)
        x = self.appnp(evidence, edge_index)
        return F.log_softmax(x, dim=1)

# Evidence类定义保持不变

# 使用和训练模型的代码也保持不变






