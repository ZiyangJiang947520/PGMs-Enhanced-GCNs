import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import softmax

# Learnable Bernoulli Dropout 类
class LearnableBernoulliDropout(nn.Module):
    def __init__(self):
        super(LearnableBernoulliDropout, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始化dropout率为0.5

    def forward(self, x):
        dropout_rate = torch.sigmoid(self.alpha)  # 将alpha转换为dropout率
        if self.training:
            mask = torch.bernoulli(torch.ones_like(x) * (1 - dropout_rate))
            return x * mask
        else:
            return x

# 合并 GCN_with_CRF 和 GCN_LBD
class GCN_Combined(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, alpha=0.1, beta=1.0):
        super(GCN_Combined, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.crf_layer = GCNConv(hidden_dim, hidden_dim)  # 作为CRF层
        self.dropout1 = LearnableBernoulliDropout()  # 放在CRF层之后的dropout
        self.conv2 = GCNConv(hidden_dim, output_dim)  # 第二个GCN层直接输出分类结果
        self.alpha = alpha  # 这些是CRF层可能用到的参数，具体实现时可以调整
        self.beta = beta

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index))  # 第一层GCN
        x_crf = self.crf_layer(x, edge_index, edge_weight)  # CRF层的输出
        x = self.alpha * x + self.beta * x_crf  # 将原始特征和CRF输出相结合
        x = self.dropout1(x)  # 应用dropout
        x = self.conv2(x, edge_index)  # 第二层GCN
        return F.log_softmax(x, dim=1)

