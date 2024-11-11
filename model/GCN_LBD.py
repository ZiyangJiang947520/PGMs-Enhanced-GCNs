import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from data import data  # 假设data.py包含数据加载和处理的代码
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_dense_adj

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from data import data  # 假设data.py包含数据加载和处理的代码
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_dense_adj

# 定义GCN_LBD模型
class GCN_LBD(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN_LBD, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = LearnableBernoulliDropout()

    def forward(self, x, edge_index):
        # 第一层GCN
        x = F.relu(self.conv1(x, edge_index))
        # 应用Learnable Bernoulli Dropout
        x = self.dropout(x)
        # 第二层GCN
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Learnable Bernoulli Dropout 类
class LearnableBernoulliDropout(nn.Module):
    def __init__(self):
        super(LearnableBernoulliDropout, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始化dropout率为0.5

    def forward(self, x):
        dropout_rate = torch.sigmoid(self.alpha)  # 将alpha转换为dropout率
        if self.training:  # 只在训练时应用dropout
            mask = torch.bernoulli(torch.ones_like(x) * (1 - dropout_rate))
            return x * mask
        else:
            return x



