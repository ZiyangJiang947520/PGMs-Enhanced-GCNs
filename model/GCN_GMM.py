import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.mixture import GaussianMixture

class GCN_GMM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gaussians):
        super(GCN_GMM, self).__init__()
        self.gcn = GCNConv(input_dim, hidden_dim)
        self.num_gaussians = num_gaussians
        self.gmm = GaussianMixture(n_components=num_gaussians)
        self.output_dim = output_dim
        self.fitted = False  # 新增一个标志来指示GMM是否已拟合

    def forward(self, x, edge_index):
        x = F.relu(self.gcn(x, edge_index))
        if self.fitted:  # 只有当GMM已经拟合时才进行预测
            x_gmm = self.predict_gmm(x)
        else:
            x_gmm = None
        return x, x_gmm

    def fit_gmm(self, x):
        self.gmm.fit(x.detach().cpu().numpy())
        self.fitted = True  # 拟合GMM后设置标志

    def predict_gmm(self, x):
        if not self.fitted:  # 使用self.fitted标志而不是检查模型属性
            raise RuntimeError("GaussianMixture model has not been fitted yet. "
                               "Call 'fit_gmm' with appropriate arguments before using this estimator.")
        return torch.Tensor(self.gmm.predict_proba(x.detach().cpu().numpy()))
