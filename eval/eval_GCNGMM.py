from torch_geometric.datasets import Planetoid
import joblib
import torch
import numpy as np
from model.GCN_GMM import GCN_GMM  # 确保正确引用了您的GCN_GMM类

def set_seed(seed=42):
    """固定随机种子以确保实验可复现性。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()  # 调用此函数以固定种子

# 加载数据集
dataset = Planetoid(root='data', name='Cora')
data = dataset[0]

# 模型参数
input_dim = dataset.num_features
hidden_dim = 16
output_dim = dataset.num_classes
num_gaussians = 10

# 初始化模型并加载权重
model = GCN_GMM(input_dim, hidden_dim, output_dim, num_gaussians)
model.load_state_dict(torch.load('model_GCNGMM.pth'))

# 加载GMM参数
gmm = joblib.load('model_GMM.joblib')

model.eval()

def evaluate(dataset):
    with torch.no_grad():
        gcn_out, _ = model(dataset.x, dataset.edge_index)  # 使用GCN提取特征
        gcn_out = gcn_out.detach().cpu().numpy()  # 转换为NumPy数组
        gmm_pred = gmm.predict(gcn_out)  # 使用GMM进行预测
        gmm_pred = torch.tensor(gmm_pred)  # 将NumPy数组转换回Tensor
        correct = gmm_pred.eq(dataset.y.cpu()).sum().item()  # 计算正确预测的数量
        total = dataset.y.size(0)
        acc = correct / total
        return acc

# 评估模型
val_acc = evaluate(data)  # 这里使用的data应该是具有正确掩码的数据
print(f'Accuracy: {val_acc:.4f}')
