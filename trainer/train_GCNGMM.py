import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import DataLoader
from sklearn.mixture import GaussianMixture
from model.GCN_GMM import GCN_GMM
import joblib

# 假设model.gmm是您的GaussianMixture实例


# 加载数据集
dataset = Planetoid(root='data', name='Cora')
data = dataset[0]  # 获取数据集中的第一个图

# 模型参数
input_dim = dataset.num_features
hidden_dim = 16
output_dim = dataset.num_classes
num_gaussians = 10

# 初始化模型
model = GCN_GMM(input_dim, hidden_dim, output_dim, num_gaussians)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 创建数据加载器
train_loader = DataLoader([data], batch_size=1, shuffle=True)

# 训练模型
# 训练模型
for epoch in range(200):
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index)  # 使用GCN的输出
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')



# 假设这里已经完成了GCN的训练
model.eval()
with torch.no_grad():
    features = []
    for data in train_loader:  # 假设train_loader包含所有数据
        out, _ = model(data.x, data.edge_index)
        features.append(out)
    features = torch.cat(features, 0)
    model.fit_gmm(features)  # 拟合GMM

# 保存模型参数
torch.save(model.state_dict(), 'model_GCNGMM.pth')
joblib.dump(model.gmm, 'model_GMM.joblib')