import torch
import torch.nn.functional as F
from torch import optim
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
from model.gpn_gcn import GPN_GCN  # 确保这个是你修改后的BayesianGCN
from data import gen_masks, create_datasets

# 加载数据集
dataset = Planetoid(root='data', name='Cora')
data = dataset[0]

# 数据预处理
data.adj_t = to_undirected(data.edge_index)
train_mask, val_mask, test_mask = gen_masks(data.y)
train_data, val_data, test_data = create_datasets(data, train_mask, val_mask, test_mask)

# 模型参数
input_dim = dataset.num_features
hidden_dim = 16
output_dim = dataset.num_classes
dropout_rate = torch.tensor(0.5, requires_grad=True)

# 初始化模型
model = GPN_GCN(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam([{'params': model.parameters()}, {'params': [dropout_rate]}], lr=0.01)

# 训练循环
def train():
    model.train()
    optimizer.zero_grad()
    out = model(train_data.x, train_data.edge_index)
    loss = F.nll_loss(out, train_data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

# 验证函数
def test(dataset):
    model.eval()
    correct = 0
    total = 0
    for _ in range(30):  # 进行30次推理，实现蒙特卡洛Dropout
        out = model(dataset.x, dataset.edge_index)
        pred = out.argmax(dim=1)
        correct += pred.eq(dataset.y).sum().item()
        total += dataset.y.size(0)
    acc = correct / (total * 30)  # 计算平均准确率
    return acc

# 训练模型
for epoch in range(200):
    loss = train()
    train_acc = test(train_data)
    val_acc = test(val_data)
    print(f'Epoch: {epoch+1}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

# 测试模型
test_acc = test(test_data)
print(f'Test Acc: {test_acc:.4f}')
torch.save(model.state_dict(), 'model_gpngcn.pth')