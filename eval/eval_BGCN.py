from torch_geometric.datasets import Planetoid
from model.BGCN import BayesianGCN
from data import gen_masks, create_datasets

import torch
import numpy as np
import random

def set_seed(seed=42):
    """固定随机种子以确保实验可复现性。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()  # 调用此函数以固定种子

# 接下来是你的模型初始化、训练和评估代码
# ...

# Load dataset
dataset = Planetoid(root='data', name='Cora')
data = dataset[0]

# Generate masks
train_mask, val_mask, test_mask = gen_masks(data.y)

# Create datasets
train_data, val_data, test_data = create_datasets(data, train_mask, val_mask, test_mask)

# Model parameters
input_dim = dataset.num_features
hidden_dim = 16
output_dim = dataset.num_classes

# Initialize model and load weights
model = BayesianGCN(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load('model_BGCN.pth'))
model.eval()

# Evaluation function
def evaluate(dataset):
    out = model(dataset.x, dataset.edge_index)
    pred = out.argmax(dim=1)
    correct = pred.eq(dataset.y).sum().item()
    total = dataset.y.size(0)
    acc = correct / total
    return acc

# Evaluate model
val_acc = evaluate(val_data)
test_acc = evaluate(test_data)
print(f'Validation Acc: {val_acc:.4f}')
print(f'Test Acc: {test_acc:.4f}')
