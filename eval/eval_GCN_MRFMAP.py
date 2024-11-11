from torch_geometric.datasets import Planetoid
from model.GCN_MRF_MAP import GCN_MRF_MAP
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
lambda_mrf = 0.01
# Initialize model and load weights
model = GCN_MRF_MAP(input_dim, hidden_dim, output_dim, lambda_mrf = 0.01)
model.load_state_dict(torch.load('model_GCN_MRF_MAP.pth'))
model.eval()

# Evaluation function
# Evaluation function modified to remove labels from model forward call
def evaluate(model, data):
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():  # No gradients needed for evaluation
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()  # Evaluate on test mask
        total = data.test_mask.sum().item()
        acc = correct / total
    return acc


# Evaluate model
val_acc = evaluate(model, val_data)
test_acc = evaluate(model, test_data)
print(f'Validation Acc: {val_acc:.4f}')
print(f'Test Acc: {test_acc:.4f}')
