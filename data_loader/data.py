from torch_geometric.datasets import Planetoid
from typing import Tuple
import torch
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

# 加载 Cora 数据集
dataset = Planetoid(root='data', name='Cora')
data = dataset[0]  # 获取数据集中的第一个图

def gen_masks(y: torch.Tensor, train_per_class: int = 20, val_per_class: int = 30,
              num_splits: int = 20) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_classes = int(y.max()) + 1

    train_mask = torch.zeros(y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(y.size(0), dtype=torch.bool)

    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        perm = torch.stack(
            [torch.randperm(idx.size(0)) for _ in range(num_splits)], dim=1)
        idx = idx[perm]

        train_idx = idx[:train_per_class]
        train_mask[train_idx] = True
        val_idx = idx[train_per_class:train_per_class + val_per_class]
        val_mask[val_idx] = True

    test_mask = ~(train_mask | val_mask)

    return train_mask, val_mask, test_mask

def create_datasets(data, train_mask, val_mask, test_mask):
    # Use the subgraph function to create subgraphs for training, validation, and testing
    # Make sure to relabel nodes to ensure indices are within the valid range for each subset

    train_edge_index, _ = subgraph(train_mask, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    val_edge_index, _ = subgraph(val_mask, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    test_edge_index, _ = subgraph(test_mask, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)

    # Create the subset datasets with the correct edge indices
    train_data = Data(x=data.x[train_mask], y=data.y[train_mask], edge_index=train_edge_index)
    val_data = Data(x=data.x[val_mask], y=data.y[val_mask], edge_index=val_edge_index)
    test_data = Data(x=data.x[test_mask], y=data.y[test_mask], edge_index=test_edge_index)

    return train_data, val_data, test_data



# 生成掩码
train_mask, val_mask, test_mask = gen_masks(data.y)

# 创建数据集
train_data, val_data, test_data = create_datasets(data, train_mask, val_mask, test_mask)

# 打印数据集
print("Train Data:")
print(train_data)

print("\nValidation Data:")
print(val_data)

print("\nTest Data:")
print(test_data)


def verify_edge_index(edge_index, num_nodes):
    # Calculate the minimum and maximum node indices in edge_index
    min_index = torch.min(edge_index).item()
    max_index = torch.max(edge_index).item()

    # Check if the minimum index is at least 0
    if min_index < 0:
        print("Error: Node indices in edge_index start from below 0.")
        return False

    # Check if the maximum index exceeds num_nodes - 1
    if max_index >= num_nodes:
        print(
            f"Error: Node indices in edge_index exceed the valid range of [0, {num_nodes - 1}]. Maximum index found: {max_index}")
        return False

    print("All node indices in edge_index are within the valid range.")
    return True


# Assuming you have the train_data, val_data, and test_data from your previous message
# Let's verify the edge indices for each dataset
print("Train Data Verification:")
verify_edge_index(train_data.edge_index, train_data.x.shape[0])

print("\nValidation Data Verification:")
verify_edge_index(val_data.edge_index, val_data.x.shape[0])

print("\nTest Data Verification:")
verify_edge_index
