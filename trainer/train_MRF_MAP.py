import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
from model.GCN_MRF_MAP import GCN_MRF_MAP
from data import gen_masks, create_datasets

# Load dataset
dataset = Planetoid(root='data', name='Cora')
data = dataset[0]
data.adj_t = to_undirected(data.edge_index)

# Generate masks
train_mask, val_mask, test_mask = gen_masks(data.y)

# Create datasets
train_data, val_data, test_data = create_datasets(data, train_mask, val_mask, test_mask)

# Model parameters
input_dim = dataset.num_features
hidden_dim = 16
output_dim = dataset.num_classes
lambda_mrf = 0.1
# Initialize model
# model = GCN(input_dim, hidden_dim, output_dim)
model = GCN_MRF_MAP(input_dim, hidden_dim, output_dim, lambda_mrf)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    total_loss = model(train_data.x, train_data.edge_index, train_data.y)

    total_loss.backward()
    optimizer.step()
    return total_loss.item()


# Train model
for epoch in range(200):
    loss = train()
    print(f'Epoch: {epoch+1}, Loss: {loss:.4f}')

# Save model
torch.save(model.state_dict(), 'model_GCN_MRF_MAP.pth')

