import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.utils import to_undirected
from data_2 import gen_masks, create_datasets
from model.GCN_CRF import GCN_with_CRF

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Load dataset
dataset = Amazon(root='data', name='Photo')
data = dataset[0].to(device)
data.adj_t = to_undirected(data.edge_index)

# Generate masks
train_mask, val_mask, test_mask = gen_masks(data.y)
train_mask = train_mask.to(device)
val_mask = val_mask.to(device)
test_mask = test_mask.to(device)
# Create datasets
train_data, val_data, test_data = create_datasets(data, train_mask, val_mask, test_mask)

# Model parameters
input_dim = dataset.num_features
hidden_dim = 16
output_dim = dataset.num_classes

# Initialize model
# model = GCN(input_dim, hidden_dim, output_dim)
model = GCN_with_CRF(input_dim, hidden_dim, output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(train_data.x, train_data.edge_index)
    loss = F.nll_loss(out, train_data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

# Train model
for epoch in range(200):
    loss = train()
    print(f'Epoch: {epoch+1}, Loss: {loss:.4f}')

# Save model
torch.save(model.state_dict(), 'model_GCNCRF_AP.pth')

