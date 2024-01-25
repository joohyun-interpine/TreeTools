import torch
from torch_geometric.data import Data

# Define the number of nodes and features
num_nodes = 10
num_features = 5  # Set to the desired number of dimensions

# Generate random node features
x = torch.randn(num_nodes, num_features)

# Generate random node positions (assuming 3D positions for simplicity)
pos = torch.randn(num_nodes, 3)

# Generate random batch assignments (assuming a single batch for simplicity)
batch = torch.zeros(num_nodes, dtype=torch.long)

# Create an edge_index for a fully connected graph (for simplicity)
edge_index = torch.cartesian_prod(torch.arange(num_nodes), torch.arange(num_nodes)).t().contiguous()

# Create the PyTorch Geometric Data object
data = Data(x=x, pos=pos, batch=batch, edge_index=edge_index)

# Print the generated data
print(data)