import torch
from torch_geometric.data import Batch
from src import EncodeProcessDecode, EncodeProcessDecodeNEW

# random node features
torch.manual_seed(0)
x = torch.rand(3, 3).type(torch.float32)

# random edge connections
torch.manual_seed(0)
edge_index = torch.randint(0, 3, (2, 3)).type(torch.long)

# random edge features
torch.manual_seed(0)
edge_attr = torch.rand(3, 5).type(torch.float32)

# random global features
torch.manual_seed(0)
global_attr = torch.rand(1, 2).type(torch.float32)

# run
data = Batch(x=x, edge_index=edge_index, edge_attr=edge_attr, u=global_attr)
net1 = EncodeProcessDecode(n_edge_feat_in=5, n_node_feat_in=3, n_global_feat_in=2,
                           mlp_latent_size=3, num_processing_steps=5, full_output=True)
net2 = EncodeProcessDecodeNEW(n_edge_feat_in=5, n_node_feat_in=3, n_global_feat_in=2,
                              mlp_latent_size=3, num_processing_steps=5, full_output=True)
out = net1(data) + net2(data)
print(out[-1])
