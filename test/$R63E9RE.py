import torch.nn as nn
from case_studies.sdf import train_sdf, get_sdf_3d_data_loader
from src import GATUNet, GraphNetworkIndependentBlock, GraphNetworkBlock, EncodeProcessDecode  # networks
from src import borderless_loss, clamped_loss, graph_loss, level_set_loss  # losses

# get data
n_objects = 2
# data_folder = "../../mesh_gen/mesh_sdf/mesh_correct_sdf2/"
data_folder = "C:/Users/amaleki/Downloads/stl_files"
batch_size = 2
edge_method = 'proximity'  # 'edge'
edge_params = {'radius': 0.25}
train_data, test_data = get_sdf_3d_data_loader(n_objects, data_folder, batch_size, eval_frac=0.1,
                                               edge_method=edge_method, edge_params=edge_params)

# choose model: deep mind graph model
n_edge_feat_in, n_edge_feat_out = 4, 1
n_node_feat_in, n_node_feat_out = 4, 1
n_global_feat_in, n_global_feat_out = 3, 1
mlp_latent_size = 64
num_processing_steps = 5
model = EncodeProcessDecode(n_edge_feat_in=n_edge_feat_in, n_edge_feat_out=n_edge_feat_out,
                            n_node_feat_in=n_node_feat_in, n_node_feat_out=n_node_feat_out,
                            n_global_feat_in=n_global_feat_in, n_global_feat_out=n_global_feat_out,
                            mlp_latent_size=mlp_latent_size, num_processing_steps=num_processing_steps,
                            encoder=GraphNetworkIndependentBlock, decoder=GraphNetworkIndependentBlock,
                            processor=GraphNetworkBlock, output_transformer=GraphNetworkIndependentBlock,
                            full_output=True
                            )