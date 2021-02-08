import torch.nn as nn
from case_studies.sdf import train_sdf, get_sdf_data_loader, plot_sdf_results
from src import GATUNet, GraphNetworkIndependentBlock, GraphNetworkBlock, EncodeProcessDecode, EncodeProcessDecodePooled  # networks
from src import borderless_loss, clamped_loss, graph_loss, level_set_loss  # losses

# get data
n_objects = 3
data_folder = "../../mesh_gen/mesh_sdf/mesh_from_numpy_spline1/"
batch_size = 3
edge_method = 'proximity'  # 'edge'
edge_params = {'radius': 0.25}
train_data, test_data = get_sdf_data_loader(n_objects, data_folder, batch_size, eval_frac=0.1,
                                            edge_method=edge_method, edge_params=edge_params)


# choose model: GAT Graph UNet
in_channels, hidden_channels, out_channels = 3, [32, 64, 128, 64, 32], 1
model = GATUNet(in_channels, hidden_channels, out_channels)

# choose loss functions; see src/loss.py for details of parameters
loss_funcs = [borderless_loss, clamped_loss]
losses_params = {"loss_func": nn.L1Loss, "radius": 0.05, "maxv": 0.025}

# train
gamma       = 0.5
lr_0        = 0.001
n_epochs     = 2
save_name   = "gat_unet"

# for the sake of testing, same data used for train and test.
# train_sdf(model, train_data, train_data, loss_funcs, n_epochs=n_epochs, use_cpu=True,
#           save_name=save_name, **losses_params)

# choose model: deep mind graph model
n_edge_feat_in, n_edge_feat_out = 3, 1
n_node_feat_in, n_node_feat_out = 3, 1
n_global_feat_in, n_global_feat_out = 3, 1
mlp_latent_size = 64
num_processing_steps = 5
model = EncodeProcessDecode(n_edge_feat_in=n_edge_feat_in, n_edge_feat_out=n_edge_feat_out,
                            n_node_feat_in=n_node_feat_in, n_node_feat_out=n_node_feat_out,
                            n_global_feat_in=n_global_feat_in, n_global_feat_out=n_global_feat_out,
                            mlp_latent_size=mlp_latent_size, num_processing_steps=num_processing_steps,
                            encoder=GraphNetworkIndependentBlock, decoder=GraphNetworkIndependentBlock,
                            processor=GraphNetworkBlock, output_transformer=GraphNetworkIndependentBlock,
                            full_output=False)

# # choose loss functions
# save_name = "epd"
# loss_funcs = [graph_loss]
# train_sdf(model, train_data, train_data, loss_funcs, n_epochs=2, use_cpu=True, save_name=save_name)

# choose model: deep mind graph model with pooling
model = EncodeProcessDecodePooled(n_edge_feat_in=n_edge_feat_in, n_edge_feat_out=n_edge_feat_out,
                                  n_node_feat_in=n_node_feat_in, n_node_feat_out=n_node_feat_out,
                                  n_global_feat_in=n_global_feat_in, n_global_feat_out=n_global_feat_out,
                                  mlp_latent_size=mlp_latent_size, num_processing_steps=num_processing_steps,
                                  encoder=GraphNetworkIndependentBlock, decoder=GraphNetworkIndependentBlock,
                                  processor=GraphNetworkBlock, output_transformer=GraphNetworkIndependentBlock)
# choose loss functions
save_name = "epd_pool"
pooling_loss_func = lambda x, y: 0. if len(x) <= 3 else x[3]
loss_funcs = [graph_loss, pooling_loss_func]
train_sdf(model, train_data, train_data, loss_funcs, n_epochs=2, use_cpu=False, save_name=save_name)

# visualization
data_loader, _ = get_sdf_data_loader(10, data_folder, 1, eval_frac=0, edge_method=edge_method, edge_params=edge_params)
output_func = lambda x: x[-1][1].numpy().reshape(-1)
plot_sdf_results(model, data_loader, save_name=save_name, output_func=output_func, levels=[-0.1, 0., 0.1])