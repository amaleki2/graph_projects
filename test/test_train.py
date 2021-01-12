import torch.nn as nn
from src import (train_sdf, get_sdf_data_loader, plot_sdf_results,
                 GatUNet2, GraphNetworkIndependentBlock, GraphNetworkBlock, EncodeProcessDecodeNEW,
                 borderless_loss, clamped_loss, deep_mind_loss, level_set_loss)

# get data
n_objects = 20
data_folder = "../../mesh_gen/mesh_sdf/mesh_correct_sdf2/"
batch_size = 2
edge_method = 'proximity'  # 'edge'
edge_params = {'radius': 0.25}
train_data, test_data = get_sdf_data_loader(n_objects, data_folder, batch_size, eval_frac=0.1,
                                            edge_method=edge_method, edge_params=edge_params)


# choose model: GAT Graph UNet
# in_channels, hidden_channels, out_channels = 3, [32, 64, 128, 64, 32], 1
# model = GatUNet2(in_channels, hidden_channels, out_channels)

# # choose loss functions; see src/loss.py for details of parameters
# loss_funcs = [borderless_loss, clamped_loss]
# losses_params = {"loss_func": nn.L1Loss, "radius": 0.05, "maxv": 0.025}
#
# # train
# gamma       = 0.5
# lr_0        = 0.001
# n_epoch     = 100
# step_size   = 50
# print_every = 25
# save_name   = "gat_unet"
#
# # for the sake of testing, same data used for train and test.
# train_sdf(model, train_data, train_data, loss_funcs, n_epoch=n_epoch, print_every=print_every,
#           save_name=save_name, lr_0=lr_0, step_size=step_size, gamma=gamma, **losses_params)


# choose model: deep mind graph model
n_edge_feat_in, n_edge_feat_out = 3, 1
n_node_feat_in, n_node_feat_out = 3, 1
n_global_feat_in, n_global_feat_out = 3, 1
mlp_latent_size = 64
num_processing_steps = 5
model = EncodeProcessDecodeNEW(n_edge_feat_in=n_edge_feat_in, n_edge_feat_out=n_edge_feat_out,
                               n_node_feat_in=n_node_feat_in, n_node_feat_out=n_node_feat_out,
                               n_global_feat_in=n_global_feat_in, n_global_feat_out=n_global_feat_out,
                               mlp_latent_size=mlp_latent_size, num_processing_steps=num_processing_steps,
                               encoder=GraphNetworkIndependentBlock, decoder=GraphNetworkIndependentBlock,
                               processor=GraphNetworkBlock, output_transformer=GraphNetworkIndependentBlock,
                               full_output=True
                               )

# choose loss functions
loss_funcs = [deep_mind_loss, level_set_loss]

# train_sdf(model, train_data, train_data, loss_funcs, n_epoch=100, print_every=25,
#           save_name="", lr_0=1e-3, step_size=200, gamma=0.2, use_cpu=True)

# visualization
save_name = "manual_sdf_proximity_nonuniform_complex"
data_loader, _ = get_sdf_data_loader(10, data_folder, 1, eval_frac=0, edge_method=edge_method, edge_params=edge_params)
output_func = lambda x: x[-1][1].numpy().reshape(-1)
plot_sdf_results(model, data_loader, save_name=save_name, output_func=output_func, levels=[-0.1, 0., 0.1])