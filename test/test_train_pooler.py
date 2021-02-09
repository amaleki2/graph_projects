import torch.nn as nn
from case_studies.sdf import train_sdf, get_sdf_data_loader, plot_sdf_results, get_pooling_data_loader
from src import (GraphNetworkIndependentBlock, GraphNetworkBlock,
                 EncodeProcessDecodePooled, EncodePooling)  # networks
from src import graph_loss, pooling_loss  # losses

# get data
n_objects = 3
data_folder = "../../mesh_gen/mesh_sdf/mesh_from_numpy_spline1/"
batch_size = 3
edge_method = 'proximity'  # 'edge'
edge_params = {'radius': 0.25}
train_data, test_data = get_sdf_data_loader(n_objects, data_folder, batch_size, eval_frac=0.1,
                                            edge_method=edge_method, edge_params=edge_params)

# choose model: deep mind graph model
n_edge_feat_in, n_edge_feat_out = 3, 1
n_node_feat_in, n_node_feat_out = 3, 1
n_global_feat_in, n_global_feat_out = 3, 1
mlp_latent_size = 16
num_processing_steps = 5

# choose model: deep mind graph model with pooling
model = EncodeProcessDecodePooled(n_edge_feat_in=n_edge_feat_in, n_edge_feat_out=n_edge_feat_out,
                                  n_node_feat_in=n_node_feat_in, n_node_feat_out=n_node_feat_out,
                                  n_global_feat_in=n_global_feat_in, n_global_feat_out=n_global_feat_out,
                                  mlp_latent_size=mlp_latent_size, num_processing_steps=num_processing_steps,
                                  encoder=GraphNetworkIndependentBlock, decoder=GraphNetworkIndependentBlock,
                                  processor=GraphNetworkBlock, output_transformer=GraphNetworkIndependentBlock,
                                  with_pooling=False)
# choose loss functions
save_name = "epd_pool"
loss_funcs = [graph_loss]
train_sdf(model, train_data, test_data, loss_funcs, n_epochs=2, use_cpu=False, save_name=save_name)

# reading pooling data
##
train_data, test_data = get_sdf_data_loader(n_objects, data_folder, 1, eval_frac=0.1,
                                            edge_method=edge_method, edge_params=edge_params)
train_pooling_data = get_pooling_data_loader(train_data, model, batch_size, sdf_model_save_name=save_name)
test_pooling_data = get_pooling_data_loader(test_data, model, batch_size, sdf_model_save_name=save_name)

# pooling model
max_encoding = 3000  # replace with function computing max number of vertices
pooling_model = EncodePooling([max_encoding, max_encoding // 2], max_encoding_size=max_encoding, activate_final=False)

pooling_loss_funcs = [pooling_loss]
train_sdf(pooling_model, train_pooling_data, test_pooling_data, pooling_loss_funcs, n_epochs=2, use_cpu=False, save_name=save_name)