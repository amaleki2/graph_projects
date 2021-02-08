from case_studies.sdf import train_sdf, get_sdf_data_loader
from src import (GATUNet, GCNUNet, EncodeProcessDecode, EncodeProcessDecodePooled,
                 regular_loss, graph_loss, parse_arguments)

# data parameters
args = parse_arguments()
n_objects   = args.n_object
data_folder = args.data_folder
edge_method = args.edge_method
edge_params = {'radius': args.prox_radius}
no_global = not args.global_features_on  # epd specific\

# choose model
network_name     = args.network_name
n_node_in        = args.n_node_in
n_node_out       = args.n_node_out
n_hidden         = args.n_hidden
last_layer_skip  = args.last_layer_skip  # GATUNet and GCNUNet specific
heads            = args.head             # GATUNet specific
negative_slope   = args.negative_slope   # GATUNet specific
n_edge_in        = args.n_edge_in        # EncodeProcessDecode specific
n_edge_out       = args.n_edge_out       # EncodeProcessDecode specific
n_global_in      = args.n_global_in      # EncodeProcessDecode specific
n_global_out     = args.n_global_out     # EncodeProcessDecode specific
n_process        = args.n_process        # EncodeProcessDecode specific
full_output      = args.full_output      # EncodeProcessDecode specific
weights_shared   = args.weights_shared   # EncodeProcessDecode specific

# train parameters
lr_0        = args.lr
batch_size  = args.batch_size
n_epochs    = args.n_epochs
lr_step     = args.lr_step
lr_gamma    = args.lr_gamma
print_every = args.print_every
save_name   = args.save_name
eval_frac   = args.eval_frac

# setup model and appropriate loss function
if network_name == "gat":
    model = GATUNet(n_node_in, n_hidden, n_node_out, heads=heads, negative_slope=negative_slope,
                    with_last_layer_skip_connection=last_layer_skip)
    loss_funcs = [regular_loss]
elif network_name == "gcn":
    model = GCNUNet(n_node_in, n_hidden, n_node_out, with_last_layer_skip_connection=last_layer_skip)
    loss_funcs = [regular_loss]
elif network_name == "epd":
    model = EncodeProcessDecode(n_edge_feat_in=n_edge_in, n_edge_feat_out=n_edge_out,
                                n_node_feat_in=n_node_in, n_node_feat_out=n_node_out,
                                n_global_feat_in=n_global_in, n_global_feat_out=n_global_out,
                                mlp_latent_size=n_hidden[0], num_processing_steps=n_process,
                                process_weights_shared=weights_shared, full_output=full_output)
    loss_funcs = [graph_loss]
elif network_name == "epd-pool":
    model = EncodeProcessDecodePooled(n_edge_feat_in=n_edge_in, n_edge_feat_out=n_edge_out,
                                      n_node_feat_in=n_node_in, n_node_feat_out=n_node_out,
                                      n_global_feat_in=n_global_in, n_global_feat_out=n_global_out,
                                      mlp_latent_size=n_hidden[0], num_processing_steps=n_process,
                                      process_weights_shared=weights_shared, with_pooling=True)
    pooling_loss_func = lambda x, y: 0. if len(x) <= 3 else x[3]
    loss_funcs = [graph_loss, pooling_loss_func]
else:
    raise(ValueError("model name %s is not recognized" %network_name))

# load data
train_data, test_data = get_sdf_data_loader(n_objects, data_folder, batch_size, eval_frac=eval_frac,
                                            edge_method=edge_method, edge_params=edge_params,
                                            no_global=no_global)
# train
train_sdf(model, train_data, test_data, loss_funcs, n_epochs=n_epochs, print_every=print_every,
          save_name=save_name, lr_0=lr_0, lr_scheduler_step_size=lr_step, lr_scheduler_gamma=lr_gamma)