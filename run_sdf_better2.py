from case_studies.sdf.train_sdf_better import train_sdf_with_shuffling
from src import (regular_loss, graph_loss, graph_loss_data_parallel, parse_arguments, get_device)
from src.loss import graph_loss_data_parallel_zero_focused
from src.graph_models_deepmind_noglobal import EncodeProcessDecode
from torch_geometric.nn import DataParallel

# data parameters
args = parse_arguments()
n_objects   = args.n_object
data_folder = args.data_folder
edge_method = args.edge_method
edge_params = {'radius': args.prox_radius, 'min_n_edges': args.min_n_edges, 'max_n_edges': args.max_n_edges}
no_edge = args.with_no_edge_feature  # EncodeProcessDecode specific
no_global = args.with_no_global_feature  # EncodeProcessDecode specific
include_reverse_edge = args.include_reverse_edge   # EncodeProcessDecode specific

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
device      = get_device(args.device)
shuffle     = args.shuffle
data_parallel = isinstance(device, list)
update_data_every = args.update_data_every
n_jobs = args.n_jobs

# setup model and appropriate loss function
assert network_name == "epd"
model = EncodeProcessDecode(n_edge_feat_in=n_edge_in, n_edge_feat_out=n_edge_out,
                            n_node_feat_in=n_node_in, n_node_feat_out=n_node_out,
                            mlp_latent_size=n_hidden[0], num_processing_steps=n_process,
                            process_weights_shared=weights_shared, full_output=full_output)
if data_parallel:
    device_ids = [int(x) for x in device]
    model = DataParallel(model, device_ids=device_ids)
    loss_funcs = [graph_loss_data_parallel]
else:
    loss_funcs = [graph_loss]

# train
train_sdf_with_shuffling(model,
                         data_folder,
                         loss_funcs,
                         update_data_every=update_data_every,
                         n_objects=n_objects,
                         eval_frac=eval_frac,
                         n_epochs=n_epochs,
                         edge_method=edge_method,
                         edge_params=edge_params,
                         no_global=no_global,
                         no_edge=no_edge,
                         print_every=print_every,
                         include_reverse_edges=include_reverse_edge,
                         device=device,
                         save_name=save_name,
                         lr_0=lr_0,
                         lr_scheduler_step_size=lr_step,
                         lr_scheduler_gamma=lr_gamma,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         n_jobs=n_jobs)
