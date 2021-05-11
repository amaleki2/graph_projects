from case_studies.sdf import test_sdf, get_sdf_data_loader, get_sdf_data_loader_3d
from src import EncodeProcessDecode, regular_loss, graph_loss, parse_arguments

# data parameters
args = parse_arguments()
n_objects   = args.n_object
data_folder = args.data_folder
edge_method = args.edge_method
edge_params = {'radius': args.prox_radius, 'min_n_edges': args.min_n_edges}
no_edge = not args.edge_features_on  # EncodeProcessDecode specific
no_global = not args.global_features_on  # EncodeProcessDecode specific
include_reverse_edge = args.include_reverse_edge   # EncodeProcessDecode specific
include_self_edge = args.include_self_edge   # EncodeProcessDecode specific
with_sdf_signs = args.with_sdf_signs

# choose model
network_name     = args.network_name
n_node_in        = args.n_node_in
n_node_out       = args.n_node_out
n_hidden         = args.n_hidden
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
model = EncodeProcessDecode(n_edge_feat_in=n_edge_in, n_edge_feat_out=n_edge_out,
                            n_node_feat_in=n_node_in, n_node_feat_out=n_node_out,
                            n_global_feat_in=n_global_in, n_global_feat_out=n_global_out,
                            mlp_latent_size=n_hidden[0], num_processing_steps=n_process,
                            process_weights_shared=weights_shared, full_output=full_output)
loss_funcs = [graph_loss]

# load data
three_d = True
with_normals = True
data_loader, _ = get_sdf_data_loader_3d(n_objects, data_folder, batch_size, eval_frac=0,
                                               edge_method=edge_method, edge_params=edge_params,
                                               no_global=no_global, no_edge=no_edge,
                                               with_normals=with_normals, with_sdf_signs=with_sdf_signs,
                                               reversed_edge_already_included=not include_reverse_edge,
                                               self_edge_already_included=not include_self_edge)

preds, losses = test_sdf(model, data_loader, loss_funcs)

print(sum(losses) / len(losses))