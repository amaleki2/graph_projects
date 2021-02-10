from case_studies.sdf import train_sdf, get_sdf_data_loader, get_pooling_data_loader, compute_max_vertices
from src import (GATUNet, GCNUNet, EncodeProcessDecode, EncodeProcessDecodePooled,  EncodePooling,
                 regular_loss, graph_loss, pooling_loss, parse_arguments)

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
with_pooling     = args.with_pooling     # EncodeProcessDecodePool specific

# train parameters
lr_0        = args.lr
batch_size  = args.batch_size
n_epochs    = args.n_epochs
lr_step     = args.lr_step
lr_gamma    = args.lr_gamma
print_every = args.print_every
save_name   = args.save_name
eval_frac   = args.eval_frac
pooling_model = args.pooling_model

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
    max_encoding = compute_max_vertices(data_folder, n_objects)
    encoding_features = 4
    n_concat_features = 2
    encode_layers = [max_encoding * (encoding_features + n_concat_features), max_encoding * 4, max_encoding * 4, max_encoding * 4]
    decode_layers = [encode_layers[-1], max_encoding * 4, max_encoding * 4]

    model = EncodeProcessDecodePooled(n_edge_feat_in=n_edge_in, n_edge_feat_out=n_edge_out,
                                      n_node_feat_in=n_node_in, n_node_feat_out=n_node_out,
                                      n_global_feat_in=n_global_in, n_global_feat_out=n_global_out,
                                      mlp_latent_size=n_hidden[0], num_processing_steps=n_process,
                                      process_weights_shared=weights_shared, encoding_features=encoding_features,
                                      ae_encode_layers=encode_layers, ae_decode_layers=decode_layers)

    #pooling_loss_func = lambda x, y: 0. if len(x) <= 3 else x[3]
    loss_funcs = [graph_loss]#, pooling_loss_func]
else:
    raise(ValueError("model name %s is not recognized" %network_name))

# load data
if not pooling_model:
    print("running sdf model %s. " %network_name)
    train_data, test_data = get_sdf_data_loader(n_objects, data_folder, batch_size, eval_frac=eval_frac,
                                                edge_method=edge_method, edge_params=edge_params,
                                                no_global=no_global)
    # train
    train_sdf(model, train_data, test_data, loss_funcs, n_epochs=n_epochs, print_every=print_every,
              save_name=save_name, lr_0=lr_0, lr_scheduler_step_size=lr_step, lr_scheduler_gamma=lr_gamma)
else:
    assert 2 == 3, 'dont run this'
    assert network_name == "epd-pool"
    # set batch size to 1
    train_data, test_data = get_sdf_data_loader(n_objects, data_folder, 1, eval_frac=0.1,
                                                edge_method=edge_method, edge_params=edge_params)
    train_pooling_data = get_pooling_data_loader(train_data, model, batch_size, sdf_model_save_name=save_name)
    test_pooling_data = get_pooling_data_loader(test_data, model, batch_size, sdf_model_save_name=save_name)

    # pooling model
    max_encoding = compute_max_vertices(data_folder, n_objects)
    encode_layers = [max_encoding*3, max_encoding*2, max_encoding*2, max_encoding]
    decode_layers = [encode_layers[-1], max_encoding, max_encoding]
    pooling_model = EncodePooling(encode_layers=encode_layers, decode_layers=decode_layers)

    # training pooling model
    pooling_loss_funcs = [pooling_loss]
    pooling_save_name = save_name + "_pooling"
    train_sdf(pooling_model, train_pooling_data, test_pooling_data, pooling_loss_funcs, n_epochs=n_epochs,
              print_every=print_every, lr_0=lr_0, lr_scheduler_step_size=lr_step, lr_scheduler_gamma=lr_gamma,
              save_name=pooling_save_name)
