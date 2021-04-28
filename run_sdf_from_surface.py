import torch
import torch.nn as nn
from case_studies.sdf.train_sdf import train_sdf
from case_studies.sdf_from_surface.get_data import get_data_loader
from src import GraphNetworkIndependentBlock, GraphNetworkBlock, graph_loss, parse_arguments


class EncodeProcessDecode2(torch.nn.Module):
    def __init__(self,
                 n_edge_feat_in=1, n_node_feat_in=1, n_global_feat_in=1,
                 n_edge_feat_out=1, n_node_feat_out=1, n_global_feat_out=1,
                 encoder=GraphNetworkIndependentBlock, processor=GraphNetworkBlock,
                 decoder=GraphNetworkIndependentBlock, output_transformer=GraphNetworkIndependentBlock,
                 mlp_latent_size=128, num_processing_steps=5,
                 process_weights_shared=False, normalize=True, full_output=False):
        super(EncodeProcessDecode2, self).__init__()
        assert not (n_edge_feat_in is None or n_node_feat_in is None or n_global_feat_in is None), \
            "input sizes should be specified"
        self.num_processing_steps = num_processing_steps
        self.full_output = full_output
        self.process_weights_shared = process_weights_shared
        self.encoder = encoder(n_edge_feat_in, mlp_latent_size,
                               n_node_feat_in, mlp_latent_size,
                               n_global_feat_in, mlp_latent_size,
                               latent_sizes=mlp_latent_size,
                               activate_final=True,
                               normalize=normalize)
        if not self.process_weights_shared:
            self.processors = torch.nn.ModuleList()
            for _ in range(num_processing_steps):
                self.processors.append(processor(2 * mlp_latent_size, mlp_latent_size,
                                                 2 * mlp_latent_size, mlp_latent_size,
                                                 2 * mlp_latent_size, mlp_latent_size,
                                                 latent_sizes=mlp_latent_size,
                                                 activate_final=True,
                                                 normalize=normalize))
        else:
            self.processors = processor(2 * mlp_latent_size, mlp_latent_size,
                                        2 * mlp_latent_size, mlp_latent_size,
                                        2 * mlp_latent_size, mlp_latent_size,
                                        latent_sizes=mlp_latent_size,
                                        activate_final=True,
                                        normalize=normalize)

        self.decoder = decoder(mlp_latent_size, mlp_latent_size,
                               mlp_latent_size+3, mlp_latent_size,
                               mlp_latent_size, mlp_latent_size,
                               latent_sizes=mlp_latent_size,
                               activate_final=True,
                               normalize=normalize)

        self.output_transformer = output_transformer(mlp_latent_size, n_edge_feat_out,
                                                     mlp_latent_size, n_node_feat_out,
                                                     mlp_latent_size, n_global_feat_out,
                                                     latent_sizes=None,
                                                     activate_final=False, normalize=False)

    def forward(self, data):
        edge_attr, edge_index, node_attr, y, z, global_attr, batch = data.edge_attr, data.edge_index, data.x, data.y, data.z, \
                                                             data.u, data.batch
        edge_attr, node_attr, global_attr = self.encoder(edge_attr, node_attr, global_attr, edge_index, batch)
        edge_attr0, node_attr0, global_attr0 = edge_attr.clone(), node_attr.clone(), global_attr.clone()

        for i in range(self.num_processing_steps):
            edge_attr = torch.cat((edge_attr0, edge_attr), dim=1)
            node_attr = torch.cat((node_attr0, node_attr), dim=1)
            global_attr = torch.cat((global_attr0, global_attr), dim=1)
            if not self.process_weights_shared:
                edge_attr, node_attr, global_attr = self.processors[i](edge_attr, node_attr, global_attr, edge_index,
                                                                       batch)
            else:
                edge_attr, node_attr, global_attr = self.processors(edge_attr, node_attr, global_attr, edge_index,
                                                                 batch)
        n_node_features = node_attr.shape[0]
        output_ops = []
        for yi in y:
            yi = torch.repeat_interleave(yi.view(1, -1), n_node_features, 0)
            node_attr_new = torch.cat((node_attr, yi), dim=1)
            edge_attr_de, node_attr_de, global_attr_de = self.decoder(edge_attr, node_attr_new, global_attr, edge_index,
                                                                      batch)
            edge_attr_op, node_attr_op, global_attr_op = self.output_transformer(edge_attr_de, node_attr_de,
                                                                                 global_attr_de, edge_index, batch)

            if len(output_ops) == 0:
                output_ops = node_attr_op
            else:
                output_ops = torch.cat((output_ops, node_attr_op), dim=1)
        pred = torch.min(output_ops, dim=0)[0]
        return pred


loss_func=nn.L1Loss()
def custom_loss(pred, data):
    return loss_func(pred, data.z)

# data parameters
args = parse_arguments()
n_objects   = args.n_object
data_folder = args.data_folder

# choose model
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
model = EncodeProcessDecode2(n_edge_feat_in=n_edge_in, n_edge_feat_out=n_edge_out,
                            n_node_feat_in=n_node_in, n_node_feat_out=n_node_out,
                            n_global_feat_in=n_global_in, n_global_feat_out=n_global_out,
                            mlp_latent_size=n_hidden[0], num_processing_steps=n_process,
                            process_weights_shared=weights_shared, full_output=full_output)
loss_funcs = [custom_loss]

# load data
train_data, test_data = get_data_loader(n_objects, data_folder, batch_size, eval_frac=eval_frac)

# train
train_sdf(model, train_data, test_data, loss_funcs, n_epochs=n_epochs, print_every=print_every,
          save_name=save_name, lr_0=lr_0, lr_scheduler_step_size=lr_step, lr_scheduler_gamma=lr_gamma)