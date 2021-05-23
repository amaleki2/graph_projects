import torch.nn
import torch
import torch.nn
from torch_scatter import scatter_sum
from src.graph_models_deepmind import (make_mlp_model, cast_edges_to_nodes,
                                       IndependentEdgeModel, IndependentNodeModel)

#+++++++++++++++++++++++++#
## block models: complex ##
#+++++++++++++++++++++++++#
class EdgeModel(torch.nn.Module):
    def __init__(self,
                 n_edge_feats_in,    # number of input edge features
                 n_edge_feats_out,   # number of output edge features
                 n_node_feats,       # number of input node features
                 latent_sizes=128,    # latent size of mlp
                 activate_final=True,  # use activate for the last layer or not?
                 normalize=True         # batch normalize the output
                 ):
        super(EdgeModel, self).__init__()
        self.params = [n_edge_feats_in, n_edge_feats_out, n_node_feats]  # useful for debugging
        self.edge_mlp = make_mlp_model(n_edge_feats_in + n_node_feats * 2,
                                       latent_sizes,
                                       n_edge_feats_out,
                                       activate_final=activate_final,
                                       normalize=normalize)

    def forward(self, receiver, sender, edge_attr):
        out = torch.cat([receiver, sender, edge_attr], dim=1)
        return self.edge_mlp(out)


class NodeModel(torch.nn.Module):
    def __init__(self,
                 n_node_feats_in,    # number of input node features
                 n_node_feats_out,  # number of output node features
                 n_edge_feats,     # number of input edge features
                 latent_sizes=128,  # latent size of mlp
                 activate_final=True,  # use activate for the last layer or not?
                 agg_func=scatter_sum,  # function to aggregation edges to nodes
                 normalize=True,        # batch normalize the output
                 senders_turned_off=True  # don't aggregate senders
                 ):
        super(NodeModel, self).__init__()
        self.agg_func = agg_func
        self.senders_turned_off = senders_turned_off
        self.params = [n_node_feats_in, n_node_feats_out, n_edge_feats]
        scalar = 1 if self.senders_turned_off else 2
        self.node_mlp = make_mlp_model(n_node_feats_in + n_edge_feats * scalar,
                                       latent_sizes,
                                       n_node_feats_out,
                                       activate_final=activate_final,
                                       normalize=normalize)

    def forward(self, x, recv_edge_attr_agg, send_edge_attr_agg):
        if self.senders_turned_off:
            out = torch.cat([x, recv_edge_attr_agg], dim=1)
        else:
            out = torch.cat([x, send_edge_attr_agg, recv_edge_attr_agg], dim=1)
        return self.node_mlp(out)


class GraphNetworkBlock(torch.nn.Module):
    def __init__(self,
                 n_edge_feat_in, n_edge_feat_out,
                 n_node_feat_in, n_node_feat_out,
                 latent_sizes=128,
                 activate_final=True,
                 normalize=True):
        super(GraphNetworkBlock, self).__init__()
        self.params = [n_edge_feat_in, n_edge_feat_out,
                       n_node_feat_in, n_node_feat_out]

        if n_edge_feat_out is not None:
            self.edge_model = EdgeModel(n_edge_feat_in, n_edge_feat_out, n_node_feat_in,
                                        latent_sizes=latent_sizes, activate_final=activate_final,
                                        normalize=normalize)
        if n_node_feat_out is not None:
            self.node_model = NodeModel(n_node_feat_in, n_node_feat_out, n_edge_feat_out,
                                        latent_sizes=latent_sizes, activate_final=activate_final,
                                        normalize=normalize)
        self.reset_parameters()

    def reset_parameters(self):
        for item in ['node_model', 'edge_model']:
            if hasattr(self, item):
                model = getattr(self, item)
                if hasattr(model, 'reset_parameters'):
                    model.reset_parameters()

    def forward(self, edge_attr, node_attr, edge_index, batch):
        [_, n_edge_feat_out,
         _, n_node_feat_out] = self.params
        row, col = edge_index
        num_edges, num_nodes = edge_attr.size(0), node_attr.size(0)

        # update edge attr
        if n_edge_feat_out is None:
            edge_attr_new = edge_attr
        else:
            sender_attr, receiver_attr = node_attr[row, :], node_attr[col, :]
            edge_attr_new = self.edge_model(receiver_attr, sender_attr, edge_attr)

        # update node attr
        if n_node_feat_out is None:
            node_attr_new = node_attr
        else:
            sender_attr_to_node = cast_edges_to_nodes(edge_attr_new, row, num_nodes=num_nodes)
            receiver_attr_to_node = cast_edges_to_nodes(edge_attr_new, col, num_nodes=num_nodes)
            node_attr_new = self.node_model(node_attr, receiver_attr_to_node, sender_attr_to_node)

        return edge_attr_new, node_attr_new


class GraphNetworkIndependentBlock(torch.nn.Module):
    def __init__(self,
                 n_edge_feat_in, n_edge_feat_out,
                 n_node_feat_in, n_node_feat_out,
                 latent_sizes=128,
                 activate_final=True,
                 normalize=True):
        super(GraphNetworkIndependentBlock, self).__init__()
        if n_edge_feat_out is None:
            self.edge_model = lambda xx: xx
        else:
            self.edge_model = IndependentEdgeModel(n_edge_feat_in, n_edge_feat_out, latent_sizes=latent_sizes,
                                                   activate_final=activate_final, normalize=normalize)
        if n_node_feat_out is None:
            self.node_model = lambda xx: xx
        else:
            self.node_model = IndependentNodeModel(n_node_feat_in, n_node_feat_out, latent_sizes=latent_sizes,
                                                   activate_final=activate_final, normalize=normalize)
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, edge_attr, node_attr, edge_index, batch):
        # signature should be similar to GraphNetworkBlock
        return self.edge_model(edge_attr), self.node_model(node_attr)


class EncodeProcessDecode(torch.nn.Module):
    def __init__(self,
                 n_edge_feat_in=1, n_node_feat_in=1,
                 n_edge_feat_out=1, n_node_feat_out=1,
                 encoder=GraphNetworkIndependentBlock, processor=GraphNetworkBlock,
                 decoder=GraphNetworkIndependentBlock, output_transformer=GraphNetworkIndependentBlock,
                 mlp_latent_size=128, num_processing_steps=5,
                 process_weights_shared=False, normalize=True, full_output=False):
        super(EncodeProcessDecode, self).__init__()
        assert not (n_edge_feat_in is None or n_node_feat_in is None), \
            "input sizes should be specified"
        self.num_processing_steps = num_processing_steps
        self.full_output = full_output
        self.process_weights_shared = process_weights_shared
        self.encoder = encoder(n_edge_feat_in, mlp_latent_size,
                               n_node_feat_in, mlp_latent_size,
                               latent_sizes=mlp_latent_size,
                               activate_final=True,
                               normalize=normalize)
        if not self.process_weights_shared:
            self.processors = torch.nn.ModuleList()
            for _ in range(num_processing_steps):
                self.processors.append(processor(2 * mlp_latent_size, mlp_latent_size,
                                                 2 * mlp_latent_size, mlp_latent_size,
                                                 latent_sizes=mlp_latent_size,
                                                 activate_final=True,
                                                 normalize=normalize))
        else:
            self.processors = processor(2 * mlp_latent_size, mlp_latent_size,
                                        2 * mlp_latent_size, mlp_latent_size,
                                        latent_sizes=mlp_latent_size,
                                        activate_final=True,
                                        normalize=normalize)

        self.decoder = decoder(mlp_latent_size, mlp_latent_size,
                               mlp_latent_size, mlp_latent_size,
                               latent_sizes=mlp_latent_size,
                               activate_final=True,
                               normalize=normalize)

        self.output_transformer = output_transformer(mlp_latent_size, n_edge_feat_out,
                                                     mlp_latent_size, n_node_feat_out,
                                                     latent_sizes=None,
                                                     activate_final=False, normalize=False)

    def forward(self, data):
        edge_attr, edge_index, node_attr, batch = data.edge_attr, data.edge_index, data.x, data.batch
        edge_attr, node_attr = self.encoder(edge_attr, node_attr, edge_index, batch)
        edge_attr0, node_attr0 = edge_attr.clone(), node_attr.clone()
        output_ops = []
        for i in range(self.num_processing_steps):
            edge_attr = torch.cat((edge_attr0, edge_attr), dim=1)
            node_attr = torch.cat((node_attr0, node_attr), dim=1)
            if not self.process_weights_shared:
                edge_attr, node_attr = self.processors[i](edge_attr, node_attr, edge_index, batch)
            else:
                edge_attr, node_attr = self.processors(edge_attr, node_attr, edge_index, batch)
            edge_attr_de, node_attr_de = self.decoder(edge_attr, node_attr, edge_index, batch)
            edge_attr_op, node_attr_op = self.output_transformer(edge_attr_de, node_attr_de, edge_index, batch)
            output_ops.append((edge_attr_op, node_attr_op))

        if self.full_output:
            return output_ops
        else:
            return output_ops[-1]