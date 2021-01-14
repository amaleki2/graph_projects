import torch.nn
import torch
import torch.nn
from torch.nn import Sequential, ReLU, Linear, LayerNorm
from torch_scatter import scatter_mean, scatter_sum


#+++++++++++++++++++++++++#
#### helper functions #####
#+++++++++++++++++++++++++#
def get_edge_counts(edge_index, batch):
    return torch.bincount(batch[edge_index[0, :]])


def make_mlp_model(n_input, latent_size, n_output, activate_final=False, n_hidden_layers=2,
                   normalize=True, initializer=False):
    if latent_size is None:
        mlp = [Linear(n_input, n_output)]
    else:
        mlp = [Linear(n_input, latent_size),
               ReLU()]
        for n in range(n_hidden_layers):
            mlp.append(Linear(latent_size, latent_size))
            mlp.append(ReLU())
        mlp.append(Linear(latent_size, n_output))

    if activate_final:
        mlp.append(ReLU())
    if normalize and latent_size is not None:
        mlp.append(LayerNorm(n_output))
    mlp = Sequential(*mlp)

    # this is only for debugging
    if initializer:
        for layer in mlp:
            if layer._get_name() == "Linear":
                layer.weight.data.fill_(0.01)
                layer.bias.data.fill_(0.01)
    return mlp


def cast_globals_to_nodes(global_attr, batch=None, num_nodes=None):
    if batch is not None:
        _, counts = torch.unique(batch, return_counts=True)
        casted_global_attr = torch.cat([torch.repeat_interleave(global_attr[idx:idx+1, :], rep, dim=0)
                                        for idx, rep in enumerate(counts)], dim=0)
    else:
        assert global_attr.size(0) == 1, "batch numbers should be provided."
        assert num_nodes is not None, "number of nodes should be specified."
        casted_global_attr = torch.cat([global_attr] * num_nodes, dim=0)
    return casted_global_attr


def cast_globals_to_edges(global_attr, edge_index=None, batch=None, num_edges=None):
    if batch is not None:
        assert edge_index is not None, "edge index should be specified"
        edge_counts = get_edge_counts(edge_index, batch)
        casted_global_attr = torch.cat([torch.repeat_interleave(global_attr[idx:idx+1, :], rep, dim=0)
                                        for idx, rep in enumerate(edge_counts)], dim=0)
    else:
        assert global_attr.size(0) == 1, "batch numbers should be provided."
        assert num_edges is not None, "number of edges should be specified"
        casted_global_attr = torch.cat([global_attr] * num_edges, dim=0)
    return casted_global_attr


def cast_edges_to_globals(edge_attr, edge_index=None, batch=None, num_edges=None, num_globals=None):
    if batch is None:
        edge_attr_aggr = torch.sum(edge_attr, dim=0, keepdim=True)
    else:
        node_indices = torch.unique(batch)
        edge_counts = get_edge_counts(edge_index, batch)
        assert sum(edge_counts) == num_edges
        # indices = [idx.view(1, 1) for idx, count in zip(node_indices, edge_counts) for _ in range(count)]
        indices = [torch.repeat_interleave(idx, count) for idx, count in zip(node_indices, edge_counts)]
        indices = torch.cat(indices)
        edge_attr_aggr = scatter_sum(edge_attr, index=indices, dim=0, dim_size=num_globals)
    return edge_attr_aggr


def cast_nodes_to_globals(node_attr, batch=None, num_globals=None):
    if batch is None:
        x_aggr = torch.sum(node_attr, dim=0, keepdim=True)
    else:
        x_aggr = scatter_sum(node_attr, index=batch, dim=0, dim_size=num_globals)
    return x_aggr


def cast_edges_to_nodes(edge_attr, indices, num_nodes=None):
    edge_attr_aggr = scatter_sum(edge_attr, indices, dim=0, dim_size=num_nodes)
    return edge_attr_aggr

#+++++++++++++++++++++++++#
## block models: simple ###
#+++++++++++++++++++++++++#
class IndependentEdgeModel(torch.nn.Module):
    def __init__(self,
                 n_edge_feats_in,  # number of input edge features
                 n_edge_feats_out,  # number of output edge features
                 latent_size=128,  # latent size of mlp
                 activate_final=True,  # use activate for the last layer or not?
                 normalize=True  # batch normalize the output
                 ):
        super(IndependentEdgeModel, self).__init__()
        self.params = [n_edge_feats_in, n_edge_feats_out]  # useful for debugging
        self.edge_mlp = make_mlp_model(n_edge_feats_in,
                                       latent_size,
                                       n_edge_feats_out,
                                       activate_final=activate_final,
                                       normalize=normalize)

    def forward(self, edge_attr):
        return self.edge_mlp(edge_attr)


class IndependentNodeModel(torch.nn.Module):
    def __init__(self,
                 n_node_feats_in,    # number of input node features
                 n_node_feats_out,   # number of output node features
                 latent_size=128,    # latent size of mlp
                 activate_final=True,  # use activate for the last layer or not?
                 normalize=True         # batch normalize the output
                 ):
        super(IndependentNodeModel, self).__init__()
        self.params = [n_node_feats_in, n_node_feats_out]
        self.node_mlp = make_mlp_model(n_node_feats_in,
                                       latent_size,
                                       n_node_feats_out,
                                       activate_final=activate_final,
                                       normalize=normalize)

    def forward(self, node_attr):
        return self.node_mlp(node_attr)


class IndependentGlobalModel(torch.nn.Module):
    def __init__(self,
                 n_global_in,
                 n_global_out,
                 latent_size=128,
                 activate_final=True,
                 normalize=True
                 ):

        super(IndependentGlobalModel, self).__init__()
        self.params = [n_global_in, n_global_out]
        self.global_mlp = make_mlp_model(n_global_in,
                                         latent_size,
                                         n_global_out,
                                         activate_final=activate_final,
                                         normalize=normalize  # careful: batch normalization does not work when
                                                              # batch size = 1;
                                                              # https://github.com/pytorch/pytorch/issues/7716
                                         )

    def forward(self, global_attr):
        return self.global_mlp(global_attr)


#+++++++++++++++++++++++++#
## block models: complex ##
#+++++++++++++++++++++++++#
class EdgeModel(torch.nn.Module):
    def __init__(self,
                 n_edge_feats_in,    # number of input edge features
                 n_edge_feats_out,   # number of output edge features
                 n_node_feats,       # number of input node features
                 n_global_feats,     # number of global (graph) features
                 latent_size=128,    # latent size of mlp
                 activate_final=True,  # use activate for the last layer or not?
                 normalize=True         # batch normalize the output
                 ):
        super(EdgeModel, self).__init__()
        self.params = [n_edge_feats_in, n_edge_feats_out, n_node_feats, n_global_feats]  # useful for debugging
        self.edge_mlp = make_mlp_model(n_edge_feats_in + n_node_feats * 2 + n_global_feats,
                                       latent_size,
                                       n_edge_feats_out,
                                       activate_final=activate_final,
                                       normalize=normalize)

    def forward(self, receiver, sender, edge_attr, global_attr):
        out = torch.cat([receiver, sender, edge_attr, global_attr], dim=1)
        return self.edge_mlp(out)


class NodeModel(torch.nn.Module):
    def __init__(self,
                 n_node_feats_in,    # number of input node features
                 n_node_feats_out,  # number of output node features
                 n_edge_feats,     # number of input edge features
                 n_global_feats,   # number of global (graph) features
                 latent_size=128,  # latent size of mlp
                 activate_final=True,  # use activate for the last layer or not?
                 agg_func=scatter_sum,  # function to aggregation edges to nodes
                 normalize=True,        # batch normalize the output
                 senders_turned_off=True  # don't aggregate senders
                 ):
        super(NodeModel, self).__init__()
        self.agg_func = agg_func
        self.senders_turned_off = senders_turned_off
        self.params = [n_node_feats_in, n_node_feats_out, n_edge_feats, n_global_feats]
        scalar = 1 if self.senders_turned_off else 2
        self.node_mlp = make_mlp_model(n_node_feats_in + n_edge_feats * scalar + n_global_feats,
                                       latent_size,
                                       n_node_feats_out,
                                       activate_final=activate_final,
                                       normalize=normalize)

    def forward(self, x, global_attr, recv_edge_attr_agg, send_edge_attr_agg):
        if self.senders_turned_off:
            out = torch.cat([x, recv_edge_attr_agg, global_attr], dim=1)
        else:
            out = torch.cat([x, send_edge_attr_agg, recv_edge_attr_agg, global_attr], dim=1)
        return self.node_mlp(out)


class GlobalModel(torch.nn.Module):
    def __init__(self,
                 n_global_in,
                 n_global_out,
                 n_node_feats,
                 n_edge_feats,
                 latent_size=128,
                 activate_final=True,
                 normalize=True,
                 ):

        super(GlobalModel, self).__init__()
        self.params = [n_global_in, n_global_out, n_node_feats, n_edge_feats]
        self.global_mlp = make_mlp_model(n_global_in + n_edge_feats + n_node_feats,
                                         latent_size,
                                         n_global_out,
                                         activate_final=activate_final,
                                         normalize=normalize  # careful: batch normalization does not work when
                                                              # batch size = 1;
                                                              # https://github.com/pytorch/pytorch/issues/7716
                                         )

    def forward(self, x_aggr, edge_attr_aggr, global_attr):
        out = torch.cat([x_aggr, edge_attr_aggr, global_attr], dim=1)
        return self.global_mlp(out)


class GraphNetworkBlock(torch.nn.Module):
    def __init__(self,
                 n_edge_feat_in, n_edge_feat_out,
                 n_node_feat_in, n_node_feat_out,
                 n_global_feat_in, n_global_feat_out,
                 latent_size=128,
                 activate_final=True,
                 normalize=True):
        super(GraphNetworkBlock, self).__init__()
        self.params = [n_edge_feat_in, n_edge_feat_out,
                       n_node_feat_in, n_node_feat_out,
                       n_global_feat_in, n_global_feat_out]

        if n_edge_feat_out is not None:
            self.edge_model = EdgeModel(n_edge_feat_in, n_edge_feat_out, n_node_feat_in, n_global_feat_in,
                                        latent_size=latent_size, activate_final=activate_final, normalize=normalize)
        if n_node_feat_out is not None:
            self.node_model = NodeModel(n_node_feat_in, n_node_feat_out, n_edge_feat_out, n_global_feat_in,
                                        latent_size=latent_size, activate_final=activate_final, normalize=normalize)
        if n_global_feat_out is not None:
            self.global_model = GlobalModel(n_global_feat_in, n_global_feat_out, n_node_feat_out, n_edge_feat_out,
                                            latent_size=latent_size, activate_final=activate_final, normalize=normalize)
        self.reset_parameters()

    def reset_parameters(self):
        for item in ['node_model', 'edge_model', 'global_model']:
            if hasattr(self, item):
                model = getattr(self, item)
                if hasattr(model, 'reset_parameters'):
                    model.reset_parameters()

    def forward(self, edge_attr, node_attr, global_attr, edge_index, batch):
        [_, n_edge_feat_out,
         _, n_node_feat_out,
         _, n_global_feat_out] = self.params
        row, col = edge_index
        num_edges, num_nodes, num_globals = edge_attr.size(0), node_attr.size(0), global_attr.size(0)

        # update edge attr
        if n_edge_feat_out is None:
            edge_attr_new = edge_attr
        else:
            sender_attr, receiver_attr = node_attr[row, :], node_attr[col, :]
            global_attr_to_edge = cast_globals_to_edges(global_attr, edge_index=edge_index, batch=batch, num_edges=num_edges)
            edge_attr_new = self.edge_model(receiver_attr, sender_attr, edge_attr, global_attr_to_edge)

        # update node attr
        if n_node_feat_out is None:
            node_attr_new = node_attr
        else:
            global_attr_to_nodes = cast_globals_to_nodes(global_attr, batch=batch, num_nodes=num_nodes)
            sender_attr_to_node = cast_edges_to_nodes(edge_attr_new, row, num_nodes=num_nodes)
            receiver_attr_to_node = cast_edges_to_nodes(edge_attr_new, col, num_nodes=num_nodes)
            node_attr_new = self.node_model(node_attr, global_attr_to_nodes, receiver_attr_to_node, sender_attr_to_node)

        # update global attr
        if n_global_feat_out is None:
            global_attr_new = global_attr
        else:
            node_attr_to_global = cast_nodes_to_globals(node_attr_new, batch=batch, num_globals=num_globals)
            edge_attr_to_global = cast_edges_to_globals(edge_attr_new, edge_index=edge_index, batch=batch,
                                                        num_edges=num_edges, num_globals=num_globals)
            global_attr_new = self.global_model(node_attr_to_global, edge_attr_to_global, global_attr)
        return edge_attr_new, node_attr_new, global_attr_new


class GraphNetworkIndependentBlock(torch.nn.Module):
    def __init__(self,
                 n_edge_feat_in, n_edge_feat_out,
                 n_node_feat_in, n_node_feat_out,
                 n_global_feat_in, n_global_feat_out,
                 latent_size=128,
                 activate_final=True,
                 normalize=True):
        super(GraphNetworkIndependentBlock, self).__init__()
        if n_edge_feat_out is None:
            self.edge_model = lambda xx: xx
        else:
            self.edge_model = IndependentEdgeModel(n_edge_feat_in, n_edge_feat_out, latent_size=latent_size,
                                                   activate_final=activate_final, normalize=normalize)
        if n_node_feat_out is None:
            self.node_model = lambda xx: xx
        else:
            self.node_model = IndependentNodeModel(n_node_feat_in, n_node_feat_out, latent_size=latent_size,
                                                   activate_final=activate_final, normalize=normalize)
        if n_global_feat_out is None:
            self.global_model = lambda xx: xx
        else:
            self.global_model = IndependentGlobalModel(n_global_feat_in, n_global_feat_out, latent_size=latent_size,
                                                       activate_final=activate_final, normalize=normalize)
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, edge_attr, node_attr, global_attr, edge_index, batch):
        # signature should be similar to GraphNetworkBlock
        return self.edge_model(edge_attr), self.node_model(node_attr), self.global_model(global_attr)


class EncodeProcessDecode(torch.nn.Module):
    def __init__(self,
                 n_edge_feat_in=1, n_node_feat_in=1, n_global_feat_in=1,
                 n_edge_feat_out=1, n_node_feat_out=1, n_global_feat_out=1,
                 encoder=GraphNetworkIndependentBlock, processor=GraphNetworkBlock,
                 decoder=GraphNetworkIndependentBlock, output_transformer=GraphNetworkIndependentBlock,
                 mlp_latent_size=128, num_processing_steps=5,
                 process_weights_shared=False, normalize=True, full_output=False):
        super(EncodeProcessDecode, self).__init__()
        assert not (n_edge_feat_in is None or n_node_feat_in is None or n_global_feat_in is None), \
            "input sizes should be specified"
        self.num_processing_steps = num_processing_steps
        self.full_output = full_output
        self.process_weights_shared = process_weights_shared
        self.encoder = encoder(n_edge_feat_in, mlp_latent_size,
                               n_node_feat_in, mlp_latent_size,
                               n_global_feat_in, mlp_latent_size,
                               latent_size=mlp_latent_size,
                               activate_final=True,
                               normalize=normalize)
        if self.process_weights_shared:
            self.processors = torch.nn.ModuleList()
            for _ in range(num_processing_steps):
                self.processors.append(processor(2 * mlp_latent_size, mlp_latent_size,
                                                 2 * mlp_latent_size, mlp_latent_size,
                                                 2 * mlp_latent_size, mlp_latent_size,
                                                 latent_size=mlp_latent_size,
                                                 activate_final=True,
                                                 normalize=normalize))
        else:
            self.processor = processor(2 * mlp_latent_size, mlp_latent_size,
                                       2 * mlp_latent_size, mlp_latent_size,
                                       2 * mlp_latent_size, mlp_latent_size,
                                       latent_size=mlp_latent_size,
                                       activate_final=True,
                                       normalize=normalize)

        self.decoder = decoder(mlp_latent_size, mlp_latent_size,
                               mlp_latent_size, mlp_latent_size,
                               mlp_latent_size, mlp_latent_size,
                               latent_size=mlp_latent_size,
                               activate_final=True,
                               normalize=normalize)

        self.output_transformer = output_transformer(mlp_latent_size, n_edge_feat_out,
                                                     mlp_latent_size, n_node_feat_out,
                                                     mlp_latent_size, n_global_feat_out,
                                                     latent_size=None,
                                                     activate_final=False, normalize=False)

    def forward(self, data):
        edge_attr, edge_index, node_attr, global_attr, batch = data.edge_attr, data.edge_index, data.x, data.u, data.batch
        edge_attr, node_attr, global_attr = self.encoder(edge_attr, node_attr, global_attr, edge_index, batch)
        edge_attr0, node_attr0, global_attr0 = edge_attr.clone(), node_attr.clone(), global_attr.clone()
        output_ops = []
        for i in range(self.num_processing_steps):
            edge_attr = torch.cat((edge_attr0, edge_attr), dim=1)
            node_attr = torch.cat((node_attr0, node_attr), dim=1)
            global_attr = torch.cat((global_attr0, global_attr), dim=1)
            if self.process_weights_shared:
                edge_attr, node_attr, global_attr = self.processor[i](edge_attr, node_attr, global_attr, edge_index, batch)
            else:
                edge_attr, node_attr, global_attr = self.processor(edge_attr, node_attr, global_attr, edge_index, batch)
            edge_attr_de, node_attr_de, global_attr_de = self.decoder(edge_attr, node_attr, global_attr, edge_index, batch)
            edge_attr_op, node_attr_op, global_attr_op = self.output_transformer(edge_attr_de, node_attr_de, global_attr_de, edge_index, batch)
            output_ops.append((edge_attr_op, node_attr_op, global_attr_op))

        if self.full_output:
            return output_ops
        else:
            return output_ops[-1]