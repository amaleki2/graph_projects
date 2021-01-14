import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class GUNet(nn.Module):
    def __init__(self,
                 estimator,   # estimator core, GCN or GAT
                 params,      # number of hidden layers tuple of [in size, [h1, h2, ...], out size]
                 act=F.relu,  # activation function
                 with_edge_weight=False,
                 with_last_layer_skip_connection=True,   # skip the input to the last layer
                 **kwargs):

        super().__init__()
        self.act = act
        self.kwargs = kwargs
        self.params = params
        self.estimator = estimator
        self.with_last_layer_skip_connection = with_last_layer_skip_connection
        self.with_edge_weight = with_edge_weight
        self.estimators = torch.nn.ModuleList()
        self.build_estimators()  # setup all estimators
        self.reset_parameters()

    def build_estimators(self):
        in_channels, hidden_channels, out_channels = self.params
        estimator = self.estimator
        n_channels = len(hidden_channels)

        # this is for multi-headed GAT
        heads = 1 if "heads" not in self.kwargs else self.kwargs["heads"]

        self.estimators.append(estimator(in_channels, hidden_channels[0], **self.kwargs))
        for i in range(n_channels - 1):
            self.estimators.append(estimator(hidden_channels[i] * heads, hidden_channels[i], **self.kwargs))
            self.estimators.append(estimator(hidden_channels[i] * heads, hidden_channels[i + 1], **self.kwargs))

        for i in range(n_channels, 1, -1):
            self.estimators.append(estimator(hidden_channels[i - 1] * heads, hidden_channels[i - 1], **self.kwargs))
            self.estimators.append(estimator(hidden_channels[i - 1] * heads + hidden_channels[i - 2] * heads,
                                             hidden_channels[i - 2], **self.kwargs))

        self.estimators.append(estimator(hidden_channels[0] * heads, hidden_channels[0], **self.kwargs))

        # last layer should be one-headed
        tmp_kwargs = self.kwargs.copy()
        if "heads" in tmp_kwargs: tmp_kwargs["heads"] = 1
        last_layer_input_size = hidden_channels[0] * heads
        last_layer_input_size += in_channels if self.with_last_layer_skip_connection else 0
        self.estimators.append(estimator(last_layer_input_size, out_channels, **tmp_kwargs))

    def prep_estimator_input(self, data, x, x_skip=None):
        #  this function prepares input to estimator.
        #  data is the pytorch_geometric batch file that contains node and edge features
        #  x is the output of previous layer.
        #  x_skip is for the second part of network when skip connection are concatenated to output of previous layer
        x = data.x if x is None else x
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=-1)
        edge_index = data.edge_index
        if self.with_edge_weight:
            # edge weight can have only one feature.
            edge_weight = torch.norm(data.edge_attr[:, :2], dim=1)
            return x, edge_index, edge_weight
        else:
            return x, edge_index

    def reset_parameters(self):
        for estimator in self.estimators:
            estimator.reset_parameters()

    def forward(self, data):
        xvec = [data.x] if self.with_last_layer_skip_connection else []
        n_channels = len(self.params[1])
        x = None
        for i in range(n_channels):
            inputs = self.prep_estimator_input(data, x)
            x = self.estimators[2 * i](*inputs)
            x = self.act(x)
            inputs = self.prep_estimator_input(data, x)
            x = self.estimators[2 * i + 1](*inputs)
            x = self.act(x)
            xvec.append(x)

        xvec.pop()
        for i in range(n_channels, 2 * n_channels - 1):
            x_skip = xvec.pop()
            inputs = self.prep_estimator_input(data, x, x_skip=x_skip)
            x = self.estimators[2 * i](*inputs)
            x = self.act(x)
            inputs = self.prep_estimator_input(data, x)
            x = self.estimators[2 * i + 1](*inputs)
            x = self.act(x)
        if self.with_last_layer_skip_connection:
            x_skip = xvec.pop()
        else:
            x_skip = None
        inputs = self.prep_estimator_input(data, x, x_skip=x_skip)
        x = self.estimators[-1](*inputs)
        return x


class GATUNet(GUNet):
    def __init__(self, in_channels, hidden_channels, out_channels, with_last_layer_skip_connection=True,
                 heads=1, negative_slope=0.2):
        params = [in_channels, hidden_channels, out_channels]
        estimator = GATConv
        self.alpha_list = None
        super().__init__(estimator, params, with_last_layer_skip_connection=with_last_layer_skip_connection,
                         heads=heads, negative_slope=negative_slope)


class GCNUNet(GUNet):
    def __init__(self, in_channels, hidden_channels, out_channels, with_last_layer_skip_connection=True):
        params = [in_channels, hidden_channels, out_channels]
        estimator = GCNConv
        super().__init__(estimator, params, with_last_layer_skip_connection=with_last_layer_skip_connection)