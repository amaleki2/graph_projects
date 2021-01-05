import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv


class GUNet(nn.Module):
    def __init__(self, estimator, params, requires_edge_weight=True, act=F.relu, with_alternate_max_aggr=False, **kwargs):
        super().__init__()
        self.act = act
        self.estimator = estimator
        self.params = params
        self.kwargs = kwargs
        self.requires_edge_weight = requires_edge_weight
        self.with_alternate_max_aggr = with_alternate_max_aggr
        self.estimators = torch.nn.ModuleList()
        self.build_estimators()
        self.reset_parameters()

    def build_estimators(self):
        in_channels, hidden_channels, out_channels = self.params
        estimator = self.estimator
        n_channels = len(hidden_channels)
        self.estimators.append(estimator(in_channels, hidden_channels[0], **self.kwargs))
        for i in range(n_channels - 1):
            self.estimators.append(estimator(hidden_channels[i], hidden_channels[i], **self.kwargs))
            self.estimators.append(estimator(hidden_channels[i], hidden_channels[i + 1], **self.kwargs))

        for i in range(n_channels, 1, -1):
            self.estimators.append(estimator(hidden_channels[i - 1], hidden_channels[i - 1], **self.kwargs))
            self.estimators.append(estimator(hidden_channels[i - 1] + hidden_channels[i - 2], hidden_channels[i - 2], **self.kwargs))

        self.estimators.append(estimator(hidden_channels[0], hidden_channels[0], **self.kwargs))
        self.estimators.append(estimator(hidden_channels[0], out_channels, **self.kwargs))

    def prep_estimator_input(self, data, x=None, x_concat=None):
        if x is None:
            x = data.x
        edge_idx = data.edge_index
        if x_concat is not None:
            x = torch.cat([x, x_concat], dim=-1)
        if self.requires_edge_weight:
            edge_weight = data.edge_attr
            edge_weight = edge_weight / edge_weight.max()
            return x, edge_idx, edge_weight
        else:
            return x, edge_idx

    def reset_parameters(self):
        for estimator in self.estimators:
            estimator.reset_parameters()

    def forward(self, data):
        xvec = []
        n_channels = len(self.params[1])
        x = None
        for i in range(n_channels):
            inputs = self.prep_estimator_input(data, x=x)
            x = self.estimators[2 * i](*inputs)
            x = self.act(x)
            inputs = self.prep_estimator_input(data, x=x)
            x = self.estimators[2 * i + 1](*inputs)
            x = self.act(x)
            xvec.append(x)

        xvec.pop()
        for i in range(n_channels, 2 * n_channels - 1):
            y = xvec.pop()
            inputs = self.prep_estimator_input(data, x=x, x_concat=y)
            x = self.estimators[2 * i](*inputs)
            x = self.act(x)
            inputs = self.prep_estimator_input(data, x=x)
            x = self.estimators[2 * i + 1](*inputs)
            x = self.act(x)

        inputs = self.prep_estimator_input(data, x=x)
        x = self.estimators[-1](*inputs)
        return x


class GatUNet(GUNet):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, code_dropout=0, negative_slope=0.2):
        params = [in_channels, hidden_channels, out_channels]
        estimator = GATConv
        self.heads = heads
        self.code_dropout = code_dropout
        self.negative_slope = negative_slope
        self.alpha_list = None
        super().__init__(estimator, params, requires_edge_weight=False)

    def build_estimators(self):
        in_channels, hidden_channels, out_channels = self.params
        estimator = self.estimator
        heads = self.heads
        negative_slope = self.negative_slope
        n_channels = len(hidden_channels)
        dropouts = np.linspace(0, self.code_dropout, n_channels)
        self.estimators.append(estimator(in_channels, hidden_channels[0], heads=heads, negative_slope=negative_slope))
        for i in range(n_channels - 1):
            dropout = dropouts[i+1]
            self.estimators.append(estimator(hidden_channels[i] * heads, hidden_channels[i], heads=heads,
                                             dropout=dropout, negative_slope=negative_slope))
            self.estimators.append(estimator(hidden_channels[i] * heads, hidden_channels[i + 1], heads=heads,
                                             dropout=dropout, negative_slope=negative_slope))

        for i in range(n_channels, 1, -1):
            dropout = dropouts[i-1]
            self.estimators.append(estimator(hidden_channels[i - 1] * heads, hidden_channels[i - 1], heads=heads,
                                             dropout=dropout, negative_slope=negative_slope))
            self.estimators.append(estimator(hidden_channels[i - 1] * heads + hidden_channels[i - 2] * heads,
                                             hidden_channels[i - 2], heads=heads, dropout=dropout,
                                             negative_slope=negative_slope))

        self.estimators.append(estimator(hidden_channels[0] * heads, hidden_channels[0],
                                         heads=heads, negative_slope=negative_slope))
        self.estimators.append(estimator(hidden_channels[0] * heads, out_channels, negative_slope=negative_slope))  # last layer is one-headed.

    def forward(self, data):
        xvec = []
        self.alpha_list = []
        n_channels = len(self.params[1])
        x = None
        for i in range(n_channels):
            inputs = self.prep_estimator_input(data, x=x)
            x, alpha = self.estimators[2 * i](*inputs, return_attention_weights=True)
            self.alpha_list.append(alpha)
            x = self.act(x)
            inputs = self.prep_estimator_input(data, x=x)
            x, alpha = self.estimators[2 * i + 1](*inputs, return_attention_weights=True)
            self.alpha_list.append(alpha)
            x = self.act(x)
            xvec.append(x)

        xvec.pop()
        for i in range(n_channels, 2 * n_channels - 1):
            y = xvec.pop()
            inputs = self.prep_estimator_input(data, x=x, x_concat=y)
            x, alpha = self.estimators[2 * i](*inputs, return_attention_weights=True)
            self.alpha_list.append(alpha)
            x = self.act(x)
            inputs = self.prep_estimator_input(data, x=x)
            x, alpha = self.estimators[2 * i + 1](*inputs, return_attention_weights=True)
            self.alpha_list.append(alpha)
            x = self.act(x)

        inputs = self.prep_estimator_input(data, x=x)
        x, alpha = self.estimators[-1](*inputs, return_attention_weights=True)
        self.alpha_list.append(alpha)
        return x


class GatUNet2(GatUNet):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, code_dropout=0, negative_slope=0.2):
        super().__init__(in_channels, hidden_channels, out_channels, heads=heads,
                         code_dropout=code_dropout, negative_slope=negative_slope)

    def build_estimators(self):
        in_channels, hidden_channels, out_channels = self.params
        estimator = self.estimator
        heads = self.heads
        negative_slope = self.negative_slope
        n_channels = len(hidden_channels)
        dropouts = np.linspace(0, self.code_dropout, n_channels)
        self.estimators.append(estimator(in_channels, hidden_channels[0], heads=heads, negative_slope=negative_slope))
        for i in range(n_channels - 1):
            dropout = dropouts[i+1]
            self.estimators.append(estimator(hidden_channels[i] * heads, hidden_channels[i], heads=heads,
                                             dropout=dropout, negative_slope=negative_slope))
            self.estimators.append(estimator(hidden_channels[i] * heads, hidden_channels[i + 1], heads=heads,
                                             dropout=dropout, negative_slope=negative_slope))

        for i in range(n_channels, 1, -1):
            dropout = dropouts[i-1]
            self.estimators.append(estimator(hidden_channels[i - 1] * heads, hidden_channels[i - 1], heads=heads,
                                             dropout=dropout, negative_slope=negative_slope))
            self.estimators.append(estimator(hidden_channels[i - 1] * heads + hidden_channels[i - 2] * heads,
                                             hidden_channels[i - 2], heads=heads, dropout=dropout,
                                             negative_slope=negative_slope))

        self.estimators.append(estimator(hidden_channels[0] * heads, hidden_channels[0],
                                         heads=heads, negative_slope=negative_slope))
        self.estimators.append(estimator(hidden_channels[0] * heads + in_channels, out_channels,
                                         heads=1, negative_slope=negative_slope))  # last layer is one-headed.

    def forward(self, data):
        xvec = [data.x]
        self.alpha_list = []
        n_channels = len(self.params[1])
        x = None
        for i in range(n_channels):
            inputs = self.prep_estimator_input(data, x=x)
            x, alpha = self.estimators[2 * i](*inputs, return_attention_weights=True)
            self.alpha_list.append(alpha)
            x = self.act(x)
            inputs = self.prep_estimator_input(data, x=x)
            x, alpha = self.estimators[2 * i + 1](*inputs, return_attention_weights=True)
            self.alpha_list.append(alpha)
            x = self.act(x)
            xvec.append(x)

        xvec.pop()
        for i in range(n_channels, 2 * n_channels - 1):
            y = xvec.pop()
            inputs = self.prep_estimator_input(data, x=x, x_concat=y)
            x, alpha = self.estimators[2 * i](*inputs, return_attention_weights=True)
            self.alpha_list.append(alpha)
            x = self.act(x)
            inputs = self.prep_estimator_input(data, x=x)
            x, alpha = self.estimators[2 * i + 1](*inputs, return_attention_weights=True)
            self.alpha_list.append(alpha)
            x = self.act(x)

        y = xvec.pop()
        inputs = self.prep_estimator_input(data, x=x, x_concat=y)
        x, alpha = self.estimators[-1](*inputs, return_attention_weights=True)
        self.alpha_list.append(alpha)
        return x
