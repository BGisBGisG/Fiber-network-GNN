import os

import torch
from torch.nn import ReLU, ELU, ModuleList, Sequential
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MessagePassing, BatchNorm, global_add_pool, global_mean_pool
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter
from torch_geometric.utils import degree

from utils import *

class CEALConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        aggregators,
        scalers,
        deg,
        edge_dim=None,
        towers=1,
        pre_layers=1,
        post_layers=1,
        divide_input=False,
        aggMLP=False,
        aggMLP_factor=0.5,
        **kwargs,
    ):

        super().__init__(node_dim=0, aggr=None)

        if divide_input:
            assert in_channels % towers == 0
        assert out_channels % towers == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregators = aggregators
        self.scalers = scalers
        self.edge_dim = edge_dim
        self.towers = towers
        self.divide_input = divide_input
        self.aggMLP = aggMLP
        self.aggMLP_factor = aggMLP_factor

        self.F_in = in_channels // towers if divide_input else in_channels
        self.F_out = self.out_channels // towers

        deg = deg.to(torch.float)

        self.avg_deg: Dict[str, float] = {
            "lin": deg.mean().item(),
            "log": (deg + 1).log().mean().item(),
            "exp": deg.exp().mean().item(),
        }

        if self.edge_dim is not None:
            self.edge_encoder = Linear(edge_dim, self.F_in)

        self.pre_nns = ModuleList()
        self.post_nns = ModuleList()
        for _ in range(towers):
            modules = [Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Linear(self.F_in, self.F_in)]
            self.pre_nns.append(Sequential(*modules))

            in_channels = (len(aggregators) * len(scalers) + 1) * self.F_in

            modules = [Linear(in_channels, self.F_out)]
            for _ in range(post_layers - 1):
                modules += [ReLU()]
                modules += [Linear(self.F_out, self.F_out)]
            self.post_nns.append(Sequential(*modules))

        self.lin = Linear(out_channels, out_channels)

        if self.aggMLP:
            factor = self.aggMLP_factor
            self.mlp_w0 = Sequential(
                torch.nn.Linear(self.F_in, round(self.F_in * factor)), ReLU(), torch.nn.Linear(round(self.F_in * factor), self.F_in)
            )
            self.mlp_w1 = Sequential(
                torch.nn.Linear(self.F_in, round(self.F_in * factor)), ReLU(), torch.nn.Linear(round(self.F_in * factor), self.F_in)
            )
            self.mlp_w2 = Sequential(
                torch.nn.Linear(self.F_in, round(self.F_in * factor)), ReLU(), torch.nn.Linear(round(self.F_in * factor), self.F_in)
            )
            self.mlp_w3 = Sequential(
                torch.nn.Linear(self.F_in, round(self.F_in * factor)), ReLU(), torch.nn.Linear(round(self.F_in * factor), self.F_in)
            )
            self.mlp_w4 = Sequential(
                torch.nn.Linear(self.F_in, round(self.F_in * factor)), ReLU(), torch.nn.Linear(round(self.F_in * factor), self.F_in)
            )

        else:

            if os.environ.get('FIXED_AGG_WEIGHTS', '0') == '1':
                self.register_buffer('agg_weights', torch.full((len(aggregators),), 1.0 / len(aggregators)))
            else:
                self.agg_weights = torch.nn.Parameter(torch.rand(len(aggregators)))

        self.reset_parameters()

    def reset_parameters(self):
        if self.edge_dim is not None:
            self.edge_encoder.reset_parameters()
        for layer in self.pre_nns:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for layer in self.post_nns:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        if self.aggMLP:
            self.mlp_w0[0].reset_parameters()

            self.mlp_w0[2].reset_parameters()

            self.mlp_w1[0].reset_parameters()

            self.mlp_w1[2].reset_parameters()

            self.mlp_w2[0].reset_parameters()

            self.mlp_w2[2].reset_parameters()

            self.mlp_w3[0].reset_parameters()

            self.mlp_w3[2].reset_parameters()

            self.mlp_w4[0].reset_parameters()

            self.mlp_w4[2].reset_parameters()

    def forward(self, x, edge_index, batch, edge_attr=None):

        if self.divide_input:
            x = x.view(-1, self.towers, self.F_in)
        else:
            x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)

        if self.aggMLP:
            out = self.propagate(edge_index, x=x, edge_attr=edge_attr, weights=None, batch=batch, size=None)
        else:
            myweights = self.agg_weights
            out = self.propagate(edge_index, x=x, edge_attr=edge_attr, weights=myweights, batch=batch, size=None)

        out = torch.cat([x, out], dim=-1)
        outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
        out = torch.cat(outs, dim=1)

        return self.lin(out)

    def message(self, x_i, x_j, edge_attr):

        h: Tensor = x_i
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
            edge_attr = edge_attr.view(-1, 1, self.F_in)
            edge_attr = edge_attr.repeat(1, self.towers, 1)
            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)

        hs = [nn(h[:, i]) for i, nn in enumerate(self.pre_nns)]
        return torch.stack(hs, dim=1)

    def aggregate(self, inputs, index, weights, batch, dim_size=None):
        outs = []

        if self.aggMLP:
            for aggregator in self.aggregators:
                if aggregator == "sum":
                    a = scatter(inputs, index, 0, None, dim_size, reduce="sum")
                    a_size = a.size()

                    w0 = self.mlp_w0(a[:, 0])
                    out = w0 * a[:, 0]
                    out = out.view(a_size[0], a_size[1], a_size[2])
                elif aggregator == "mean":
                    a = scatter(inputs, index, 0, None, dim_size, reduce="mean")
                    a_size = a.size()

                    w1 = self.mlp_w1(a[:, 0])
                    out = w1 * a[:, 0]
                    out = out.view(a_size[0], a_size[1], a_size[2])
                elif aggregator == "min":
                    a = scatter(inputs, index, 0, None, dim_size, reduce="min")
                    a_size = a.size()

                    w2 = self.mlp_w2(a[:, 0])
                    out = w2 * a[:, 0]
                    out = out.view(a_size[0], a_size[1], a_size[2])
                elif aggregator == "max":
                    a = scatter(inputs, index, 0, None, dim_size, reduce="max")
                    a_size = a.size()

                    w3 = self.mlp_w3(a[:, 0])
                    out = w3 * a[:, 0]
                    out = out.view(a_size[0], a_size[1], a_size[2])
                elif aggregator == "var" or aggregator == "std":
                    mean = scatter(inputs, index, 0, None, dim_size, reduce="mean")
                    mean_squares = scatter(inputs * inputs, index, 0, None, dim_size, reduce="mean")
                    a = mean_squares - mean * mean
                    a_size = a.size()

                    w4 = self.mlp_w4(a[:, 0])
                    out = w4 * a[:, 0]
                    out = out.view(a_size[0], a_size[1], a_size[2])

                    if aggregator == "std":
                        out = torch.sqrt(torch.relu(out) + 1e-5)
                else:
                    raise ValueError(f'Unknown aggregator "{aggregator}".')
                outs.append(out)
        else:
            for i, aggregator in enumerate(self.aggregators):
                if aggregator == "sum":
                    out = weights[i] * scatter(inputs, index, 0, None, dim_size, reduce="sum")
                elif aggregator == "mean":
                    out = weights[i] * scatter(inputs, index, 0, None, dim_size, reduce="mean")
                elif aggregator == "min":
                    out = weights[i] * scatter(inputs, index, 0, None, dim_size, reduce="min")
                elif aggregator == "max":
                    out = weights[i] * scatter(inputs, index, 0, None, dim_size, reduce="max")
                elif aggregator == "var" or aggregator == "std":
                    mean = scatter(inputs, index, 0, None, dim_size, reduce="mean")
                    mean_squares = scatter(inputs * inputs, index, 0, None, dim_size, reduce="mean")
                    out = weights[i] * (mean_squares - mean * mean)
                    if aggregator == "std":
                        out = torch.sqrt(torch.relu(out) + 1e-5)
                else:
                    raise ValueError(f'Unknown aggregator "{aggregator}".')
                outs.append(out)

        out = torch.cat(outs, dim=-1)

        deg = degree(index, dim_size, dtype=inputs.dtype)
        deg = deg.clamp_(1).view(-1, 1, 1)

        outs = []
        for scaler in self.scalers:
            if scaler == "identity":
                pass
            elif scaler == "amplification":
                out = out * (torch.log(deg + 1) / self.avg_deg["log"])
            elif scaler == "attenuation":
                out = out * (self.avg_deg["log"] / torch.log(deg + 1))
            elif scaler == "linear":
                out = out * (deg / self.avg_deg["lin"])
            elif scaler == "inverse_linear":
                out = out * (self.avg_deg["lin"] / deg)
            else:
                raise ValueError(f'Unknown scaler "{scaler}".')
            outs.append(out)

        return torch.cat(outs, dim=-1)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, " f"{self.out_channels}, towers={self.towers}, " f"edge_dim={self.edge_dim})"
