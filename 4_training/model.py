import math
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GCNConv, PNAConv, BatchNorm, global_add_pool, global_mean_pool
from tqdm import tqdm
import os.path as osp
from sklearn.metrics import r2_score

from ceal import *
from utils import *
from dataset import MyDataset
from config import *

class MyCEALNetwork(torch.nn.Module):

   def __init__(self,
                in_channels,
                out_channels,
                aggregators,
                scalers,
                deg,
                numLayers = 2,
                edge_dim = None,
                towers = 1,
                pre_layers = 1,
                post_layers = 1,
                divide_input = False,
                aggMLP = True,
                dropout = 0.3
               ):

      super().__init__()

      self.in_channels = in_channels
      self.out_channels = out_channels
      self.aggregators = aggregators
      self.scalers = scalers
      self.deg = deg
      self.numLayers = numLayers
      self.edge_dim = edge_dim
      self.towers = towers
      self.pre_layers = pre_layers
      self.post_layers = post_layers
      self.divide_input = divide_input
      self.aggMLP = aggMLP
      self.dropout = dropout

      self.ceal_1 = CEALConv( in_channels = in_channels,
                              out_channels = out_channels,
                              aggregators = aggregators,
                              scalers = scalers,
                              deg = deg,
                              edge_dim = edge_dim,
                              towers = 1,
                              pre_layers = 1,
                              post_layers = 1,
                              divide_input = False,
                              aggMLP = aggMLP,
                              aggMLP_factor=0.5,
                            )
      self.batch_norm_1 = BatchNorm(self.out_channels)

      self.cealConvs = ModuleList()
      self.batch_norms = ModuleList()
      for _ in range(1, self.numLayers):
         cealconv = CEALConv( in_channels = out_channels,
                              out_channels = out_channels,
                              aggregators = aggregators,
                              scalers = scalers,
                              deg = deg,
                              edge_dim = edge_dim,
                              towers = 1,
                              pre_layers = 1,
                              post_layers = 1,
                              divide_input = False,
                              aggMLP = aggMLP,
                              aggMLP_factor=0.5,
                            )
         batch_norm = BatchNorm(self.out_channels)
         self.cealConvs.append(cealconv)
         self.batch_norms.append(batch_norm)

      self.pre_mlp = Sequential(Linear(in_channels, out_channels//2), ReLU(), Linear(out_channels//2, in_channels))
      self.post_mlp = Sequential(Linear(self.out_channels, self.out_channels//2), ReLU(), Linear(self.out_channels//2, 1))

   def forward(self, batch_data):

      x, edge_index, edge_attr = batch_data.x, batch_data.edge_index, batch_data.edge_attr

      x = self.ceal_1(x, edge_index, batch_data.batch, edge_attr)
      x = self.batch_norm_1(x)
      x = F.relu(x)
      x = F.dropout(x, p=self.dropout, training=self.training)

      for cealconv, batch_norm in zip(self.cealConvs, self.batch_norms):
          x = batch_norm(cealconv(x, edge_index, batch_data.batch, edge_attr))
          x = F.relu(x)
          x = F.dropout(x, p=self.dropout,  training=self.training)

      x = global_add_pool(x, batch_data.batch)
      out = self.post_mlp(x)
      return out

class GCNNetwork(torch.nn.Module):
   def __init__(self, in_channels, out_dim, numLayers=2):
      super().__init__()
      self.in_channels = in_channels
      self.out_dim = out_dim
      self.numLayers = numLayers
      self.conv1 = GCNConv(self.in_channels, out_dim)
      self.batch_norm1 = BatchNorm(self.out_dim)

      self.gcnConvs = ModuleList()
      self.batch_norms = ModuleList()
      for _ in range(1, self.numLayers):
         gcnconv = GCNConv(self.out_dim, out_dim)
         batch_norm = BatchNorm(self.out_dim)
         self.gcnConvs.append(gcnconv)
         self.batch_norms.append(batch_norm)

      self.post_mlp = Sequential(Linear(out_dim, 100), ReLU(), Linear(100, 1))

   def forward(self, batch_data):

      x, edge_index = batch_data.x, batch_data.edge_index

      x = self.conv1(x, edge_index)
      x = self.batch_norm1(x)
      x = F.relu(x)
      x = F.dropout(x, p=0.2, training=self.training)

      for gcnconv, batch_norm in zip(self.gcnConvs, self.batch_norms):
         x = batch_norm(gcnconv(x, edge_index))
         x = F.relu(x)
         x = F.dropout(x, p=0.2,  training=self.training)

      x = global_add_pool(x, batch_data.batch)

      out = self.post_mlp(x)
      return out

class PNANetwork(torch.nn.Module):
   def __init__(self,
                in_channels,
                out_channels,
                aggregators,
                scalers,
                deg,
                edge_dim = None,
                towers = 1,
                pre_layers = 1,
                post_layers = 1,
                divide_input = False,
                numLayers = 2
               ):
      super().__init__()
      self.in_channels = in_channels
      self.out_channels = out_channels
      self.aggregators = aggregators
      self.scalers = scalers
      self.deg = deg
      self.edge_dim = edge_dim
      self.towers = towers
      self.pre_layers = pre_layers
      self.post_layers = post_layers
      self.divide_input = divide_input
      self.numLayers = numLayers

      self.convs = ModuleList()
      self.batch_norms = ModuleList()
      for _ in range(self.numLayers):
         conv = PNAConv(out_channels,
                        out_channels,
                        aggregators,
                        scalers,
                        deg,
                        edge_dim = edge_dim,
                        towers=1,
                        pre_layers=1,
                        post_layers=1,
                        divide_input=False)

         batch_norm = BatchNorm(self.out_channels)
         self.convs.append(conv)
         self.batch_norms.append(batch_norm)

      self.pre_mlp = Sequential(Linear(self.in_channels, self.out_channels//2), ReLU(), Linear(self.out_channels//2, self.out_channels))
      self.post_mlp = Sequential(Linear(self.out_channels, 100), ReLU(), Linear(100, 1))

   def forward(self, batch_data):

      x, edge_index, edge_attr = batch_data.x, batch_data.edge_index, batch_data.edge_attr

      x = self.pre_mlp(x)

      for conv, batch_norm in zip(self.convs, self.batch_norms):
         x = batch_norm(conv(x, edge_index, edge_attr))
         x = F.relu(x)
         x = F.dropout(x, p=0.3,  training=self.training)

      x = global_add_pool(x, batch_data.batch)

      out = self.post_mlp(x)
      return out

class EdgeSumNetwork(torch.nn.Module):
   def __init__(self, edge_dim, hidden=100, dropout=0.2):
      super().__init__()
      self.edge_dim = edge_dim; self.out_channels = hidden; self.dropout = dropout
      self.mlp = Sequential(Linear(edge_dim, hidden), ReLU(), Linear(hidden, hidden), ReLU(), Linear(hidden, 1))
      self.bias = torch.nn.Parameter(torch.zeros(1))
   def forward(self, batch_data):
      e = self.mlp(batch_data.edge_attr)
      eb = batch_data.batch[batch_data.edge_index[0]]
      return global_add_pool(e, eb, size=batch_data.num_graphs) + self.bias

from torch.nn import ModuleList as _RevModuleList
from torch_geometric.nn import SAGEConv as _RevSAGEConv, GATConv as _RevGATConv
from torch_geometric.utils import scatter as _rev_scatter

class BaselineNetwork(torch.nn.Module):
   def __init__(self, kind, in_channels, hidden, edge_dim, numLayers=2, node_edge_agg=True, dropout=0.2, heads=4):
      super().__init__()
      self.kind = kind; self.in_channels = in_channels; self.out_channels = hidden; self.edge_dim = edge_dim
      self.numLayers = numLayers; self.node_edge_agg = node_edge_agg; self.dropout = dropout
      d_in = in_channels + (2 * edge_dim if node_edge_agg else 0)
      self.convs = _RevModuleList(); self.batch_norms = _RevModuleList()
      for l in range(numLayers):
         i = d_in if l == 0 else hidden
         if kind == 'GCN':
            conv = GCNConv(i, hidden)
         elif kind == 'SAGE':
            conv = _RevSAGEConv(i, hidden)
         elif kind == 'GAT':
            conv = _RevGATConv(i, hidden // heads, heads=heads, edge_dim=(edge_dim if node_edge_agg else None))
         else:
            raise ValueError(kind)
         self.convs.append(conv); self.batch_norms.append(BatchNorm(hidden))
      self.post_mlp = Sequential(Linear(hidden, 100), ReLU(), Linear(100, 1))

   def forward(self, batch_data):
      x, edge_index, edge_attr = batch_data.x, batch_data.edge_index, batch_data.edge_attr
      x = x.float()
      if self.node_edge_agg:
         es = _rev_scatter(edge_attr, edge_index[1], dim=0, dim_size=x.size(0), reduce='sum')
         em = _rev_scatter(edge_attr, edge_index[1], dim=0, dim_size=x.size(0), reduce='mean')
         x = torch.cat([x, es, em], dim=-1)
      for conv, bn in zip(self.convs, self.batch_norms):
         if self.kind == 'GAT' and self.node_edge_agg:
            x = conv(x, edge_index, edge_attr)
         else:
            x = conv(x, edge_index)
         x = F.dropout(F.relu(bn(x)), p=self.dropout, training=self.training)
      x = global_add_pool(x, batch_data.batch)
      return self.post_mlp(x)
