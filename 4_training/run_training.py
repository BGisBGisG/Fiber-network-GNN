import os
import torch
from torch_geometric.loader import DataLoader

from config import *
from dataset import *
from utils_data_laisuo import *
from model import *
from train import *

import importlib

def import_config_module(N):
    module_name = f"config_{N}"
    print('module_name = ', module_name)
    module = importlib.import_module(module_name)
    globals().update(vars(module))

configID = int(os.getenv('configID', '1'))
import_config_module(configID)

device = config.device
print('device = ', device)
seed = config.seed
shuffle = config.shuffle
batch_size = config.batch_size
num_workers = config.num_workers
disp_step = config.disp_step
dispProgress = config.dispProgress
learning_rate = config.learning_rate
epochs = config.epochs

if os.environ.get('SEED'):
    seed = int(os.environ['SEED'])
    import random as _rev_random, numpy as _rev_np
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); _rev_random.seed(seed); _rev_np.random.seed(seed)
    print('SEED ->', seed)
if os.environ.get('LR'):
    learning_rate = float(os.environ['LR']); print('LR ->', learning_rate)
if os.environ.get('EPOCHS'):
    epochs = int(os.environ['EPOCHS']); print('EPOCHS ->', epochs)

train_dataset_idx = config.train_dataset_idx
test_dataset_idx = config.test_dataset_idx
train_p = config.train_p
val_p = config.val_p
split_shuffle = config.split_shuffle

model_type = config.model_type
out_channels = config.out_channels
aggregators = config.aggregators
towers = config.towers
numLayers = config.numLayers
pre_layers = config.pre_layers
post_layers = config.post_layers
divide_input = config.divide_input
aggMLP = config.aggMLP
dropout = config.dropout

combined_train_dataset, combined_val_dataset, combined_test_dataset = prep_datasets(
    config,
    config.train_groups,
    config.val_groups,
    config.test_groups
)

print('combined_train_dataset ', len(combined_train_dataset))
print('combined_val_dataset ', len(combined_val_dataset))
print('combined_test_dataset ', len(combined_test_dataset))

g = torch.Generator()
g.manual_seed(seed)
train_data_loader = DataLoader(combined_train_dataset,
                               batch_size=batch_size,
                               shuffle=shuffle,
                               num_workers=num_workers,
                               worker_init_fn=worker_init_fn,
                               generator=g)
val_data_loader =   DataLoader(combined_val_dataset,
                               batch_size=batch_size*10,
                               shuffle=shuffle,
                               num_workers=num_workers,
                               worker_init_fn=worker_init_fn,
                               generator=g)
test_data_loader =  DataLoader(combined_test_dataset,
                               batch_size=batch_size*10,
                               shuffle=shuffle,
                               num_workers=num_workers,
                               worker_init_fn=worker_init_fn,
                               generator=g)

in_channels = combined_train_dataset[0].x.size(dim=-1)
print('combined_train_dataset[0].x ', combined_train_dataset[0].x.shape)
print('combined_train_dataset[0].edge_index ', combined_train_dataset[0].edge_index.shape)
print('edge_attr ', combined_train_dataset[0].edge_attr.shape)
print('in_channels ', in_channels)

scalers = ['identity']
scalers_PNA = ['identity', 'amplification', 'attenuation', 'linear', 'inverse_linear']
deg = generate_deg(combined_train_dataset)
edge_dim = combined_train_dataset[0].edge_attr.shape[1]

if model_type=='ChemGNN':
   model = MyCEALNetwork( in_channels,
                          out_channels,
                          aggregators,
                          scalers,
                          deg,
                          edge_dim = edge_dim,
                          towers = towers,
                          numLayers = numLayers,
                          pre_layers = pre_layers,
                          post_layers = post_layers,
                          divide_input = False,
                          aggMLP = aggMLP,
                          dropout = dropout
                        )
elif model_type=='GCN':
   print('GCN model is used')
   model =  GCNNetwork(in_channels, out_channels, numLayers=numLayers)
elif model_type=='PNA':
   model =  PNANetwork(in_channels,
                       out_channels,
                       aggregators,
                       scalers_PNA,
                       deg,
                       edge_dim = edge_dim,
                       towers = towers,
                       pre_layers = pre_layers,
                       post_layers = post_layers,
                       divide_input = False,
                       numLayers = numLayers)
elif model_type in ('GCNb', 'SAGEb', 'GATb'):
   model = BaselineNetwork(model_type[:-1], in_channels, out_channels, edge_dim, numLayers=numLayers,
                           node_edge_agg=(os.environ.get('NODE_EDGE_AGG', '1') == '1'), dropout=dropout)
   print('baseline', model_type, 'node_edge_agg =', model.node_edge_agg)
elif model_type=='EdgeSum':
   model = EdgeSumNetwork(edge_dim, out_channels, dropout=dropout)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=40, min_lr=1e-9)

train_str = ''.join(str(x) for x in train_dataset_idx)
test_str = ''.join(str(x) for x in test_dataset_idx)
if aggMLP:
   mtype = 'mlp'
else:
   mtype = 'weight'
players = str(numLayers) +  '_layers'

if model_type == 'ChemGNN' or model_type == 'PNA':
   prefix = model_type + '_' + mtype + '_' + players + '_' + train_str + '_' + test_str
elif model_type == 'GCN':
   prefix = model_type + '_' + train_str + '_' + test_str
else:
   prefix = model_type + '_' + players + '_' + train_str + '_' + test_str

hyper_dict = {'model_type'        : model_type,
              'aggMLP'            : aggMLP,
              'pre_layers'        : pre_layers,
              'post_layers'       : post_layers,
              'numLayers'         : numLayers,
              'learning_rate'     : learning_rate,
              'batch_size'        : batch_size,
              'epochs'            : epochs,
              'train_p'           : train_p,
              'val_p'             : val_p,
              'train_dataset_idx' : train_dataset_idx,
              'test_dataset_idx'  : test_dataset_idx,
              'out_channels'      : out_channels,
              'aggregators'       : aggregators,
              'scalers_PNA'       : scalers_PNA,
              'dropout'           : dropout,
              'dataset_name'      : os.environ.get('DATASET_NAME', 'NA'),
              'group_size'        : os.environ.get('GROUP_SIZE', 'NA'),
             }

print('hyper_dict: ', hyper_dict)

train(epochs, model, train_data_loader, val_data_loader, test_data_loader, optimizer, scheduler, device, disp_step, prefix, hyper_dict, dispProgress=dispProgress)
