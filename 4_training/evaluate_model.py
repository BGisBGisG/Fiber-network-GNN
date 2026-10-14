import json
import os
import os.path as osp

import torch
from torch_geometric.loader import DataLoader

from config import *
from dataset import *
from model import *
from train import test

import importlib

globals().update(vars(importlib.import_module('config_1')))

model_dir = os.environ.get('MODEL_DIR', '../5_trained_model')
model_name = os.environ.get('MODEL_NAME', 'tier4_ceal_wf')
device = config.device

with open(osp.join(model_dir, model_name + '_model_para.json')) as f:
    para = json.load(f)
with open(osp.join(model_dir, model_name + '_hyper_parameters.json')) as f:
    hyper = json.load(f)
with open(osp.join(model_dir, model_name + '_normalization.json')) as f:
    norm = json.load(f)

model = MyCEALNetwork(para['in_channels'],
                      para['out_channels'],
                      para['aggregators'],
                      para['scalers'],
                      torch.tensor(para['deg'], dtype=torch.long),
                      edge_dim=para['edge_dim'],
                      towers=para['towers'],
                      numLayers=para['numLayers'],
                      pre_layers=para['pre_layers'],
                      post_layers=para['post_layers'],
                      divide_input=False,
                      aggMLP=para['aggMLP'],
                      dropout=hyper['dropout'])
model.load_state_dict(torch.load(osp.join(model_dir, model_name + '_model.pt'), map_location=device, weights_only=True))
model.to(device)

norm_stats = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in norm.items()}

_, _, test_dataset = prep_datasets(config, config.train_groups, config.val_groups, config.test_groups)
loader = DataLoader(test_dataset, batch_size=1280, shuffle=False)

mae, out, y = test(model, loader, device, norm_stats=norm_stats)
out = out.view(-1)
mape = ((out - y).abs() / y.abs()).mean().item() * 100
r2 = 1 - ((out - y) ** 2).sum().item() / ((y - y.mean()) ** 2).sum().item()
print('test graphs', y.numel())
print('MAE (Pa)', round(mae, 4))
print('MAPE (%)', round(mape, 4))
print('R2', round(r2, 6))
