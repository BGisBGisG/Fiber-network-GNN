import os
import os.path as osp
import torch
import argparse

# Number of networks per junction count that are used (the first GROUP_SIZE networks of every count).
# The paper has 200 networks per junction count and uses all of them.
_GS = int(os.environ.get('GROUP_SIZE', '200'))

# Aggregators of the CEAL-WF layer: any combination of sum, mean, min, max, std.
_AGG_STR = os.environ.get('AGGREGATORS', 'sum,mean,min,max,std')

# Model: ChemGNN (this is CEAL-WF), PNA, EdgeSum (edge-wise sum without message passing),
# GCNb / SAGEb / GATb (baselines that receive the edge attributes) or GCN.
_MODEL_TYPE = os.environ.get('MODEL_TYPE', 'ChemGNN')

# Dataset folder under ./dataset, i.e. the folder that contains processed/data.pt (written by 3_dataset/build_dataset.py).
if 'DATASET_NAME' not in os.environ:
    raise RuntimeError("Set the environment variable DATASET_NAME, for example: export DATASET_NAME=Data_0.5")
_DATASET_NAME = os.environ['DATASET_NAME']

# Junction counts of the networks used for training, validation and testing.
# The paper uses 50-80 for training, 81-90 for validation and 91-100 for testing.
# Every network has a group number = junction count * 10000 + index of the network (0, 1, 2, ...),
# which is also the name of its simulation folder.
_TRAIN_N = range(50, 81)
_VAL_N = range(81, 91)
_TEST_N = range(91, 101)
_TRAIN_GROUPS = [n * 10000 + k for n in _TRAIN_N for k in range(_GS)]
_VAL_GROUPS = [n * 10000 + k for n in _VAL_N for k in range(_GS)]
_TEST_GROUPS = [n * 10000 + k for n in _TEST_N for k in range(_GS)]


class Config:
   def __init__(self):
      p = argparse.ArgumentParser()
      # Number of CEAL-WF layers (paper: 2), mini-batch size (paper: 128),
      # hidden width (paper: 100 or 200) and dropout rate (paper: 0.2).
      p.add_argument("--numLayers",   type=int,   default=1)
      p.add_argument("--batch_size",  type=int,   default=32)
      p.add_argument("--out_channels",type=int,   default=50)
      p.add_argument("--dropout",     type=float, default=0.1)
      args = p.parse_args()

      self.numLayers   = args.numLayers
      self.batch_size  = args.batch_size
      self.out_channels= args.out_channels
      self.dropout     = args.dropout

   main_path = './'
   dataset_root = 'dataset'
   cutoff = 3
   dataset = [_DATASET_NAME]
   element_list = []

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   print('device = ', device)
   # Seed of the data loader, initial learning rate and number of epochs.
   # The environment variables SEED, LR and EPOCHS override them (SEED also seeds the model initialization).
   seed = 23647
   shuffle = True
   num_workers = 2
   disp_step = 10
   dispProgress = False
   learning_rate = 0.001
   epochs = 1500

   train_groups = _TRAIN_GROUPS
   val_groups   = _VAL_GROUPS
   test_groups  = _TEST_GROUPS
   train_dataset_idx = [0]
   test_dataset_idx = [0]
   train_p = 0.6
   val_p = 0.2
   split_shuffle = True

   model_type = _MODEL_TYPE
   aggregators = [a.strip() for a in _AGG_STR.split(',')]
   towers = 1
   pre_layers = 1
   post_layers = 1
   divide_input = False
   # False: the aggregated messages are combined with learned scalar weights (CEAL-WF).
   aggMLP = False
   aggMLP_factor = 0.5
   dropout = 0.1
config = Config()
