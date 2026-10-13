RTT='data'
import os
import os.path as osp
import re
import torch

def extract_elements(formula):

    pattern = r'([A-Z][a-z]?)\d*'
    return re.findall(pattern, formula)

def get_all_elements(file_path):
    elements = set()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            formula = line.split()[0]
            elems = extract_elements(formula)
            elements.update(elems)
    return sorted(elements)

class Config:
   main_path = './'
   dataset_root = 'dataset'

   cutoff = 3

   dataset = ['Data_0.5',
             ]
   element_list = set()
   element_list = ['C', 'Cl', 'Ga', 'Li', 'Ta', 'Zr']

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   print('device = ', device)
   seed = 23647
   shuffle = True
   batch_size = 128
   num_workers = 2
   disp_step = 10
   dispProgress = False
   learning_rate = 0.1
   epochs = 1000

   train_dataset_idx = [0]
   test_dataset_idx = [0]
   train_p = 0.6
   val_p = 0.2
   split_shuffle = True

   model_type = 'ChemGNN'
   out_channels = 20
   aggregators = ['sum', 'mean', 'min', 'max', 'std']
   towers = 1
   numLayers = 1
   pre_layers = 1
   post_layers = 1
   divide_input = False
   aggMLP = True
   aggMLP_factor = 0.5
   dropout = 0.3

config = Config
