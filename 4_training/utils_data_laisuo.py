import os
import os.path as osp
from pathlib import Path
import numpy as np
import re
import random
import torch
from torch_geometric.data import Data

import ase
from ase.io import read as ase_read
from ase.data import atomic_numbers
from sklearn import preprocessing
import copy

def replace_C_with_Cl(file_path):
    pattern = re.compile(r'(?<![A-Za-z])C(?=\d)')

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = [pattern.sub('Cl', line) for line in lines]

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

def replace_C_with_Cl_in_filenames(folder_path):

    pattern = re.compile(r'C(?=\d)')

    for name in os.listdir(folder_path):
        new_name = pattern.sub('Cl', name)
        if new_name != name:
            src = os.path.join(folder_path, name)
            dst = os.path.join(folder_path, new_name)
            print(f"Renaming:\n {src}\n --> {dst}\n")
            os.rename(src, dst)

def count_poscar_files(folder_path):
    folder = Path(folder_path)

    return sum(1 for _ in folder.glob("POSCAR*") if _.is_file())

def get_compound_energy(compound, file_path):
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == compound:
                try:
                    return float(parts[1])
                except (IndexError, ValueError):

                    return None
    return None

def load_atomCharges(folder, file):
   filename = osp.join(folder, file)
   with open(filename, "r") as f:
      data = f.readlines()
   data = np.asarray(data, dtype=float)
   data = torch.tensor(data, dtype=torch.float64)
   return data

def get_adj_matrix(dmatrix, cutoff):
   adjmatrix = torch.zeros_like(dmatrix, dtype=int)
   mask = dmatrix<=cutoff
   adjmatrix[mask] = 1
   adjmatrix.fill_diagonal_(0)
   return adjmatrix

def get_edge_index(bt_data):

   edge_index = bt_data.nonzero().t().contiguous()
   return edge_index

def gen_one_hot_dict(symbols):
   symbols_original = symbols

   le = preprocessing.LabelEncoder()
   symbols_num_list = le.fit_transform(symbols)
   symbols = torch.as_tensor(symbols_num_list)
   symbols = torch.unique(symbols)

   numS = symbols.size(dim=0)
   code = [0.0] * numS
   atom_type_dict = {}
   pos = 0
   for atom in symbols:
      a_code = copy.deepcopy(code)
      a_code[pos] = 1

      atom_type_dict[le.inverse_transform([atom]).item()] = a_code
      pos += 1
   return atom_type_dict

def atom_list_2_one_hot(atom_list, atom_type_dict):

   atom_type_list = []
   for atom in atom_list:
      acode = atom_type_dict[atom]
      atom_type_list.append(acode)
   atom_type_list = torch.tensor(atom_type_list)
   return atom_type_list

def read_one_molecule_info(datasrc, config, cutoff, atom_type_onehot_dict, poscar_file):

   compound_name = poscar_file.split("_", 1)[1]
   ef_data_file = osp.join(datasrc, 'amorphous_formation_free_energy.dat')
   energy = get_compound_energy(compound_name, file_path=ef_data_file)
   if energy is None:
      print(compound_name, 'not found in ef_data_file')
      return None
   y = energy

   filename = osp.join(datasrc, poscar_file)
   print(filename)

   front1 = "P1_"
   name1 = front1 + compound_name
   filename1 = osp.join(datasrc, name1)
   edge_index = np.loadtxt(filename1)
   edge_index = torch.from_numpy(edge_index).long().contiguous()
   edge_attr = load_edge_attr(datasrc, compound_name, edge_index, prefix="P2_")
   front1 = "PX_"
   name1 = front1 + compound_name
   filename1 = osp.join(datasrc, name1)
   x = np.loadtxt(filename1)
   x = torch.from_numpy(x).long()
   if x.dim() == 1:
    x = x.unsqueeze(1)

   data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

   return data

def split_dataset(dataset, train_p=70, val_p=15, shuffle=True):

   if shuffle:

      idx = list(range(dataset.len()))
      random.shuffle(idx)

      idx = torch.tensor(idx)
   else:

      idx = torch.tensor(range(dataset.len()))

   len_train = int(len(idx) * train_p)
   len_val = int(len(idx) * val_p)

   train_dataset = dataset.index_select(idx[0:len_train])
   val_dataset = dataset.index_select(idx[len_train:len_train+len_val])
   test_dataset = dataset.index_select(idx[len_train+len_val:])

   return train_dataset, val_dataset, test_dataset

def load_edge_attr(datasrc, compound_name, edge_index, prefix="P2_"):
    name1 = prefix + compound_name
    filename1 = osp.join(datasrc, name1)

    edge_attr_np = np.loadtxt(filename1, comments='#')
    E = int(edge_index.size(1))

    if edge_attr_np.ndim == 1:
        if edge_attr_np.size == E:
            edge_attr_np = edge_attr_np.reshape(E, 1)
        elif edge_attr_np.size == 2 * E:
            edge_attr_np = edge_attr_np.reshape(2, E).T
        else:

            edge_attr_np = edge_attr_np.reshape(-1, 1)

    rows, cols = edge_attr_np.shape
    if cols == E:
        edge_attr_np = edge_attr_np.T
    elif rows == E:
        pass
    else:

        if cols > E:
            edge_attr_np = edge_attr_np[:, :E].T
        elif cols < E:
            pad_width = ((0, 0), (0, E - cols))
            edge_attr_np = np.pad(edge_attr_np, pad_width, mode='constant').T

    edge_attr = torch.as_tensor(edge_attr_np, dtype=torch.float32).contiguous()

    if edge_attr.dim() == 1:
        edge_attr = edge_attr.view(-1, 1)

    assert edge_attr.size(0) == E, \
        f"{filename1}: edge_attr has {edge_attr.size(0)} rows, expected E={E}"

    return edge_attr
