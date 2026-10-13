import re
import os
import os.path as osp
import torch
from tqdm import tqdm
from torch.utils.data import Subset
from torch_geometric.data import (
    Data,
    InMemoryDataset,
)
from torch.utils.data import ConcatDataset

from config import config
from utils_data_laisuo import *

class MyDataset(InMemoryDataset):

    def __init__(self, root, config, dataset_id, transform=None, pre_transform=None, pre_filter=None):
        self.config = config
        self.dataset_id = dataset_id
        self.cutoff = config.cutoff
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.my_processed_dir, 'data.pt')
        self.data, self.slices = torch.load(path, weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def my_processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        datasrc = osp.join('data', self.config.dataset[self.dataset_id])
        print('datasrc ', datasrc)

        if not os.path.exists(self.my_processed_dir):
            os.makedirs(self.my_processed_dir)

        numFiles = count_poscar_files(datasrc)
        print('Number of files = ', numFiles)

        pbar = tqdm(total=numFiles, miniters=1, mininterval=0.0)
        pbar.set_description(f"Processing dataset: {self.config.dataset[self.dataset_id]}")

        atom_type_onehot_dict = gen_one_hot_dict(self.config.element_list)
        data_list = []
        dataset_len = 0

        for fname in os.listdir(datasrc):
            if not fname.startswith("POSCAR_"):
                continue

            m = re.match(r"POSCAR_(\d+)Li(\d{2})", fname)
            if not m:
                continue

            group_id = int(m.group(1)) * 100 + int(m.group(2))

            data = read_one_molecule_info(
                datasrc,
                config=self.config,
                cutoff=self.cutoff,
                atom_type_onehot_dict=atom_type_onehot_dict,
                poscar_file=fname
            )
            if data is None:
                continue
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data.group = group_id
            data_list.append(data)
            dataset_len += 1
            pbar.update(1)

        pbar.close()
        print('dataset_len = ', dataset_len)

        torch.save(self.collate(data_list), osp.join(self.my_processed_dir, 'data.pt'))

def split_by_group(dataset, train_groups, val_groups, test_groups):
    train_idx, val_idx, test_idx = [], [], []
    for i, data in enumerate(dataset):
        g = data.group
        if g in train_groups:
            train_idx.append(i)
        elif g in val_groups:
            val_idx.append(i)
        elif g in test_groups:
            test_idx.append(i)
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)

def genDataset(config):
    root_list = [osp.join(config.dataset_root, ds) for ds in config.dataset]
    print('numDatasets = ', len(root_list))
    for root in root_list:
        print(root)
    for idx, root in enumerate(root_list):
        MyDataset(root=root, config=config, dataset_id=idx)

def _as_group_selector(groups):
    if groups is None:
        return {"type": "empty"}
    if isinstance(groups, range):

        if groups.step == 1 and groups.start < groups.stop:
            return {"type": "interval", "lo": groups.start, "hi": groups.stop - 1}
        return {"type": "set", "set": set(groups)}

    if not isinstance(groups, (list, tuple)) or len(groups) == 0:
        return {"type": "empty"}

    lo, hi = int(groups[0]), int(groups[-1])
    if len(groups) >= 2:
        ok = True

        head_n = min(256, len(groups) - 1)
        for i in range(head_n):
            if int(groups[i + 1]) - int(groups[i]) != 1:
                ok = False
                break
        if ok:
            tail_n = min(256, len(groups) - 1)
            base = len(groups) - 1 - tail_n
            for i in range(base, len(groups) - 1):
                if int(groups[i + 1]) - int(groups[i]) != 1:
                    ok = False
                    break
        if ok:
            return {"type": "interval", "lo": lo, "hi": hi}

    return {"type": "set", "set": set(int(x) for x in groups)}

def _indices_by_selector(group_tensor_1d: torch.Tensor, selector):
    if selector["type"] == "empty":
        return []

    if selector["type"] == "interval":
        lo, hi = selector["lo"], selector["hi"]
        mask = (group_tensor_1d >= lo) & (group_tensor_1d <= hi)
        return mask.nonzero(as_tuple=False).view(-1).tolist()

    s = selector["set"]

    g_list = group_tensor_1d.tolist()
    return [i for i, g in enumerate(g_list) if int(g) in s]

def prep_datasets(config, train_groups, val_groups, test_groups):
    root_list = [osp.join(config.dataset_root, ds) for ds in config.dataset]
    train_subsets, val_subsets, test_subsets = [], [], []

    train_sel = _as_group_selector(train_groups)
    val_sel   = _as_group_selector(val_groups)
    test_sel  = _as_group_selector(test_groups)

    for idx, root in enumerate(root_list):
        ds = MyDataset(root=root, config=config, dataset_id=idx)

        _ovr = os.environ.get('DATA_PT_OVERRIDE')
        if _ovr:
            _obj = torch.load(_ovr, map_location='cpu', weights_only=False)
            ds.data, ds.slices = _obj[0], _obj[1]
            print('DATA_PT_OVERRIDE ->', _ovr, 'edge_attr', tuple(ds._data.edge_attr.shape))

        _cols = os.environ.get('EDGE_COLS')
        if _cols:
            _idx = [int(c) for c in _cols.split(',')]
            ds._data.edge_attr = ds._data.edge_attr[:, _idx].contiguous()
            print('EDGE_COLS ->', _idx, 'edge_attr', tuple(ds._data.edge_attr.shape))

        if not hasattr(ds.data, "group"):
            raise AttributeError(f"[ERR] dataset {root} has no attribute 'group' in processed data.pt")

        group_tensor = ds.data.group
        if not torch.is_tensor(group_tensor):
            group_tensor = torch.tensor(group_tensor, dtype=torch.long)
        group_tensor = group_tensor.view(-1).to(torch.long).cpu()

        train_indices = _indices_by_selector(group_tensor, train_sel)
        val_indices   = _indices_by_selector(group_tensor, val_sel)
        test_indices  = _indices_by_selector(group_tensor, test_sel)

        _tif = os.environ.get('TRAIN_INDEX_FILE')
        if _tif:
            import numpy as _rev_np
            _keep = set(int(v) for v in _rev_np.load(_tif).tolist())
            train_indices = [i for i in train_indices if i in _keep]
            print('TRAIN_INDEX_FILE', _tif, '-> training graphs', len(train_indices))

        if train_indices:
            train_subsets.append(Subset(ds, train_indices))
        if val_indices:
            val_subsets.append(Subset(ds, val_indices))
        if test_indices:
            test_subsets.append(Subset(ds, test_indices))

    return (
        ConcatDataset(train_subsets) if train_subsets else None,
        ConcatDataset(val_subsets)   if val_subsets   else None,
        ConcatDataset(test_subsets)  if test_subsets  else None
    )
def prep_datasetsf(config, train_groups, val_groups, test_groups):
    root_list = [osp.join(config.dataset_root, ds) for ds in config.dataset]
    train_subsets, val_subsets, test_subsets = [], [], []

    train_sel = _as_group_selector(train_groups)
    val_sel   = _as_group_selector(val_groups)
    test_sel  = _as_group_selector(test_groups)

    for idx, root in enumerate(root_list):
        ds = MyDataset(root=root, config=config, dataset_id=idx)

        if not hasattr(ds.data, "group"):
            raise AttributeError(f"[ERR] dataset {root} has no attribute 'group' in processed data.pt")

        group_tensor = ds.data.group
        if not torch.is_tensor(group_tensor):
            group_tensor = torch.tensor(group_tensor, dtype=torch.long)
        group_tensor = group_tensor.view(-1).to(torch.long).cpu()

        train_indices = _indices_by_selector(group_tensor, train_sel)
        val_indices   = _indices_by_selector(group_tensor, val_sel)
        test_indices  = _indices_by_selector(group_tensor, test_sel)

        if train_indices:
            train_subsets.append(Subset(ds, train_indices))
        if val_indices:
            val_subsets.append(Subset(ds, val_indices))
        if test_indices:
            test_subsets.append(Subset(ds, test_indices))

    return (
        ConcatDataset(train_subsets) if train_subsets else None,
        ConcatDataset(val_subsets)   if val_subsets   else None,
        ConcatDataset(test_subsets)  if test_subsets  else None
    )

if __name__ == "__main__":
    genDataset(config)
