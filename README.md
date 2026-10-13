# Mechanics-informed graph learning predicts axial stress in fibrin networks

Beikang Gu<sup>1</sup>, Kun Jiang<sup>1</sup>, Yuhang Cao<sup>1</sup>, Yanyu Chen<sup>1</sup>, Kenan Song<sup>1</sup>,
Xiaowei Zeng<sup>2</sup>, Yong Wei<sup>3,*</sup>, He Li<sup>4,*</sup> and Xianqiao Wang<sup>1</sup>

<sup>1</sup> School of Environmental, Civil, Agricultural and Mechanical Engineering, College of Engineering, University of Georgia, Athens, GA 30602  
<sup>2</sup> Department of Mechanical, Aerospace & Industrial Engineering, University of Texas at San Antonio, San Antonio, TX 78249  
<sup>3</sup> Department of Computer Science and Cybersecurity, University of North Georgia, Dahlonega, GA 30597  
<sup>4</sup> School of Chemical, Materials and Biomedical Engineering, College of Engineering, University of Georgia, Athens, GA 30602  
<sup>*</sup> Corresponding authors

## Acknowledgment and contribution of Prof. Yong Wei

### The graph neural network algorithm and code of this work, i.e. the CEAL-WF model and the training program in `4_training/`, were provided by Prof. Yong Wei (Department of Computer Science and Cybersecurity, University of North Georgia). We sincerely thank Prof. Wei for this contribution.

This repository holds the code of the paper.
It covers the complete workflow of the heterogeneous dataset (called Tier 4 in the paper):

1. generation of the random fibrin networks (MATLAB),
2. coarse-grained tensile simulations (LAMMPS),
3. construction of the junction-fiber graph dataset from the simulation output (Python),
4. training of the CEAL-WF model and of the reference models (Python, PyTorch Geometric).

The dataset itself is not distributed. The simulation output of the 10,200 networks of the paper is far too large to be
hosted (more than 1 GB per network), so it has to be produced with the steps below. One simulation of the paper took
15 h on average on six CPU cores, and the networks are generated at random, so a new dataset is statistically
equivalent to the one of the paper but not identical to it.

## Requirements

- MATLAB with the Parallel Computing Toolbox (tested with R2025b).
- LAMMPS, version 31 Mar 2017, built with MPI and with the MOLECULE and MC packages. The input uses only standard styles
  (`bond_style table`, `pair_style table`, `angle_style cosine/delta`, `fix bond/break`, `fix deform`).
- Python 3.10 with torch 2.5.1, torch_geometric 2.6.1, torch_scatter 2.1.2, numpy 2.0.1, pandas 2.2.3, scipy 1.15.2,
  scikit-learn 1.6.1, matplotlib 3.10.1, tqdm 4.67.1 and ase 3.25.0.

## Folders

```
1_generate_networks/   generate_networks.m and its functions
2_simulation/          LAMMPS input (IN3), run_all.sh and the script that writes the potential tables (potential/)
3_dataset/             build_dataset.py
4_training/            graph neural network code by Yong Wei (run_training.py is the main program, config_1.py holds the settings)
runs/                  created by step 1; one folder per network
```

## Step 1. Generate the networks

Open `1_generate_networks/generate_networks.m` and set, at the top of the file,

- `n`: the junction counts (the paper uses `50:100`),
- `need`: the number of networks per junction count (the paper uses `200`),
- `dlr` and `dsel`: the fiber diameters (the paper uses the 16 classes from 0.05 to 0.20 um).

`runname` is the name of the folder that is created under `runs/`. Then run, inside `1_generate_networks`,

```
matlab -batch generate_networks
```

Every network gets its own folder `runs/<runname>/<group>`, where `group = junction count * 10000 + index`
(for example `500000` to `500199` for the 200 networks with 50 junctions). The folder contains the LAMMPS data file
`voc2.data` and a copy of the input file `IN3`. The box is a periodic cube with an edge of 30 um.

## Step 2. Run the simulations

The tabulated potentials are not stored in the repository because of their size (1.2 GB). Write them once with MATLAB,
inside `2_simulation/potential`:

```
matlab -batch fibrin_potentialf
```

This gives `f1T` to `f16T` (stretching tables of the 16 diameter classes), `f1C` to `f16C` (contact tables; the input
uses `f1C` for all pairs) and `EI` (the bending coefficients, which are already entered as `angle_coeff` in `IN3`).
The script reproduces the tables used for the paper exactly.

Every network is one LAMMPS run in its own folder:

```
cd runs/<runname>/<group>
mpirun -np 6 lmp_mpi -screen none -var route /absolute/path/to/2_simulation/potential/ -in IN3
```

`2_simulation/run_all.sh` does this for all folders of a run:

```
LMP=lmp_mpi LAUNCH="mpirun -np 6" JOBS=4 bash 2_simulation/run_all.sh runs/<runname>
```

`LMP` is the LAMMPS executable, `LAUNCH` the MPI launcher with the number of processes of one simulation and `JOBS` the
number of simulations that run at the same time. On a cluster, submit the same LAMMPS command with your own scheduler
script; no scheduler script is included here.

The input stretches the box along x at 7.5 mm/s at constant volume with a time step of 5 ns, pauses the loading for
150,000 steps after every rupture event and stops at a strain of 6. It writes

- `cfg/<step>.d` and `cfg/<step>.b` every 10,000 steps (bead positions with per-bead stress, and the list of intact bonds),
- `informationinthesimulation` every 10 steps (step, strain, axial stress and energies).

## Step 3. Build the graph dataset

```
python 3_dataset/build_dataset.py runs/<runname> 4_training/dataset/Data_0.5 --keep 0.5 --nproc 32
```

The script reads the simulation folders directly and in parallel and writes `processed/data.pt` (PyTorch Geometric
format) and `processed/graphs.csv` (one line per graph) into the output folder. Every retained snapshot becomes one graph:
the junctions are the nodes (node feature: number of intact fibers at the junction, limited to 1-4), every intact fiber
is a pair of directed edges, and the edge attributes are the rest length of the fiber, the component of its
junction-to-junction vector along the tensile axis, the magnitude of the perpendicular component, the cosine of the
angle to the tensile axis and the fiber diameter. The label is the axial stress of the snapshot. Snapshots written
during the pauses after a rupture are not used, and no graphs are made once the network no longer spans the box
along the tensile axis.

`--keep` is the fraction of the graphs of every network that is kept (drawn at random with `--seed`, default 42).
The paper uses 0.5 for the reference model and 0.25 for the comparisons between architectures and aggregators;
`--keep 1` keeps all graphs.

## Step 4. Train

The model and the training program in `4_training/` were written by Yong Wei.

Open `4_training/config_1.py` and check the junction counts used for training, validation and testing
(`_TRAIN_N`, `_VAL_N`, `_TEST_N`; the paper uses 50-80, 81-90 and 91-100). Then run, inside `4_training`,

```
DATASET_NAME=Data_0.5 GROUP_SIZE=200 MODEL_TYPE=ChemGNN AGGREGATORS=sum,mean,min,max,std \
python run_training.py --numLayers 2 --batch_size 128 --out_channels 100 --dropout 0.2
```

- `DATASET_NAME`: folder under `4_training/dataset/` written in step 3.
- `GROUP_SIZE`: number of networks per junction count that are used (200 = all networks of the paper).
- `MODEL_TYPE`: `ChemGNN` is CEAL-WF; `PNA`, `EdgeSum`, `GCNb`, `SAGEb`, `GATb` and `GCN` are the reference models.
- `AGGREGATORS`: aggregators of the CEAL-WF layer.
- Optional: `SEED`, `LR` and `EPOCHS` override the seed, the learning rate (0.001) and the number of epochs (1500).

The trained model, the normalization constants, the training history and the predictions for the test set are written
to `4_training/Results/`.

## A small test of the whole workflow

To check an installation without the cost of the full dataset, set for example `n=10:15`, `need=2` and
`runname='test'` in `generate_networks.m`, run steps 1 to 3 with `--keep 1`, set `_TRAIN_N = range(10, 14)`,
`_VAL_N = range(14, 15)` and `_TEST_N = range(15, 16)` in `config_1.py` and train with `GROUP_SIZE=2 EPOCHS=50`
(the training history is recorded every 10 epochs, so use at least 10 epochs). Twelve such networks have 1,000 to 2,300
beads each; one simulation takes a few hours on six cores.
