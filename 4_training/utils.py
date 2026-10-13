import os

import os.path as osp
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
import re
import json
from scipy.stats import gaussian_kde

import torch
from torch_geometric.utils import degree

def generate_deg(dataset):
    max_degree = -1
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)

        max_degree = max(max_degree, int(d.max()))

    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg

def worker_init_fn(worker_id):

    np.random.seed(np.random.get_state()[1][0] + worker_id)
    torch.manual_seed(np.random.get_state()[1][0] + worker_id)

def plot_training_progress(epoch, train_losses, val_losses, test_losses, title):
   lw = 0.7
   ms = 0.7

   threshold = 1
   for i in range(len(train_losses)):
      if train_losses[i]>threshold:
         train_losses[i] = threshold
      if val_losses[i]>threshold:
         val_losses[i] = threshold
      if test_losses[i]>threshold:
         test_losses[i] = threshold

   plt.plot(range(10, epoch+1, 10), train_losses, marker='o', linestyle='-', color='b', lw=lw, ms=ms)
   plt.plot(range(10, epoch+1, 10), val_losses, marker='s', linestyle='-', color='r', lw=lw, ms=ms)
   plt.plot(range(10, epoch+1, 10), test_losses, marker='*', linestyle='-', color='g', lw=lw, ms=ms)

   legend_entries = [Line2D([0], [0], color='blue', label='train_loss (Blue)'),
                     Line2D([0], [0], color='red', label='validate_loss (Red)'),
                     Line2D([0], [0], color='green', label='test_loss (Green)')]
   plt.title(title)

   plt.legend(handles=legend_entries, loc='upper right')

   plt.tight_layout()
   plt.show()

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

def save_mae(folder, maeFile, test_loss, msg):

   result_file = osp.join(folder, maeFile)
   test_loss = test_loss
   if osp.exists(result_file):
      with open(result_file, "a") as file:
         file.write(msg + '\n')
         file.write(str(test_loss) + '\n')
         file.close()
   else:
      with open(result_file, "w") as file:
         file.write(msg + '\n')
         file.write(str(test_loss) + '\n')
         file.close()

def save_train_progress(folder, filename, epochs, train_losses, val_losses, test_losses):

   result_file = osp.join(folder, filename)
   with open(result_file, 'w') as file:
      row_str = " ".join(map(str, epochs))
      file.write(row_str + "\n")
      row_str = " ".join(map(str, train_losses))
      file.write(row_str + "\n")
      row_str = " ".join(map(str, val_losses))
      file.write(row_str + "\n")
      row_str = " ".join(map(str, test_losses))
      file.write(row_str + "\n")
      file.close()

def extract_prefix(filename):
   match = re.match(r'^([^_]+_[^_]+_[^_]+_[^_]+)_.+', filename)
   if match:
       return match.group(1)
   else:
      return None

def plot_training_from_file(folder, filename, plotfilename=None, disp=True):

   prefix = extract_prefix(filename)

   train_sets = ''
   test_sets = ''
   match = re.search(r'\d+_\d+', filename)
   if match:
      datasets = match.group()
   datasets = datasets.split('_')
   train_sets = datasets[0]
   test_sets = datasets[1]
   title = prefix + ', Training Sets: ' + train_sets + ' Testing Sets: ' + test_sets

   result_file = osp.join(folder, filename)
   matrix = []
   with open(result_file, 'r') as file:
      for line in file:
         numbers = list(map(float, line.split()))
         matrix.append(numbers)
      file.close()

   epochs = list(map(int, matrix[0]))
   train_losses = matrix[1]
   val_losses = matrix[2]
   test_losses = matrix[3]
   maxepoch = max(epochs)

   threshold = 1000000
   for i in range(len(epochs)):
      if train_losses[i]>threshold:
         train_losses[i] = threshold
      if val_losses[i]>threshold:
         val_losses[i] = threshold
      if test_losses[i]>threshold:
         test_losses[i] = threshold

   plt.figure(figsize=(8, 6))
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.title('Loss vs. Epoch during Training')
   plt.grid(True)
   lw = 0.7
   ms = 0.7
   plt.plot(range(10, maxepoch+1, 10), train_losses, marker='o', linestyle='-', color='b', lw=lw, ms=ms)
   plt.plot(range(10, maxepoch+1, 10), val_losses, marker='s', linestyle='-', color='r', lw=lw, ms=ms)
   plt.plot(range(10, maxepoch+1, 10), test_losses, marker='*', linestyle='-', color='g', lw=lw, ms=ms)
   legend_entries = [Line2D([0], [0], color='blue', label='train_loss (Blue)'),
                     Line2D([0], [0], color='red', label='validate_loss (Red)'),
                     Line2D([0], [0], color='green', label='test_loss (Green)')]
   plt.title(title)
   plt.legend(handles=legend_entries, loc='upper right')

   if disp:
      plt.show()

   if plotfilename != None:
      plotfilename = osp.join(folder, plotfilename)
      plt.savefig(plotfilename)

def save_agg_weights(folder, filename, model):
   result_file = osp.join(folder, filename)
   with open(result_file, 'w') as file:
      for name, param in model.named_parameters():
         if 'agg_weights' in name or 'aggregator_weights' in name:
            x = param.data.clone()
            x = torch.nn.functional.softmax(x)
            x = x.cpu().numpy()
            msg = 'Parameter name: ' + name + ', '
            file.write(f"{msg}{x}" + '\n')

      file.close()

def save_regression_result(folder, filename, test_out, test_y):
   result_file = osp.join(folder, filename)
   test_y = torch.unsqueeze(test_y, dim=1)
   result = torch.cat((test_out, test_y), dim=1)
   result = result.cpu().numpy()
   with open(result_file, 'w') as file:
      for row in result:
         row_str = ' '.join(str(element) for element in row)
         file.write(row_str + '\n')
      file.close()

def save_model_CEAL(model, folder, model_filename, model_para_filename):
   model_filename = osp.join(folder, model_filename)
   torch.save(model.state_dict(), model_filename)

   in_channels = model.in_channels
   out_channels = model.out_channels
   aggregators = model.aggregators
   scalers =  model.scalers
   deg = model.deg.tolist()
   numLayers = model.numLayers
   edge_dim = model.edge_dim
   towers = model.towers
   pre_layers = model.pre_layers
   post_layers = model.post_layers
   divide_input = model.divide_input
   aggMLP = model.aggMLP

   data = {'in_channels': in_channels,
           'out_channels': out_channels,
           'aggregators': aggregators,
           'scalers': scalers,
           'deg': deg,
           'numLayers': numLayers,
           'edge_dim': edge_dim,
           'towers': towers,
           'pre_layers': pre_layers,
           'post_layers': post_layers,
           'divide_input': divide_input,
           'aggMLP': aggMLP,
          }
   model_para_filename = osp.join(folder, model_para_filename)
   with open(model_para_filename, 'w') as file:
      json.dump(data, file)

def save_model_GCN(model, folder, model_filename, model_para_filename):
   model_filename = osp.join(folder, model_filename)
   torch.save(model, model_filename)

   in_channels = model.in_channels
   out_dim = model.out_dim
   numLayers = model.numLayers

   data = {'in_channels': in_channels,
           'out_dim': out_dim,
           'numLayers': numLayers
          }
   model_para_filename = osp.join(folder, model_para_filename)
   with open(model_para_filename, 'w') as file:
      json.dump(data, file)

def save_model_PNA(model, folder, model_filename, model_para_filename):
   model_filename = osp.join(folder, model_filename)
   torch.save(model, model_filename)

   in_channels = model.in_channels
   out_channels = model.out_channels
   aggregators = model.aggregators
   scalers =  model.scalers
   deg = model.deg.tolist()
   numLayers = model.numLayers
   edge_dim = model.edge_dim
   towers = model.towers
   pre_layers = model.pre_layers
   post_layers = model.post_layers
   divide_input = model.divide_input

   data = {'in_channels': in_channels,
           'out_channels': out_channels,
           'aggregators': aggregators,
           'scalers': scalers,
           'deg': deg,
           'numLayers': numLayers,
           'edge_dim': edge_dim,
           'towers': towers,
           'pre_layers': pre_layers,
           'post_layers': post_layers,
           'divide_input': divide_input
          }
   model_para_filename = osp.join(folder, model_para_filename)
   with open(model_para_filename, 'w') as file:
      json.dump(data, file)

def save_hyper_parameter(folder, hyper_para_filename, hyper_dict):
   print('In save_hyper_parameter')

   hyper_para_filename = osp.join(folder, hyper_para_filename)
   with open(hyper_para_filename, 'w') as file:
      json.dump(hyper_dict, file)
   print('Done save_hyper_parameter')

def plot_regression_result(folder, filename, title, plotfilename=None, normalized=False, disp=False):

   y, x = load_regression_results(folder, filename)
   if normalized == True:
      y = reverse_min_max_scaler_1d(y)
      x = reverse_min_max_scaler_1d(x)

   dmin = float(min(x.min(), y.min()))
   dmax = float(max(x.max(), y.max()))
   plt.figure(figsize=(7,7))
   plt.scatter(x, y, s=8, marker='o')
   plt.plot([dmin, dmax], [dmin, dmax], linestyle='--', lw=0.8)

   plt.xlabel("True Values")
   plt.ylabel("Predicted Values")
   plt.title(title)
   plt.xlim(dmin, dmax)
   plt.ylim(dmin, dmax)
   plt.grid(True)

   if plotfilename != None:
      plotfilename = osp.join(folder, plotfilename)
      plt.savefig(plotfilename)

   if disp:
      plt.show()

def load_regression_results(folder, filename):
   filename = osp.join(folder, filename)

   with open(filename, "r") as f:
      data = f.readlines()
      f.close()
   data = [line.split() for line in data]
   data = np.asarray(data, dtype=float)
   y = data[:,0]
   x = data[:,1]
   return y, x

def calculate_r_squared(y_predicted, y_actual):
   ssr = np.sum((y_actual - y_predicted) ** 2)
   sst = np.sum((y_actual - np.mean(y_actual)) ** 2)
   r_squared = 1 - (ssr / sst)
   return r_squared

def get_density(x, y):
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    density = kernel(values)
    return density

def save_normalization(folder, filename, stats):
   result_file = osp.join(folder, filename)
   out = {}
   for k, v in stats.items():
      if hasattr(v, "detach"):
         out[k] = v.detach().cpu().numpy().tolist()
      else:
         out[k] = v
   with open(result_file, "w") as f:
      json.dump(out, f, indent=2)
