import os
import os.path as osp
import json
import numbers

import matplotlib.pyplot as plt
import torch

from utils import *
from utils_data_laisuo import *
from model import *

def _canon(v):
    if v is None:
        return "NA"
    try:
        if hasattr(v, "item"):
            v = v.item()
    except Exception:
        pass
    if isinstance(v, numbers.Integral):
        return str(int(v))
    if isinstance(v, numbers.Real):
        return f"{float(v):g}"
    return str(v)

def _get_any(hparams, model, keys_or_attrs, *, warn_name=""):
    for k in keys_or_attrs:
        if isinstance(hparams, dict) and k in hparams and hparams[k] is not None:
            return _canon(hparams[k])
    for k in keys_or_attrs:
        if hasattr(model, k) and getattr(model, k) is not None:
            return _canon(getattr(model, k))
    if warn_name:
        print(f"[WARN] Missing '{warn_name}' (tried {keys_or_attrs}); using NA in filenames.")
    return "NA"

def _as_float(x: torch.Tensor) -> torch.Tensor:
    if x is None:
        return x
    if torch.is_floating_point(x):
        return x
    return x.float()

@torch.no_grad()
def compute_normalization_stats(train_data_loader, device):
    xs, es, ys = [], [], []
    for data in train_data_loader:
        data = data.to(device)
        xs.append(_as_float(data.x))
        es.append(_as_float(data.edge_attr))
        y = _as_float(torch.as_tensor(data.y, device=device)).view(-1, 1)
        ys.append(y)

    if len(xs) == 0 or len(es) == 0 or len(ys) == 0:
        raise RuntimeError("Empty train loader encountered while computing normalization statistics.")

    x_all = torch.cat(xs, dim=0)
    e_all = torch.cat(es, dim=0)
    y_all = torch.cat(ys, dim=0)

    stats = {
        "x_mean": x_all.mean(dim=0),
        "x_std":  x_all.std(dim=0).clamp_min(1e-12),
        "e_mean": e_all.mean(dim=0),
        "e_std":  e_all.std(dim=0).clamp_min(1e-12),
        "y_mean": y_all.mean().view(()),
        "y_std":  y_all.std().clamp_min(1e-12).view(()),
    }
    return stats

def _normalize_inplace(data, stats):
    x0, e0, y0 = data.x, data.edge_attr, data.y

    x = _as_float(data.x)
    e = _as_float(data.edge_attr)
    y = _as_float(torch.as_tensor(data.y, device=x.device))

    data.x = (x - stats["x_mean"]) / stats["x_std"]
    data.edge_attr = (e - stats["e_mean"]) / stats["e_std"]
    data.y = (y - stats["y_mean"]) / stats["y_std"]
    return x0, e0, y0

def _restore_inplace(data, x0, e0, y0):
    data.x, data.edge_attr, data.y = x0, e0, y0

def _denorm_y(y_norm, stats):
    return y_norm * stats["y_std"] + stats["y_mean"]

def train_step(model, train_data_loader, optimizer, device, norm_stats=None):
    model.train()
    total_loss_raw = 0.0
    total_n = 0

    use_norm = norm_stats is not None

    for _, data in enumerate(train_data_loader):
        data = data.to(device)
        optimizer.zero_grad()

        if use_norm:
            x0, e0, y0 = _normalize_inplace(data, norm_stats)

        out = model(data)

        out1 = out.view(-1)
        y1 = _as_float(torch.as_tensor(data.y, device=device)).view(-1)

        if use_norm:
            loss_norm = (out1 - y1).abs().mean()
            loss_norm.backward()
            optimizer.step()

            loss_raw = loss_norm.detach() * norm_stats["y_std"]

            _restore_inplace(data, x0, e0, y0)
        else:
            loss_raw = (out1 - y1).abs().mean()
            loss_raw.backward()
            optimizer.step()

        n = y1.numel()
        total_loss_raw += float(loss_raw) * n
        total_n += n

    return model, total_loss_raw / float(total_n)

@torch.no_grad()
def test(model, data_loader, device, norm_stats=None):
    model.eval()
    total_error_raw = 0.0
    total_n = 0

    my_out = torch.empty((0, 1), device=device)
    my_y   = torch.empty((0,), device=device)

    use_norm = norm_stats is not None

    for data in data_loader:
        data = data.to(device)

        if use_norm:
            x0, e0, y0 = _normalize_inplace(data, norm_stats)
            out_norm = model(data)
            out_raw = _denorm_y(out_norm, norm_stats)

            y_raw = _as_float(torch.as_tensor(y0, device=device)).view(-1)

            _restore_inplace(data, x0, e0, y0)
        else:
            out_raw = model(data)
            y_raw = _as_float(torch.as_tensor(data.y, device=device)).view(-1)

        out_raw = out_raw.view(-1, 1)

        total_error_raw += (out_raw.view(-1) - y_raw).abs().sum().item()
        total_n += y_raw.numel()

        my_out = torch.cat((my_out, out_raw), dim=0)
        my_y   = torch.cat((my_y,   y_raw),  dim=0)

    loss = total_error_raw / float(total_n)
    return loss, my_out, my_y

def train(epochs, model, train_data_loader, val_data_loader, test_data_loader,
          optimizer, scheduler, device, disp_step, prefix, hyper_dict, dispProgress):

    model_type  = hyper_dict.get('model_type', 'MODEL')
    aggMLP_flag = str(hyper_dict.get('aggMLP', 'NA'))

    num_layer  = _get_any(hyper_dict, model, ["numLayers", "num_layer", "num_layers", "n_layers", "depth"], warn_name="num_layer")
    batch_size = _get_any(hyper_dict, model, ["batch_size", "bs"],                                           warn_name="batch_size")
    outchannel = _get_any(hyper_dict, model, ["out_channels", "outchannel", "hidden_dim", "hidden_channels", "emb_dim"], warn_name="outchannel")
    droprate   = _get_any(hyper_dict, model, ["dropout", "droprate", "drop_rate", "p_drop"],                warn_name="droprate")
    ds_name    = _get_any(hyper_dict, model, ["dataset_name"],                                              warn_name="dataset_name")
    gs         = _get_any(hyper_dict, model, ["group_size"],                                                warn_name="group_size")

    _agg_short = {'sum':'S', 'mean':'M', 'min':'m', 'max':'X', 'std':'V', 'var':'V'}
    agg_list = hyper_dict.get('aggregators', [])
    agg_tag = ''.join(_agg_short.get(a, a[0]) for a in agg_list) if agg_list else 'NA'

    name_suffix = f"_L{num_layer}_B{batch_size}_C{outchannel}_D{droprate}_DS{ds_name}_GS{gs}"
    folder = f'./Results/{model_type}_agg{agg_tag}_aggMLP={aggMLP_flag}{name_suffix}'
    create_folder_if_not_exists(folder)

    norm_stats = compute_normalization_stats(train_data_loader, device)

    print("========== Normalization parameters (train-set only) ==========")
    print(f"x_mean shape: {tuple(norm_stats['x_mean'].shape)}, x_std shape: {tuple(norm_stats['x_std'].shape)}")
    print(f"e_mean shape: {tuple(norm_stats['e_mean'].shape)}, e_std shape: {tuple(norm_stats['e_std'].shape)}")
    print(f"y_mean: {float(norm_stats['y_mean']):.10g}, y_std: {float(norm_stats['y_std']):.10g}")
    print("===============================================================")

    norm_filename = f"{prefix}_normalization{name_suffix}.json"
    if "save_normalization" in globals():
        save_normalization(folder, norm_filename, norm_stats)
    else:

        out = {k: v.detach().cpu().numpy().tolist() for k, v in norm_stats.items()}
        with open(osp.join(folder, norm_filename), "w") as f:
            json.dump(out, f, indent=2)

    if dispProgress:
        plt.figure(figsize=(8, 6))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs. Epoch during Training')
        plt.grid(True)

    train_losses, val_losses, test_losses = [], [], []
    epochs_list = []

    for epoch in range(1, epochs + 1):
        model, train_loss = train_step(model, train_data_loader, optimizer, device, norm_stats=norm_stats)
        val_loss,  val_out,  val_y  = test(model,  val_data_loader,  device, norm_stats=norm_stats)
        test_loss, test_out, test_y = test(model, test_data_loader, device, norm_stats=norm_stats)

        scheduler.step(val_loss)

        if epoch % disp_step == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch}, training loss(MAE)={round(train_loss, 6)}, "
                f"Validating loss(MAE)={round(val_loss, 6)}, "
                f"testing loss(MAE)={round(test_loss, 6)}, lr={round(current_lr, 8)}"
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_losses.append(test_loss)
            epochs_list.append(epoch)

            if dispProgress:
                plot_training_progress(epoch, train_losses, val_losses, test_losses, title=prefix)

        if epoch % 10 == 0:
            plt.close('all')
            regressionFile_mid   = f"{prefix}_ep{epoch}_regression{name_suffix}.txt"
            plotfilename_mid_png = f"{prefix}_ep{epoch}_regression{name_suffix}.png"

            save_regression_result(folder, regressionFile_mid, test_out, test_y)
            plot_regression_result(
                folder,
                filename=regressionFile_mid,
                title=f"{prefix}: Predicted vs. Truth @Epoch {epoch}",
                plotfilename=plotfilename_mid_png,
                disp=False
            )

    test_loss, test_out, test_y = test(model, test_data_loader, device, norm_stats=norm_stats)
    print('test_loss ', test_loss)

    maeFile = f"{prefix}_MAE{name_suffix}.txt"
    msg = f"{prefix}{name_suffix}"
    save_mae(folder=folder, maeFile=maeFile, test_loss=test_loss, msg=msg)

    progress_filename = f"{prefix}_training_progress{name_suffix}.txt"
    save_train_progress(
        folder,
        progress_filename,
        epochs=epochs_list,
        train_losses=train_losses,
        val_losses=val_losses,
        test_losses=test_losses
    )

    plotfilename = f"{prefix}_progress_plotting{name_suffix}.png"
    plot_training_from_file(
        folder,
        filename=progress_filename,
        plotfilename=plotfilename,
        disp=False
    )

    if getattr(model, 'aggMLP', True) is False:
        agg_weight_filename = f"{prefix}_agg_weights{name_suffix}.txt"
        save_agg_weights(folder, filename=agg_weight_filename, model=model)

    regressionFile_final   = f"{prefix}_regression{name_suffix}.txt"
    plotfilename_reg_final = f"{prefix}_regression{name_suffix}.png"
    save_regression_result(folder, regressionFile_final, test_out, test_y)
    plot_regression_result(
        folder,
        filename=regressionFile_final,
        title=f"{prefix}: Predicted vs. Truth (Test)",
        plotfilename=plotfilename_reg_final,
        disp=False
    )

    model_filename      = f"{prefix}_model{name_suffix}.pt"
    model_para_filename = f"{prefix}_model_para{name_suffix}.json"
    if isinstance(model, MyCEALNetwork):
        save_model_CEAL(model, folder, model_filename, model_para_filename)
    elif isinstance(model, GCNNetwork):
        save_model_GCN(model, folder, model_filename, model_para_filename)
    elif isinstance(model, PNANetwork):
        save_model_PNA(model, folder, model_filename, model_para_filename)
    else:
        torch.save(model.state_dict(), osp.join(folder, model_filename))

    hyper_para_filename = f"{prefix}_hyper_parameters{name_suffix}.json"
    save_hyper_parameter(folder, hyper_para_filename, hyper_dict)

    print('Done save_hyper_parameter')
    print('DONE')
