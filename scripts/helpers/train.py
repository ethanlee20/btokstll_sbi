from tqdm import tqdm
from pandas import concat, DataFrame, Series
from torch import Tensor, zeros, no_grad
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .util import pandas_from_tensor, get_model_device
from .dataloader import SetDataLoader
from .dataset import Dataset


import pandas
import matplotlib
import numpy
import torch


def _train_batch(
    features: Tensor,
    labels: Tensor,
    model: Module,
    loss_fn,
    optimizer: Optimizer,
) -> Tensor:

    device = get_model_device(model)

    features = features.to(device)
    labels = labels.to(device)

    model.train()
    predictions = model(features)
    train_loss = loss_fn(predictions, labels)
    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return train_loss


def _val_batch(
    features: Tensor,
    labels: Tensor,
    model: Module,
    loss_fn,
) -> Tensor:

    device = get_model_device(model)

    features = features.to(device)
    labels = labels.to(device)

    model.eval()
    with no_grad():
        predictions = model(features)
        eval_loss = loss_fn(predictions, labels)
    return eval_loss


def _train_epoch(
    data_loader: SetDataLoader,
    model: Module,
    loss_fn,
    optimizer: Optimizer,
) -> Tensor:

    device = get_model_device(model)

    cum_loss = zeros(1)
    cum_loss = cum_loss.to(device)
    for features, labels in data_loader:
        batch_loss = _train_batch(
            features,
            labels,
            model,
            loss_fn,
            optimizer,
        )
        cum_loss += batch_loss

    num_batches = len(data_loader)
    avg_batch_loss = cum_loss / num_batches

    return avg_batch_loss


def _val_epoch(
    data_loader: SetDataLoader, 
    model: Module, 
    loss_fn,
) -> Tensor:

    device = get_model_device(model)

    cum_loss = zeros(1).to(device)
    for features, labels in data_loader:
        batch_loss = _val_batch(features, labels, model, loss_fn)
        cum_loss += batch_loss

    num_batches = len(data_loader)
    avg_batch_loss = cum_loss / num_batches
    return avg_batch_loss


def _train_val_epoch(
    train_data_loader: SetDataLoader,
    model: Module,
    loss_fn,
    optimizer: Optimizer,
    val_data_loader: SetDataLoader | None = None,
    lr_scheduler: None | LRScheduler = None,
) -> DataFrame | Series:

    train_loss = _train_epoch(
        train_data_loader,
        model,
        loss_fn,
        optimizer,
    )
    train_loss_series = pandas_from_tensor(train_loss, names="train")

    run_val = val_data_loader is not None
    if run_val:
        val_loss = _val_epoch(
            val_data_loader,
            model,
            loss_fn,
        )
        val_loss_series = pandas_from_tensor(val_loss, names="val")
    if lr_scheduler is not None:
        lr_scheduler.step()

    loss_table = (
        train_loss_series
        if not run_val
        else concat([train_loss_series, val_loss_series], axis=1)
    )
    return loss_table


def train_val(
    train_data_loader: SetDataLoader,
    model: Module,
    loss_fn,
    optimizer: Optimizer,
    num_epochs: int,
    val_data_loader: SetDataLoader | None = None,
    lr_scheduler: None | LRScheduler = None,
) -> DataFrame | Series:

    losses = []
    for ep in range(num_epochs):
        epoch_losses = _train_val_epoch(
            train_data_loader=train_data_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            val_data_loader=val_data_loader,
            lr_scheduler=lr_scheduler,
        )
        losses.append(epoch_losses)
        if val_data_loader is not None:
            print(f"Epoch {ep}: Train: {epoch_losses['train'].item():.4g}, Val: {epoch_losses['val'].item():.4g}")
        else:  
            print(f"Epoch {ep}: Train: {epoch_losses.item():.4g}")

    losses = concat(losses, ignore_index=True)
    return losses


"""
        labels = []
        mles = []
        confs = []
        for trial_num in range(100):
            df_val_set = df_val_sets.xs(trial_num, level="trial_num")
            label = df_val_set["dC_9"].unique()
            #if label <= 1 and label >= -2:
            features = df_val_set[feature_cols]
            mle, conf = get_mle_and_conf(model, features, -5, 2, 100)
            labels.append(label)
            mles.append(mle)
            confs.append(conf)
        labels = numpy.array(labels)
        mles = numpy.array(mles)
        confs = numpy.array(confs)

        fig, ax = matplotlib.pyplot.subplots()
        ax.plot([-5, 2], [-5, 2], color="#377eb8", label="Ideal", zorder=-100)
        ax.scatter(labels, mles, color="#ff7f00")
        ax.vlines(labels, confs[:,0], confs[:, 1], color='#4daf4a', linestyle="-", zorder=-101, label=r"Log Likelihood $1\sigma$")
        ax.set_ylabel(r"Predicted $\delta C_9$", fontsize=16)
        ax.set_xlabel(r"True $\delta C_9$", fontsize=16)
        ax.legend()
        fig.savefig("plots/lin.png", bbox_inches="tight")
        matplotlib.pyplot.close()
"""