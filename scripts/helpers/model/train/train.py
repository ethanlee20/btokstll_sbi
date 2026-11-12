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
    parameters: Tensor,
    features: Tensor,
    labels: Tensor,
    model: Module,
    loss_fn,
    optimizer: Optimizer,
) -> Tensor:

    device = get_model_device(model)

    parameters = parameters.to(device)
    features = features.to(device)
    labels = labels.to(device)

    model.train()
    predictions = model(
        parameters=parameters, 
        features=features
    )
    train_loss = loss_fn(predictions, labels)
    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return train_loss


def _val_batch(
    parameters: Tensor,
    features: Tensor,
    labels: Tensor,
    model: Module,
    loss_fn,
) -> Tensor:

    device = get_model_device(model)

    parameters = parameters.to(device)
    features = features.to(device)
    labels = labels.to(device)

    model.eval()
    with no_grad():
        predictions = model(
            parameters=parameters, 
            features=features
        )
        eval_loss = loss_fn(predictions, labels)
    return eval_loss


def _train_epoch(
    dataloader: NreCDataLoader,
    model: Module,
    loss_fn,
    optimizer: Optimizer,
) -> Tensor:

    device = get_model_device(model)

    cum_loss = zeros(1)
    cum_loss = cum_loss.to(device)
    for parameters, features, labels in dataloader:
        batch_loss = _train_batch(
            parameters,
            features,
            labels,
            model,
            loss_fn,
            optimizer,
        )
        cum_loss += batch_loss

    num_batches = len(dataloader)
    avg_batch_loss = cum_loss / num_batches

    return avg_batch_loss


def _val_epoch(
    dataloader: NreCDataLoader, 
    model: Module, 
    loss_fn,
) -> Tensor:

    device = get_model_device(model)

    cum_loss = zeros(1).to(device)
    for parameters, features, labels in dataloader:
        batch_loss = _val_batch(parameters, features, labels, model, loss_fn)
        cum_loss += batch_loss

    num_batches = len(dataloader)
    avg_batch_loss = cum_loss / num_batches
    return avg_batch_loss


def _train_val_epoch(
    train_dataloader: NreCDataLoader,
    model: Module,
    loss_fn,
    optimizer: Optimizer,
    val_dataloader: SetDataLoader | None = None,
    lr_scheduler: None | LRScheduler = None,
) -> DataFrame | Series:

    train_loss = _train_epoch(
        train_dataloader,
        model,
        loss_fn,
        optimizer,
    )
    train_loss_series = pandas_from_tensor(train_loss, names="train")

    run_val = val_dataloader is not None
    if run_val:
        val_loss = _val_epoch(
            val_dataloader,
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
    train_dataloader: NreCDataLoader,
    model: Module,
    loss_fn,
    optimizer: Optimizer,
    num_epochs: int,
    val_dataloader: SetDataLoader | None = None,
    lr_scheduler: None | LRScheduler = None,
) -> DataFrame | Series:

    losses = []
    for ep in range(num_epochs):
        epoch_losses = _train_val_epoch(
            train_dataloader=train_dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            val_dataloader=val_dataloader,
            lr_scheduler=lr_scheduler,
        )
        losses.append(epoch_losses)
        if val_dataloader is not None:
            print(
                f"Epoch {ep}: Train: {epoch_losses['train'].item():.4g}," 
                f" Val: {epoch_losses['val'].item():.4g}"
            )
        else:  
            print(f"Epoch {ep}: Train: {epoch_losses.item():.4g}")

    losses = concat(losses, ignore_index=True)
    return losses


def plot_loss(
    ax: Axes,
    train_losses: ArrayLike,
    val_losses: ArrayLike | None = None,
    yscale: str = "linear",
    scatter_or_plot: str = "scatter",
    compute_log: bool = False,
    **kwargs,
):
    if scatter_or_plot not in ("scatter", "plot"):
        raise ValueError(
            "scatter_or_plot should be 'scatter' or 'plot'." f"\nGot: {scatter_or_plot}"
        )

    if val_losses is not None:
        if len(val_losses) != len(train_losses):
            raise ValueError

    plot_fn = ax.scatter if scatter_or_plot == "scatter" else ax.plot if scatter_or_plot == "plot" else None
    if plot_fn is None:
        raise ValueError

    epochs = arange(len(train_losses))

    if compute_log:
        train_losses = log10(train_losses)
    if compute_log and val_losses is not None:
        val_losses = log10(val_losses)

    plot_fn(epochs, train_losses, label="Train", **kwargs)

    if val_losses is not None:
        plot_fn(epochs, val_losses, label="Eval.", **kwargs)

    ax.set_yscale(yscale)
    ax.legend()
    ax.set_ylabel("Log Loss" if compute_log else "Loss", fontsize=15)
    ax.set_xlabel("Epoch", fontsize=15)
