from pandas import concat, DataFrame, Series
from torch import Tensor, zeros, no_grad
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .util import pandas_from_tensor, get_model_device
from .dataloader import DataLoader
from .dataset import Dataset


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


def _evaluate_batch(
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
    data_loader: DataLoader,
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


def _evaluate_epoch(data_loader: DataLoader, model: Module, loss_fn) -> Tensor:

    device = get_model_device(model)

    cum_loss = zeros(1).to(device)
    for features, labels in data_loader:
        batch_loss = _evaluate_batch(features, labels, model, loss_fn)
        cum_loss += batch_loss

    num_batches = len(data_loader)
    avg_batch_loss = cum_loss / num_batches

    return avg_batch_loss


def _run_training_epoch(
    train_data_loader: DataLoader,
    model: Module,
    loss_fn,
    optimizer: Optimizer,
    eval_data_loader: DataLoader | None = None,
    lr_scheduler: None | LRScheduler = None,
) -> DataFrame | Series:

    train_loss = _train_epoch(
        train_data_loader,
        model,
        loss_fn,
        optimizer,
    )
    train_loss_series = pandas_from_tensor(train_loss, names="train")

    run_eval = eval_data_loader is not None
    if run_eval:
        eval_loss = _evaluate_epoch(
            eval_data_loader,
            model,
            loss_fn,
        )
        eval_loss_series = pandas_from_tensor(eval_loss, names="eval")
    if lr_scheduler is not None: 
        lr_scheduler.step()
    
    loss_table = (
        train_loss_series
        if not run_eval
        else concat([train_loss_series, eval_loss_series], axis=1)
    )
    return loss_table


def run_training(
    train_data_loader: DataLoader,
    model: Module,
    loss_fn,
    optimizer: Optimizer,
    num_epochs: int,
    eval_data_loader: DataLoader | None = None,
    lr_scheduler: None | LRScheduler = None,
) -> DataFrame | Series:
    losses = []
    for _epoch in range(num_epochs):
        epoch_losses = _run_training_epoch(
            train_data_loader=train_data_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            eval_data_loader=eval_data_loader,
            lr_scheduler=lr_scheduler,
        )
        losses.append(epoch_losses)
    losses = concat(losses, ignore_index=True)
    return losses


def run_training_on_datasets(
    train_dataset: Dataset,
    train_batch_size: int,
    model: Module,
    loss_fn,
    optimizer: Optimizer,
    num_epochs: int,
    eval_dataset: Dataset | None = None,
    eval_batch_size: int | None = None,
    lr_scheduler: None | LRScheduler = None
) -> DataFrame | Series:
    train_data_loader = DataLoader(train_dataset, train_batch_size)
    eval_data_loader = (
        DataLoader(eval_dataset, eval_batch_size) if eval_dataset is not None else None
    )
    losses = run_training(
        train_data_loader,
        model,
        loss_fn,
        optimizer,
        num_epochs,
        eval_data_loader=eval_data_loader,
        lr_scheduler=lr_scheduler
    )
    return losses
