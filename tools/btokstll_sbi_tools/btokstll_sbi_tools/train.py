
from types import NoneType
from dataclasses import dataclass, asdict, field
from math import floor

from torch import (
    Tensor, 
    no_grad, 
    arange, 
    randperm, 
    reshape, 
    bincount, 
    sum,
    equal,
)
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer, Adam, AdamW
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau, CosineAnnealingLR

from .interval import are_instance, Interval
from .type import are_instance
from .json_ import dump_json, load_json
from .dataset import Dataset
from .hardware import get_model_current_device





def _generate_batched_indices(
    dataset_size:int, 
    batch_size:int, 
    shuffle:bool,
):
    assert are_instance([dataset_size, batch_size], int)
    assert isinstance(shuffle, bool)
    assert dataset_size > batch_size

    indices = arange(dataset_size)
    if shuffle: 
        indices = indices[randperm(len(indices))]
    num_batches = floor(dataset_size / batch_size)
    batched_indices = reshape(
        indices[:num_batches*batch_size], 
        shape=(num_batches, batch_size)
    )
    return batched_indices


class Data_Loader:

    def __init__(
        self,
        dataset:Dataset,
        batch_size:int,
        shuffle:bool,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.dataset_size = len(self.dataset)
        self.batched_indices = _generate_batched_indices(
            self.dataset_size, 
            self.batch_size, 
            self.shuffle
        )

    def __len__(
        self,
    ):
        return len(self.batched_indices)
    
    def __iter__(
        self,
    ):
        self.index = 0
        return self
    
    def __next__(
        self,
    ):
        if self.index >= len(self):
            self.batched_indices = _generate_batched_indices(
                self.dataset_size, 
                self.batch_size, 
                self.shuffle
            )
            raise StopIteration
        
        batch_indices = self.batched_indices[self.index]
        batch_features = (
            None if self.dataset.features is None 
            else self.dataset.features[batch_indices]
        )
        batch_labels = (
            None if self.dataset.labels is None 
            else self.dataset.labels[batch_indices]
        )

        self.index += 1

        return batch_features, batch_labels
    



@dataclass
class Adam_Hyperparams:
    lr: float = 0.001


@dataclass
class AdamW_Hyperparams:
    lr: float = 0.001
    weight_decay: float = 1e-2


@dataclass
class CrossEntropyLoss_Hyperparams:
    weight: Tensor|None = None


@dataclass
class ReduceLROnPlateau_Hyperparams:
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    eps: float = 1e-8 


@dataclass
class CosineAnnealingLR_Hyperparams:
    T_max: int = 0
    eta_min: float = 0.0
    last_epoch: int = -1


@dataclass
class Hyperparams:
    optimizer: Adam_Hyperparams | AdamW_Hyperparams = field(default_factory=AdamW_Hyperparams)
    train_batch_size: int = 32
    eval_batch_size: int = 32
    shuffle: bool = True
    epochs: Interval = field(default_factory=Interval)
    loss_fn: CrossEntropyLoss_Hyperparams = field(default_factory=CrossEntropyLoss_Hyperparams)
    lr_scheduler: ReduceLROnPlateau_Hyperparams|CosineAnnealingLR_Hyperparams|None = None
    num_bins: int = 8
    binned_interval: Interval = field(default_factory=Interval)



class Loss_Table:
    def __init__(
        self
    ):
        self.table = {}

    def add_to_table(
        self, 
        epoch:int, 
        train_loss:Tensor, 
        eval_loss:Tensor
    ):
        self.table[epoch] = dict(
            train=train_loss.item(), 
            eval=eval_loss.item()
        )

    def get_losses(
        self,
        epoch,
    ):
        return self.table[epoch]

    def save_table_as_json(
        self, 
        path,
    ):
        dump_json(self.table, path)

    def load_table_from_json(
        self,
        path,
    ):
        self.table = load_json(path)

    def as_lists(
        self,
    ) -> dict:
        epochs = []
        train_losses = []
        eval_losses = []
        for epoch, losses in self.table.items():
            epochs.append(epoch)
            train_losses.append(losses["train"])
            eval_losses.append(losses["eval"]) 
        return dict(
            epochs=epochs, 
            train=train_losses, 
            eval=eval_losses
        )
    





def _train_batch(
    x:Tensor, 
    y:Tensor, 
    model:Module, 
    loss_fn, 
    optimizer:Optimizer,
) -> Tensor:
    
    device = get_model_current_device(model)
    x = x.to(device)
    y = y.to(device)

    model.train()
    yhat = model(x)    
    train_loss = loss_fn(yhat, y)
    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return train_loss
    

def _evaluate_batch(
    x:Tensor, 
    y:Tensor, 
    model:Module, 
    loss_fn,
) -> Tensor:
    
    device = get_model_current_device(model)
    x = x.to(device)
    y = y.to(device)
    
    model.eval()
    with no_grad():
        yhat = model(x)
        eval_loss = loss_fn(yhat, y)
    return eval_loss


def _train_epoch(
    data_loader:Data_Loader, 
    model:Module, 
    loss_fn, 
    optimizer:Optimizer,
) -> Tensor:
    
    device = get_model_current_device(model)
    
    cumulative_batch_loss = Tensor([0]).to(device)
    for x, y in data_loader:
        batch_loss = _train_batch(x, y, model, loss_fn, optimizer)
        cumulative_batch_loss += batch_loss

    num_batches = len(data_loader)
    avg_batch_loss = cumulative_batch_loss / num_batches
    return avg_batch_loss


def _evaluate_epoch(
    data_loader:Data_Loader, 
    model:Module, 
    loss_fn, 
    scheduler:LRScheduler|None=None
) -> Tensor:
    
    device = get_model_current_device(model)
    
    cumulative_batch_loss = Tensor([0]).to(device)
    for x, y in data_loader:
        batch_loss = _evaluate_batch(x, y, model, loss_fn)
        cumulative_batch_loss += batch_loss
    
    num_batches = len(data_loader)
    avg_batch_loss = cumulative_batch_loss / num_batches
    
    if scheduler is not None:
        scheduler.step(avg_batch_loss)
    
    return avg_batch_loss


def _train_evaluate_epoch(
    train_data_loader: Data_Loader, 
    eval_data_loader: Data_Loader, 
    model: Module, 
    loss_fn,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler|None = None,
) -> tuple[Tensor, Tensor]:
    
    train_loss = _train_epoch(
        train_data_loader, 
        model, 
        loss_fn, 
        optimizer,
    )

    eval_loss = _evaluate_epoch(
        eval_data_loader, 
        model, 
        loss_fn, 
        scheduler=lr_scheduler
    )

    return train_loss, eval_loss


_available_optimizers = {
    Adam_Hyperparams: Adam,
    AdamW_Hyperparams : AdamW,
}


_available_loss_fns = {
    CrossEntropyLoss_Hyperparams: CrossEntropyLoss,
}


_available_lr_schedulers = {
    ReduceLROnPlateau_Hyperparams: ReduceLROnPlateau, 
    CosineAnnealingLR_Hyperparams: CosineAnnealingLR,
    NoneType: None,
}


def train(
    model: Module,
    hyperparams: Hyperparams,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    device: str,
    loss_table: Loss_Table|None = None,
    verbose: bool = True,  
) -> Loss_Table:
    
    model = model.to(device)
    train_dataset.to(device)
    eval_dataset.to(device)
    
    optimizer_class = _available_optimizers[
        type(hyperparams.optimizer)
    ]
    optimizer = optimizer_class(
        model.parameters(), 
        **asdict(hyperparams.optimizer)
    )

    loss_fn_class = _available_loss_fns[
        type(hyperparams.loss_fn)
    ]
    loss_fn = loss_fn_class(
        **asdict(hyperparams.loss_fn)
    )

    lr_scheduler_class = _available_lr_schedulers[
        type(hyperparams.lr_scheduler)
    ]
    lr_scheduler = (
        lr_scheduler_class(
            optimizer=optimizer,
            **asdict(hyperparams.lr_scheduler)
        ) if lr_scheduler_class is not None
        else None
    )
    
    train_data_loader = Data_Loader(
        train_dataset, 
        hyperparams.train_batch_size, 
        hyperparams.shuffle
    )
    eval_data_loader = Data_Loader(
        eval_dataset, 
        hyperparams.eval_batch_size, 
        hyperparams.shuffle
    )
    
    if loss_table is None:
        loss_table = Loss_Table()

    if verbose:
        print('\n')

    for ep in range(*hyperparams.epochs):

        train_loss, eval_loss = _train_evaluate_epoch(
            train_data_loader, 
            eval_data_loader, 
            model, 
            loss_fn,
            optimizer,
            lr_scheduler=lr_scheduler,
        )
        loss_table.add_to_table(
            ep, 
            train_loss=train_loss, 
            eval_loss=eval_loss
        )
        if verbose:
            print(f"Epoch {ep}:")
            print(f"Train loss: {train_loss.item()}")
            print(f"Eval loss: {eval_loss.item()}")
            print('\n')

    if verbose:
        print("Training completed.\n")

    return loss_table