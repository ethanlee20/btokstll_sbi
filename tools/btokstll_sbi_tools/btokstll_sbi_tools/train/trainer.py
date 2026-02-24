
from pathlib import Path
import json
from dataclasses import asdict
from types import NoneType

from torch import Tensor, no_grad, save
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from ..util import get_model_current_device, Dataset
from .data_loader import Data_Loader
from .hyperparams import (
    Adam_Hyperparams,
    CrossEntropyLoss_Hyperparams, 
    ReduceLROnPlateau_Hyperparams,
    Hyperparams
)
from .loss_table import Loss_Table


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
    
    cumulative_batch_loss = 0
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
    
    cumulative_batch_loss = 0
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


def _save_torch_model(
    model:Module, 
    path:Path|str,
):
    save(model.state_dict(), path)


_available_optimizers = {
    Adam_Hyperparams: Adam,
}


_available_loss_fns = {
    CrossEntropyLoss_Hyperparams: CrossEntropyLoss,
}


_available_lr_schedulers = {
    ReduceLROnPlateau_Hyperparams: ReduceLROnPlateau, 
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
    
    model = model.to(device)
    
    if loss_table is None:
        loss_table = Loss_Table()

    if verbose:
        print('\n')

    for ep in hyperparams.epochs:

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


# class Trainer:

#     def __init__(
#         self, 
#         train_dataset:Dataset, 
#         eval_dataset:Dataset, 
#         model:Module, 
#         params:dict
#     ):
#         self.params = params
#         self.model = model

#         self.datasets = {"train": train_dataset, "eval": eval_dataset}
#         self.dataloaders = {
#             split : Data_Loader(dataset, self.params["batch_sizes"][split], shuffle=True) 
#             for split, dataset in self.datasets.items()
#         }

#         self.optimizer = self.available_optimizers[self.params["optimizer"]](
#             self.model.parameters(),
#             **self.params["optimizer_params"] 
#         )

#         self.loss_fn = self.available_loss_fns[self.params["loss_fn"]](**self.params["loss_fn_params"])

#         self.lr_scheduler = self.available_lr_schedulers[self.params["lr_scheduler"]]
#         if self.lr_scheduler is not None: 
#             self.lr_scheduler = self.lr_scheduler(
#                 self.optimizer, 
#                 **self.params["lr_scheduler_params"]
#             )

#         self.loss_table = Loss_Table()

#         self.setup_save_dir()

#         self.save_params_json()

#     def print_current_epoch_loss(self):
        
#         row = self.loss_table.get_last_row()
#         print(
#             f"\nEpoch {row["epoch"]} complete:\n"
#             f"    Train loss: {row["train_loss"]}\n"
#             f"    Eval loss: {row["eval_loss"]}\n"
#         )

#     def print_current_learning_rate(self):

#         if self.lr_scheduler is not None:
#             print(f"Learning rate: {self.lr_scheduler.get_last_lr()}")
#         else: 
#             print(self.params["optimizer_params"]["lr"])

#     def save_dir(self):

#         path = Path(self.params["parent_dir"]).joinpath(self.params["name"])
#         return path

#     def setup_save_dir(self, no_overwrite=True):

#         if self.save_dir().exists() and no_overwrite:
#             raise ValueError(f"Save location exists (delete to continue): {self.save_dir()}")
#         self.save_dir().mkdir()

#         checkpoints_dir = self.save_dir().joinpath("checkpoints")
#         checkpoints_dir.mkdir()

#     def save_params_json(self):

#         params_to_save = self.params.copy()

#         try: params_to_save["loss_fn_params"]["weight"] = params_to_save["loss_fn_params"]["weight"].tolist()
#         except KeyError: pass

#         path = self.save_dir().joinpath("params.json")
#         with open(path, "x") as file:
#             json.dump(params_to_save, file, sort_keys=False, indent=4)

#     def save_loss_table(self):

#         path = self.save_dir().joinpath("loss_table.jsonl")
#         self.loss_table.save_table_as_jsonl(path)

#     def save_model(self, epoch):

#         path = (
#             self.save_dir().joinpath(f"checkpoints/epoch_{epoch}.pt")
#             if epoch < self.params["epochs"] - 1
#             else self.save_dir().joinpath("final.pt")
#         )
#         _save_torch_model(self.model, path)

#     def plot_loss(self, yscale="log"): 

#         _, ax = plt.subplots()

#         for losses, label in zip(
#             [self.loss_table.train_losses, self.loss_table.eval_losses], 
#             ["train", "eval"]
#         ):
#             ax.plot(self.loss_table.epochs, losses, label=label)
        
#         ax.set_xlabel("Epoch")
#         ax.set_yscale(yscale)
#         ax.set_ylabel(f"Loss ({self.params["loss_fn"]})")
#         ax.legend()
#         plt.savefig(self.save_dir().joinpath("loss.png"), bbox_inches="tight")
#         plt.close()

#     def train(self, device, verbosity=1):
    
#         self.model = self.model.to(device)

#         for epoch in range(self.params["epochs"]):
        
#             train_loss, eval_loss = _train_evaluate_epoch(
#                 self.dataloaders["train"],
#                 self.dataloaders["eval"],
#                 self.model,
#                 self.loss_fn,
#                 self.optimizer,
#                 self.lr_scheduler
#             )

#             self.loss_table.add_to_table(epoch, train_loss.item(), eval_loss.item())
        
#             if verbosity >= 1:
#                 self.print_current_epoch_loss()
#                 self.print_current_learning_rate()

#             if (epoch % self.params["checkpoint_epochs"]) == 0:

#                 self.save_model(epoch)
#                 self.save_loss_table()
#                 self.plot_loss()
            
#         self.save_model(epoch)
#         self.save_loss_table()
#         self.plot_loss()
    
#         if verbosity >= 1:    
#             print("Training completed.")


