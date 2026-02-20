
from pathlib import Path
import json

import torch
import pandas
import matplotlib.pyplot as plt

from ..util.hardware import get_model_current_device
from ..util.save_load import save_torch_model
from ..util.types import to_torch_tensor
from .data_loading import Data_Loader


def train_batch(x, y, model, loss_fn, optimizer):
    
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
    

def evaluate_batch(x, y, model, loss_fn):

    device = get_model_current_device(model)
    x = x.to(device)
    y = y.to(device)
    
    model.eval()
    with torch.no_grad():
        yhat = model(x)
        eval_loss = loss_fn(yhat, y)
    return eval_loss


def train_epoch(dataloader, model, loss_fn, optimizer):
    
    cumulative_batch_loss = 0
    for x, y in dataloader:
        batch_loss = train_batch(x, y, model, loss_fn, optimizer)
        cumulative_batch_loss += batch_loss

    num_batches = len(dataloader)
    avg_batch_loss = cumulative_batch_loss / num_batches
    return avg_batch_loss


def evaluate_epoch(dataloader, model, loss_fn, scheduler=None):
    
    cumulative_batch_loss = 0
    for x, y in dataloader:
        batch_loss = evaluate_batch(x, y, model, loss_fn)
        cumulative_batch_loss += batch_loss
    
    num_batches = len(dataloader)
    avg_batch_loss = cumulative_batch_loss / num_batches
    
    if scheduler:
        scheduler.step(avg_batch_loss)
    
    return avg_batch_loss


def train_evaluate_epoch(
    dataloader_train, 
    dataloader_eval, 
    model, 
    loss_fn,
    optimizer,
    lr_scheduler,
):
        
    train_loss = train_epoch(
        dataloader_train, 
        model, 
        loss_fn, 
        optimizer,
    )

    eval_loss = evaluate_epoch(
        dataloader_eval, 
        model, 
        loss_fn, 
        scheduler=lr_scheduler
    )

    return train_loss, eval_loss


class Loss_Table:

    def __init__(self):
    
        self.epochs = []
        self.train_losses = []
        self.eval_losses = []

    def add_to_table(self, epoch, train_loss, eval_loss):

        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.eval_losses.append(eval_loss)

    def get_last_row(self): 

        return {
            "epoch": self.epochs[-1], 
            "train_loss": self.train_losses[-1], 
            "eval_loss": self.eval_losses[-1]
        }

    def save_table_as_jsonl(self, path):

        pandas.DataFrame(
            {
                "epochs": self.epochs, 
                "train_losses": self.train_losses, 
                "eval_losses": self.eval_losses
            }
        ).to_json(path, orient="records", lines=True)


class Trainer:

    available_optimizers = {"adam": torch.optim.Adam}
    available_loss_fns = {"mse": torch.nn.MSELoss, "cross_entropy": torch.nn.CrossEntropyLoss}
    available_lr_schedulers = {"reduce_lr_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau, "none":None}

    def __init__(self, dataset_train, dataset_eval, model, params):

        self.params = params

        self.model = model

        self.datasets = {"train": dataset_train, "eval": dataset_eval}
        self.dataloaders = {
            split : Data_Loader(dataset, self.params["batch_sizes"][split], shuffle=True) 
            for split, dataset in self.datasets.items()
        }

        self.optimizer = self.available_optimizers[self.params["optimizer"]](
            self.model.parameters(),
            **self.params["optimizer_params"] 
        )

        self.loss_fn = self.available_loss_fns[self.params["loss_fn"]](**self.params["loss_fn_params"])

        self.lr_scheduler = self.available_lr_schedulers[self.params["lr_scheduler"]]
        if self.lr_scheduler is not None: 
            self.lr_scheduler = self.lr_scheduler(
                self.optimizer, 
                **self.params["lr_scheduler_params"]
            )

        self.loss_table = Loss_Table()

        self.setup_save_dir()

        self.save_params_json()

    def print_current_epoch_loss(self):
        
        row = self.loss_table.get_last_row()
        print(
            f"\nEpoch {row["epoch"]} complete:\n"
            f"    Train loss: {row["train_loss"]}\n"
            f"    Eval loss: {row["eval_loss"]}\n"
        )

    def print_current_learning_rate(self):

        if self.lr_scheduler is not None:
            print(f"Learning rate: {self.lr_scheduler.get_last_lr()}")
        else: 
            print(self.params["optimizer_params"]["lr"])

    def save_dir(self):

        path = Path(self.params["parent_dir"]).joinpath(self.params["name"])
        return path

    def setup_save_dir(self, no_overwrite=True):

        if self.save_dir().exists() and no_overwrite:
            raise ValueError(f"Save location exists (delete to continue): {self.save_dir()}")
        self.save_dir().mkdir()

        checkpoints_dir = self.save_dir().joinpath("checkpoints")
        checkpoints_dir.mkdir()

    def save_params_json(self):

        params_to_save = self.params.copy()

        try: params_to_save["loss_fn_params"]["weight"] = params_to_save["loss_fn_params"]["weight"].tolist()
        except KeyError: pass

        path = self.save_dir().joinpath("params.json")
        with open(path, "x") as file:
            json.dump(params_to_save, file, sort_keys=False, indent=4)

    def save_loss_table(self):

        path = self.save_dir().joinpath("loss_table.jsonl")
        self.loss_table.save_table_as_jsonl(path)

    def save_model(self, epoch):

        path = (
            self.save_dir().joinpath(f"checkpoints/epoch_{epoch}.pt")
            if epoch < self.params["epochs"] - 1
            else self.save_dir().joinpath("final.pt")
        )
        save_torch_model(self.model, path)

    def plot_loss(self, yscale="log"): 

        _, ax = plt.subplots()

        for losses, label in zip(
            [self.loss_table.train_losses, self.loss_table.eval_losses], 
            ["train", "eval"]
        ):
            ax.plot(self.loss_table.epochs, losses, label=label)
        
        ax.set_xlabel("Epoch")
        ax.set_yscale(yscale)
        ax.set_ylabel(f"Loss ({self.params["loss_fn"]})")
        ax.legend()
        plt.savefig(self.save_dir().joinpath("loss.png"), bbox_inches="tight")
        plt.close()

    def train(self, device, verbosity=1):
    
        self.model = self.model.to(device)

        for epoch in range(self.params["epochs"]):
        
            train_loss, eval_loss = train_evaluate_epoch(
                self.dataloaders["train"],
                self.dataloaders["eval"],
                self.model,
                self.loss_fn,
                self.optimizer,
                self.lr_scheduler
            )

            self.loss_table.add_to_table(epoch, train_loss.item(), eval_loss.item())
        
            if verbosity >= 1:
                self.print_current_epoch_loss()
                self.print_current_learning_rate()

            if (epoch % self.params["checkpoint_epochs"]) == 0:

                self.save_model(epoch)
                self.save_loss_table()
                self.plot_loss()
            
        self.save_model(epoch)
        self.save_loss_table()
        self.plot_loss()
    
        if verbosity >= 1:    
            print("Training completed.")


def calculate_class_weights_for_uniform_prior(labels:pandas.Series):

    """Calculate class weights for reweighting classes to uniform distribution."""

    normalized_class_counts = to_torch_tensor(
        labels.value_counts(normalize=True).sort_index()
    )
    weights = 1 / normalized_class_counts
    return weights