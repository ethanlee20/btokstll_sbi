
from json import dump, load

from torch import Tensor


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
        with open(path, 'x') as f:
            dump(self.table, f, indent=4)

    def load_table_from_json(
        self,
        path
    ):
        with open(path, 'r') as f:
            self.table = load(f)

    def as_lists(
        self,
    ) -> dict[str:list, str:list, str:list]:
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