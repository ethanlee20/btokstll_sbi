
from dataclasses import dataclass

from torch import Tensor


@dataclass
class Adam_Hyperparams:
    lr: float = 0.001


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
class Hyperparams:
    optimizer: Adam_Hyperparams
    train_batch_size: int
    eval_batch_size: int
    shuffle: bool
    epochs: range[int]
    loss_fn: CrossEntropyLoss_Hyperparams
    lr_scheduler: ReduceLROnPlateau_Hyperparams|None
    num_bins: int
    binned_interval_left: float
    binned_interval_right: float

