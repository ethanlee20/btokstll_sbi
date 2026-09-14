
from torch import Tensor, sum, ones, zeros
from torch.nn import (
    Module, 
    ReLU, 
    Linear, 
    Dropout, 
    Softmax, 
    Sequential
)
from torch.nn.parameter import Buffer


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.layers = Sequential(
            Linear(5, 100),
            ReLU(),
            Linear(100, 100),
            ReLU(),
            Linear(100, 100),
            ReLU(),
            Linear(100, 1),
        )
        self.train_means = Buffer(zeros(5))
        self.train_stds = Buffer(ones(5))

    def forward(self, x):
        rescaled_x = self.std_scale(x) 
        raw = self.layers(rescaled_x)
        out = sum(raw, dim=(1,2))
        return out

    def std_scale(self, x: Tensor):
        out = (x - self.train_means) / self.train_stds
        return out
    
    def set_std_scale(
        self, 
        train_means: Tensor, 
        train_stds: Tensor
    ):
        self.train_means = Buffer(train_means)
        self.train_stds = Buffer(train_stds)



