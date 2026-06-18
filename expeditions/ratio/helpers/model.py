from torch import Tensor
from torch.nn import Module, Sequential, Linear, ReLU


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.layers = Sequential(
            Linear(5, 16),
            ReLU(),
            Linear(16, 32),
            ReLU(),
            Linear(32, 32),
            ReLU(),
            Linear(32, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
