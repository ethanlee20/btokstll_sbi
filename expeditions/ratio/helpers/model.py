from torch import Tensor, zeros, ones
from torch.nn import Module, Sequential, Linear, ReLU
from torch.nn.parameter import Buffer
from torch.nn.functional import logsigmoid


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.layers = Sequential(
            Linear(5, 500),
            ReLU(),
            Linear(500, 500),
            ReLU(),
            Linear(500, 500),
            ReLU(),
            Linear(500, 1),
        )

        self.train_means = Buffer(zeros(5))
        self.train_stds = Buffer(ones(5))

    def set_std_scale(self, train_means: Tensor, train_stds: Tensor):
        self.train_means = Buffer(train_means)
        self.train_stds = Buffer(train_stds)

    def std_scale(self, x: Tensor):
        out = (x - self.train_means) / self.train_stds
        return out

    def forward(self, x: Tensor) -> Tensor:
        scaled_x = self.std_scale(x)
        logits = self.layers(scaled_x)
        return logits

    def log_likelihood_ratio(self, x: Tensor) -> Tensor:
        logits = self.forward(x)
        out = logsigmoid(logits) - logsigmoid(-logits)
        return out
