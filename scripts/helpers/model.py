from pathlib import Path

from torch import Tensor, zeros, ones, save, load, log
from torch.nn import Module, Sequential, Linear, ReLU, Dropout
from torch.nn.parameter import Buffer
from torch.nn.functional import logsigmoid, sigmoid


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.layers = Sequential(
            Linear(5, 500),
            ReLU(),
            # Dropout(p=0.1),
            Linear(500, 500),
            ReLU(),
            Dropout(p=0.2),
            Linear(500, 500),
            ReLU(),
            Dropout(p=0.2),
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

    def probability(self, x: Tensor) -> Tensor:
        logits = self.forward(x)
        out = sigmoid(logits)
        return out

    def log_likelihood_ratio(self, x: Tensor) -> Tensor:
        likelihood_ratio = 1 / self.probability(x) - 1
        out = log(likelihood_ratio)
        return out


def save_model_state_dict(
    model: Module,
    path: Path | str,
    overwrite_ok: bool = True,
):
    path = Path(path)
    if not path.parent.is_dir():
        raise ValueError(f"Parent directory doesn't exist: {path.parent}")
    if path.exists() and not overwrite_ok:
        raise ValueError(f"File exists: {path}")
    save(model.state_dict(), path)


def load_model_state_dict(path: Path | str):
    state_dict = load(path, weights_only=True)
    return state_dict
