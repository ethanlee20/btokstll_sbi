
from math import floor
from dataclasses import dataclass, field

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor, arange, reshape, randperm
from torch.utils.data import Subset
from pandas import read_parquet, concat
from torch.nn import Module, ReLU, Linear, Dropout, Softmax, Sequential
from skorch import NeuralNetBinaryClassifier
from skorch.dataset import ValidSplit, Dataset

from matplotlib.pyplot import subplots, close

from helpers.plot import plot_loss


@dataclass
class DataLoader:

    data: Subset | Dataset
    batch_size: int
    shuffle: bool = True
    batched_indices: Tensor = field(init=False)

    def __post_init__(self):
        self.batched_indices = _generate_batched_indices(
            self.data, self.batch_size, self.shuffle
        )

    def __len__(self):
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
                self.data, self.batch_size, self.shuffle
            )
            raise StopIteration

        batch_indices = self.batched_indices[self.index]
        dataset = self.data.dataset if isinstance(self.data, Subset) else self.data
        batch_features = dataset.X[batch_indices]
        batch_labels = dataset.y[batch_indices] if dataset.y is not None else None

        self.index += 1
        return batch_features, batch_labels


def _generate_batched_indices(
    data: Subset | Dataset,
    batch_size: int,
    shuffle: bool,
) -> Tensor:

    indices = (
        data.indices if isinstance(data, Subset) 
        else np.arange(len(data)) if isinstance(data, Dataset) 
        else None
    )
    if indices is None: 
        raise ValueError
    
    if shuffle:
        indices = np.random.permutation(indices)

    num_batches = floor(len(data) / batch_size)
    truncated_dataset_size = num_batches * batch_size
    truncated_indices = indices[:truncated_dataset_size]

    batched_indices = np.reshape(truncated_indices, shape=(num_batches, batch_size))
    return batched_indices











df_sm = read_parquet("data/train_sm/combo.parquet")
df_vary = read_parquet("data/train_vary/combo.parquet")

df_sm["dC_9"] = df_vary["dC_9"].values

df_sm = df_sm.assign(label=1)
df_vary = df_vary.assign(label=0)

df = concat([df_sm, df_vary])

X = df[["q_sq", "cos_theta_lepton", "cos_theta_k", "chi", "dC_9"]].astype("float32").values
y = df["label"].astype("float32").values


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

    def forward(self, x):
        out = self.layers(x)
        return out








net = NeuralNetBinaryClassifier(
    MLP,
    max_epochs=1,
    lr=3e-4,
    batch_size=10_000,
    train_split=ValidSplit(stratified=True),
    iterator_train=DataLoader,
    iterator_valid=DataLoader,
    iterator_train__shuffle=True,
    device="cuda"
)

pipe = Pipeline(
    [
        ('scale', StandardScaler()),
        ('net', net)
    ]
)


pipe.fit(X, y)

train_losses = net.history[:, "train_loss"]
val_losses = net.history[:, "valid_loss"]
fig, ax = subplots()
plot_loss(
    ax, 
    train_losses=train_losses,
    val_losses=val_losses,
    compute_log=True
)
fig.savefig("plots/loss.png", bbox_inches="tight")
close()






df_val_sets = read_parquet("data/val/combo.parquet")

breakpoint()



y_proba = pipe.predict_proba(X)

print(y_proba)