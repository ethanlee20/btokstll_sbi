
import numpy as np
from pandas import read_parquet, concat
from torch.nn import Module, ReLU, Linear, Dropout, Softmax, Sequential
from skorch import NeuralNetBinaryClassifier
from skorch.dataset import ValidSplit

from helpers.dataloader import DataLoader


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
    max_epochs=10,
    lr=3e-4,
    batch_size=10_000,
    train_split=ValidSplit(stratified=True),
    iterator_train__shuffle=True,
    device="cuda"
)

net.fit(X, y)

y_proba = net.predict_proba(X)