
from pathlib import Path
from dataclasses import dataclass

from matplotlib.pyplot import subplots, close
from torch import (
    Tensor, 
    from_numpy, 
    linspace, 
    abs, 
    bucketize, 
    bin_count, 
    all, 
    arange, 
    randperm, 
    reshape,
)
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.nn import (
    Module, 
    Linear, 
    ReLU, 
    LogSoftmax, 
    CrossEntropyLoss,
)
from pandas import DataFrame, read_parquet


def tensor_from_pandas(
    obj
):
    if obj is None:
        return None
    return from_numpy(
        obj.to_numpy(
            copy=True
        )
    )


def tensor_from_dataframe(
    df:DataFrame|None, 
    cols:list|None=None,
):
    if df is None:
        return None
    if cols is None:
        return tensor_from_pandas(df)
    else: 
        return tensor_from_pandas(df[cols])


@dataclass
class Split:
    x : Tensor|None = None
    y : Tensor|None = None

    @classmethod
    def from_pandas(
        cls, 
        path:str|Path|None=None,
        x_cols:list[str]|None=None,
        y_cols:list[str]|None=None,
    ):
        df = (
            None if path is None
            else read_parquet(path)
        )
        x = tensor_from_dataframe(df, x_cols)
        y = tensor_from_dataframe(df, y_cols)
        return cls(x, y)

    def to_device(
        self, 
        device:str,
    ):
        if self.x is not None:
            self.x = self.x.to(device)
        if self.y is not None:
            self.y = self.y.to(device)

    def __len__(self,):
        if self.y is not None and self.x is not None:
            assert len(self.y) == len(self.x)
        if self.y is not None:
            return len(self.y)
        elif self.x is not None:
            return len(self.x)
        else:
            return 0


@dataclass
class Dataset:
    train : Split|None = None
    val : Split|None = None
    test : Split|None = None

    @classmethod
    def from_pandas(
        cls,
        train_path:str|Path|None=None,
        val_path:str|Path|None=None,
        test_path:str|Path|None=None,
        x_cols:list[str]|None=None,
        y_cols:list[str]|None=None,
    )
        paths = {
            "train": train_path,
            "val": val_path,
            "test": test_path,
        }
        kwargs = {
            split : Split.from_pandas(
                path, 
                x_cols, 
                y_cols,
            )
            for split, path in paths.items()
        }
        return cls(**kwargs)

    def to_device(
        self, 
        device:str,
    ):
        if self.train is not None:
            self.train.to_device(device)
        if self.val is not None:
            self.val.to_device(device)
        if self.test is not None:
            self.test.to_device(device)


def make_bins(
    interval:tuple,
    num:int,
) -> tuple[Tensor, Tensor]:
    edges = linspace(
        *interval,
        num+1,
    )
    mids = (                    # fix this later
        edges[:-1] + 
        0.5 * (edges[1] - edges[0])
    )
    return edges, mids


def to_bins(
    data:Tensor,
    edges:Tensor,
    eps:float=1e-2
) -> Tensor:
    if any((data < edges[0]) | (data > edges[-1])):
        raise ValueError(
            "Data outside of binned interval."
        )
    edges[0] -= abs(Tensor([eps])).item()
    return bucketize(
        input=data, 
        boundaries=edges, 
        out_int32=False, 
        right=False
    ) - 1


@dataclass
class Binner:
    edges : Tensor
    mids : Tensor

    @classmethod
    from_interval(
        cls, 
        interval:tuple, 
        num:int,
    ):
        edges, mids = make_bins(
            interval=interval, 
            num=num,
        )
        return cls(
            edges=edges, 
            mids=mids,
        )
    
    def bin(
        self, 
        data:Tensor,
        eps:float=1e-2,
    ):
        return to_bins(
            data=data,
            edges=self.edges,
            eps=eps,
        )

    def unbin(
        self, 
        data:Tensor,
    ):
        return self.mids[data]


def calc_weights(
    labels:Tensor,
    num_classes:int,
) -> Tensor:
    """
    Calculate class weights for 
    reweighting to uniform distribution.
    """
    counts = bincount(
        input=labels, 
        minlength=num_classes,
    )
    if (counts == 0).any():
        raise ValueError(
            f"Some bins are empty!"
            f" Bin counts:\n{bin_counts}"
        )
    inverse_counts = 1 / counts
    return inverse_counts / sum(inverse_counts)


def std_scale(
    data:Tensor, 
    means:Tensor, 
    stds:Tensor,
):
    return (data - means) / stds


def undo_std_scale(
    data:Tensor,
    means:Tensor,
    stds:Tensor,
):
    return data * stds + means


@dataclass
class StdScaler:
    means: Tensor
    stds: Tensor

    def scale(
        self, 
        data:Tensor,
    ):
        return std_scale(
            data, 
            means=self.means,
            stds=self.stds,
        )
    
    def unscale(
        self, 
        data:Tensor,
    ):
        return undo_std_scale(
            data,
            means=self.means,
            stds=self.stds,
        )

    @classmethod
    def from_data(
        cls, 
        data:Tensor,
    ):
        means = data.mean(dim=0)
        stds = data.std(dim=0)
        return cls(
            means=means, 
            stds=stds,
        )


def all_(
    input:Tensor, 
    keep_dims:tuple|None=None,
) -> Tensor:
    """
    torch.all but specify dimensions to not reduce.
    """
    dims = range(input.dim())
    reduce_dims = tuple(
        d for d in dims 
        if d not in keep_dims
    )
    result = all(tensor, dim=reduce_dims)
    return result


def group(
    data:Tensor, 
    by:Tensor,
) -> Tensor|list[Tensor]:
    uniques = by.unique(dim=0)
    uniques = uniques.unsqueeze(dim=1)
    select = uniques == by
    select = all_(
        select, 
        keep_dims=(0,1),
    )
    grouped = [data[i] for i in select]
    try: 
        return cat(
            [g.unsqueeze(-1) for g in grouped], 
            dim=0,
        )
    except RuntimeError:          # fix this
        return grouped


def make_batched_indices(
    dataset_size:int, 
    batch_size:int, 
    shuffle:bool=True,
):
    indices = arange(dataset_size)
    if shuffle: 
        indices = indices[
            randperm(len(indices))
        ]
    num_batches = floor(
        dataset_size 
        / batch_size
    )
    return reshape(
        indices[:num_batches*batch_size], 
        shape=(num_batches, batch_size)
    )


@dataclass
class DataLoader:
    data : Split
    batch_size : int
    shuffle : bool = True

    def __postinit__(self):
        self.set_indices()

    def set_indices(self):
        self.indices = make_batched_indices(
            dataset_size=len(self.data), 
            batch_size=self.batch_size, 
            shuffle=self.shuffle
        )

    def __len__(
        self,
    ):
        return len(self.indices)
    
    def __iter__(
        self,
    ):
        self.index = 0
        return self
    
    def __next__(
        self,
    ) -> tuple[Tensor, Tensor]:
        if self.index >= len(self):
            self.set_indices()
            raise StopIteration
        batch_indices = self.indices[self.index]
        batch_x = (
            None if self.data.x is None 
            else self.data.x[batch_indices]
        )
        batch_y = (
            None if self.data.y is None 
            else self.data.y[batch_indices]
        )
        self.index += 1
        return batch_x, batch_y


def train_batch(
    x:Tensor, 
    y:Tensor, 
    model:Module, 
    loss_fn, 
    optim:Optimizer,
) -> Tensor:
    model.train()
    yhat = model(x)    
    loss = loss_fn(yhat, y)
    loss.backward()
    optim.step()
    optim.zero_grad()
    return loss
    

def eval_batch(
    x:Tensor, 
    y:Tensor, 
    model:Module, 
    loss_fn,
) -> Tensor:
    model.eval()
    with no_grad():
        yhat = model(x)
        loss = loss_fn(yhat, y)
    return loss


def train_epoch(
    dloader:DataLoader, 
    model:Module, 
    loss_fn, 
    optim:Optimizer,
) -> Tensor:
    cum_loss = 0
    for x, y in dloader:
        batch_loss = train_batch(
            x=x, 
            y=y, 
            model=model, 
            loss_fn=loss_fn, 
            optim=optim,
        )
        cum_loss += batch_loss.item()
    num_batches = len(dloader)
    return cum_loss / num_batches


def eval_epoch(
    dloader:DataLoader, 
    model:Module, 
    loss_fn, 
    scheduler:LRScheduler|None=None
) -> Tensor:
    cum_loss = 0
    for x, y in data_loader:
        batch_loss = eval_batch(
            x=x, 
            y=y, 
            model=model, 
            loss_fn=loss_fn,
        )
        cum_loss += batch_loss.item()
    num_batches = len(dloader)
    avg_loss = cum_loss / num_batches
    if scheduler is not None:      # fix this
        scheduler.step(avg_loss)
    return avg_loss


def train_eval_epoch(
    train_dloader: DataLoader, 
    eval_dloader: DataLoader, 
    model: Module, 
    loss_fn,
    optim: Optimizer,
    scheduler: LRScheduler|None = None,
) -> tuple[Tensor, Tensor]:
    train_loss = train_epoch(
        dloader=train_dloader, 
        model=model, 
        loss_fn=loss_fn, 
        optim=optim,
    )
    eval_loss = eval_epoch(
        dloader=eval_dloader, 
        model=model, 
        loss_fn=loss_fn, 
        scheduler=scheduler
    )
    return train_loss, eval_loss


class Model(Module):
    def __init__(self):
        super().__init__()  
        self.layers = Sequential(
            Linear(4, 16),
            ReLU(),
            Linear(16, 32),
            ReLU(),
            Linear(32, 64),
            ReLU(),
            Linear(64, 20),
            LogSoftmax(),          ### ???
        )

    def forward(
        self, 
        x:Tensor,
    ) -> Tensor:
        log_probs = self.layers(x)
        return log_probs


train_path = "data/vary_dc9_train.parquet"
val_path = "data/vary_dc9_val.parquet"

x_cols = ["cos_theta_mu",]
y_cols = ["delta_wc_values_dc9",]

dset = Dataset.from_pandas(
    train_path=train_path,
    val_path=val_path,
    x_cols=x_cols,
    y_cols=y_cols,
)

interval = (-10, 0)
num_bins = 20

binner = Binner.from_interval(
    interval=interval, 
    num=num_bins
)

unbinned_train_y = dset.train.y.copy()
unbinned_val_y = dset.val.y.copy()

dset.train.y = binner.bin(dset.train.y)
dset.val.y = binner.bin(dset.val.y)

std_scaler = StdScaler.from_data(dset.train.x)

unscaled_train_x = dset.train.x.copy()
unscaled_val_x = dset.val.x.copy()

dset.train.x = std_scaler.scale(dset.train.x)
dset.val.x = std_scaler.scale(dset.val.x)

dset.train.x = group(data=dset.train.x, by=dset.train.y)
dset.train.y = group(data=dset.train.y, by=dset.train.y)
dset.val.x = group(data=dset.val.x, by=dset.val.y)
dset.val.y = group(data=dset.val.y, by=dset.val.y)

dset.train.y = dset.train.y.unique(dim=1)
dset.val.y = dset.val.y.unique(dim=1)

weights = calc_weights(
    labels=dset.train.y, 
    num_classes=num_bins,
)

model = Model()

device = "cuda"

dset.to_device(device)
model = model.to(device)

train_batch_size = 64
eval_batch_size = 4

train_dloader = DataLoader(
    dset.train, 
    batch_size=train_batch_size,
)
eval_dloader = DataLoader(
    dset.train, 
    batch_size=eval_batch_size,
)

loss_fn = CrossEntropyLoss(
    weight=weights
)
optim = AdamW(
    params=model.parameters()
)

num_epochs = 10

train_losses = []
eval_losses = []
for epoch in num_epochs:
    train_loss, eval_loss = train_eval_epoch(
        train_dloader=train_dloader, 
        eval_dloader=eval_dloader, 
        model=model, 
        loss_fn=loss_fn,
        optim=optim,
    )
    train_losses.append(train_loss)
    val_losses.append(val_loss)

fig, ax = subplots(layout="constrained")
ax.plot(train_losses, label="train")
ax.plot(val_losses, label="val")
fig.savefig("plots/loss.png", bbox_inches="tight")
close(fig)




















