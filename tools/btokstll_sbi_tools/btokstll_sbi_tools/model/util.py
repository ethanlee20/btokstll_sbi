
from typing import Any
from pathlib import Path
from dataclasses import dataclass, asdict, astuple, field

from numpy import array, repeat, ndarray
from torch import (
    cuda, 
    device, 
    Tensor, 
    load, 
    save, 
    any, 
    abs, 
    mean, 
    std, 
    linspace,
    bucketize, 
    from_numpy, 
    bincount,
    equal,
)
from torch.nn import Module
from pandas import read_parquet, DataFrame, Series, Index

from ..util.misc import Interval


def select_device(
    verbose=True
) -> str:
    """
    Select a device to compute with.

    Returns the name of the selected device.
    "cuda" if cuda is available, otherwise "cpu".
    """
    device = "cuda" if cuda.is_available() else "cpu"
    if verbose: 
        print("Device: ", device)
    return device


def get_model_current_device(
    model:Module
) -> device:
    device = next(model.parameters()).device
    return device


def torch_tensor_from_pandas(
    obj:DataFrame|Series|Index, 
    dtype:str|None=None,
) -> Tensor:
    """
    Convert a pandas dataframe to a torch tensor.
    """
    tensor = from_numpy(
        obj.to_numpy(
            dtype=dtype, 
            copy=True,
        )
    )
    return tensor


# def to_torch_tensor(
#     x:ndarray|DataFrame|Series|Index|Tensor,
#     dtype:str,
# ) -> Tensor:
#     """
#     Convert to torch tensor.
#     """
#     if isinstance(x, DataFrame|Series|Index):
#         return _torch_tensor_from_pandas(x, dtype)
#     elif isinstance(x, ndarray):
#         return from_numpy(x)
#     elif isinstance(x, Tensor):
#         return x
#     else: raise ValueError(f"Unsupported type: {type(x)}")



def save_torch_model_state_dict(
    model:Module, 
    path:Path|str,
):
    path = Path(path)
    if not path.parent.is_dir():
        raise ValueError(f"Parent directory doesn't exist: {path.parent}")
    if path.exists():
        raise ValueError(f"File exists: {path}")
    save(model.state_dict(), path)


def load_torch_model_state_dict( 
    path: Path|str
):
    state_dict = load(path, weights_only=True)
    return state_dict


def std_scale(
    data:Tensor, 
    means:Tensor, 
    stdevs:Tensor,
):
    return (data - means) / stdevs


def make_bins(
    interval:Interval,
    num_bins:int,
) -> tuple[Tensor, Tensor]:
    bin_edges = linspace(
        interval.left,
        interval.right,
        num_bins+1,
    )
    bin_mids = (
        bin_edges[:-1] + 0.5 
        * (bin_edges[1] - bin_edges[0])
    )
    return bin_edges, bin_mids


def to_bins(
    data:Tensor,
    bin_edges:Tensor,
    eps:float=1e-2
) -> Tensor:
    if any(
        (data < bin_edges[0]) 
        | (data > bin_edges[-1])
    ):
        raise ValueError(
            "Data outside of binned interval."
        )
    bin_edges[0] -= abs(Tensor([eps])).item()
    binned_data = bucketize(
        input=data, 
        boundaries=bin_edges, 
        out_int32=False, 
        right=False
    ) - 1
    return binned_data


def calc_label_reweights(
    labels:Tensor,
    num_labels:int,
) -> Tensor:
    """
    Calculate class weights for reweighting 
    classes to uniform distribution.
    """
    bin_counts = bincount(
        input=labels, 
        minlength=num_labels,
    )
    if (bin_counts == 0).any():
        raise ValueError(
            f"Some bins are empty!"
            f" Bin counts:\n{bin_counts}"
        )
    inverse_bin_counts = 1 / bin_counts
    return inverse_bin_counts / sum(inverse_bin_counts)


@dataclass
class Data:
    data: Tensor = Tensor()

    def to(self, arg):
        self.data = self.data.to(arg)
 
    def std_scale(
        self, 
        by:Data,
        dim:int,
        keep:bool=False
    ):
        if keep:
            self.no_scale = self.data
        means = by.data.mean(dim=dim)
        stds = by.data.std(dim=dim) 
        self.data = std_scale(self.data, means, stds)
        self.scale_means = means
        self.scale_stds = stds

    def group(
        self, 
        by:Data, 
        keep:bool=False,
    ):
        if keep:
            self.ungrouped = self.data
        selection = (
            by.data.unique().unsqueeze(-1) 
            == by.data
        )
        self.data = [
            self.data[i] for i in selection
        ]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self.data.__iter__()
    
    def __getitem__(self, index:int):
        return self.data[index]


@dataclass
class Dataset:
    features: Data|None = None
    labels: Data|None = None
    trials: Data|None = None

    def std_scale(
        self, 
        by:Dataset, 
        scale_features:bool, 
        scale_labels:bool, 
        dim:int,
    ):
        if scale_features:
            self.features.std_scale(by=by.features, dim=dim)
        if scale_labels:
            self.labels.std_scale(by=by.labels, dim=dim)

    def __len__(
        self,
    ) -> int: 
        return max(
            [
                len(x) for x in 
                (
                    self.features, 
                    self.labels, 
                    self.trials,
                ) 
                if x is not None
            ]
        )

    @classmethod
    def from_pandas(
        cls,
        features:DataFrame|Series|None=None,
        labels:DataFrame|Series|None=None,
        trials:Series|Index|None=None,
        features_dtype:str|None=None,
        labels_dtype:str|None=None,
        trials_dtype:str|None=None,#"int64",
    ):
        kwargs = {}
        for arg, pandas_obj, dtype in zip(
            ("features", "labels", "trials"),
            (features, labels, trials),
            (features_dtype, labels_dtype, trials_dtype),
        ):
            if pandas_obj is None:
                continue
            tensor = torch_tensor_from_pandas(
                pandas_obj, 
                dtype=dtype
            )
            data = Data(tensor)
            kwargs[arg] = data
        return cls(**kwargs)
    
    @classmethod
    def from_pandas_parquet_file(
        cls, 
        path:Path|str, 
        features:list[str]|None=None, 
        labels:list[str]|None=None,
        trials:str|None=None,#"trial_num",
        features_dtype:str|None=None,
        labels_dtype:str|None=None,
        trials_dtype:str|None=None,
    ):
        dataframe = read_parquet(path)
        kwargs = {
            "features": (
                None 
                if features is None 
                else dataframe[features]
            ),
            "labels": (
                None
                if labels is None
                else dataframe[labels]
            ),
            "trials": (
                None
                if trials is None
                else dataframe.index.get_level_values(trials)
            )
        }
        return cls.from_pandas(
            **kwargs,
            features_dtype=features_dtype, 
            labels_dtype=labels_dtype,
            trials_dtype=trials_dtype,
        )
    
    def to(
        self, 
        arg:Any,
    ) -> None:
        if self.features is not None:
            self.features.to(arg)
        if self.labels is not None:
            self.labels.to(arg) 
        if self.trials is not None:
            self.trials.to(arg)

    def bin_labels(
        self,
        interval:Interval,
        num_bins:int,
        save_orig:bool=True,
    ):
        if save_orig:
            self.metadata.orig_labels = self.labels
        self.metadata.bin_edges, self.metadata.bin_mids = make_bins(
            interval, 
            num_bins,
        )
        self.labels = to_bins(
            self.labels, 
            self.metadata.bin_edges,
        )

    def calc_label_reweights(
        self, 
        num_labels:int,
    ):
        self.metadata.bin_reweights = calc_label_reweights(
            self.labels, 
            num_labels,
        )

    def group_by_trial(
        self,
    ):
        if self.features is not None:
            self.features.group(by=self.trials)
        if self.labels is not None:
            self.labels.group(by=self.trials)
        if self.trials is not None:
            self.trials.group(by=self.trials)


@dataclass
class Dataset_Set_File_Paths:
    train:str|Path|None = None
    val:str|Path|None = None
    test:str|Path|None = None

    def __post_init__(self):
        if self.train is not None:
            self.train = Path(self.train)
        if self.val is not None:
            self.val = Path(self.val)
        if self.test is not None:
            self.test = Path(self.test)


@dataclass
class Dataset_Set:
    train: Dataset|None = None
    val: Dataset|None = None
    test: Dataset|None = None

    def __iter__(self):
        return (
            self.train, 
            self.val, 
            self.test
        ).__iter__()

    @classmethod
    def from_pandas_parquet_files(
        cls,
        paths:Dataset_Set_File_Paths,
        features:list[str]|None=None,
        labels:list[str]|None=None,
        trials:str|None=None,
        features_dtype:str|None=None,
        labels_dtype:str|None=None,
        trials_dtype:str|None=None,
    ):
        datasets = {
            split: (
                None if path is None
                else Dataset.from_pandas_parquet_file(
                    path, 
                    features=features, 
                    labels=labels, 
                    trials=trials,
                    features_dtype=features_dtype, 
                    labels_dtype=labels_dtype,
                    trials_dtype=trials_dtype,
                )
            )
            for split, path in asdict(paths).items()
        }
        return cls(**datasets)
    
    def apply_std_scale(
        self,
        scale_features:bool=True,
        scale_labels:bool=False,
        dim:int=0,
    ) -> None:
        
        if self.val is not None:
            self.val.std_scale(
                self.train, 
                scale_features=scale_features, 
                scale_labels=scale_labels, 
                dim=dim,
            )
        if self.train is not None:
            self.train.std_scale(
                self.train,
                scale_features=scale_features,
                scale_labels=scale_labels,
                dim=dim,
            )

    def apply_binning(
        self,
        interval:Interval,
        num_bins:int,
    ) -> None:
        for dset in self:
            if dset == Dataset(): 
                continue
            dset.bin_labels(
                interval, 
                num_bins, 
            )

    def calc_label_reweights(
        self, 
        num_labels:int,
    ):
        self.train.calc_label_reweights(
            num_labels
        )