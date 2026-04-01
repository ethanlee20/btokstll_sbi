
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

from ..util import Interval


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
    path:Path,
):
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
class Dataset_Metadata:
    trials: Tensor = Tensor()
    bin_mids: Tensor = Tensor()
    bin_edges: Tensor = Tensor()
    bin_reweights: Tensor = Tensor()
    std_scale_means: Tensor = Tensor()
    std_scale_stds: Tensor = Tensor()
    orig_features: Tensor = Tensor()
    orig_labels: Tensor = Tensor()


@dataclass
class Dataset:
    metadata: Dataset_Metadata = field(
        default_factory=Dataset_Metadata
    )
    features: Tensor = field(default_factory=Tensor)
    labels: Tensor = field(default_factory=Tensor)

    def __postinit__(
        self,
    ):
        if (
            len(self.features) 
            != len(self.labels)
        ):
            raise ValueError(
                "Inconsistent array length."
            )

    def __len__(
        self,
    ) -> int: 
        return len(self.labels)
    
    def __iter__(self):
        return (
            self.features,
            self.labels,
        ).__iter__()

    def __eq__( # fix this
        self,
        other,
    ) -> bool:
        for self_array, other_array in zip(
            self, 
            other
        ):
            if not equal(
                self_array, 
                other_array,
            ):
                return False
        return True
    
    @classmethod
    def from_pandas(
        cls,
        features:DataFrame|Series,
        labels:DataFrame|Series,
        trials:Series|Index,
        features_dtype:str|None=None,
        labels_dtype:str|None=None,
        trials_dtype:str|None="int64",
    ):
        features_tensor = torch_tensor_from_pandas(
            features, 
            dtype=features_dtype,
        )
        labels_tensor = torch_tensor_from_pandas(
            labels, 
            dtype=labels_dtype,
        )
        trials_tensor = torch_tensor_from_pandas(
            trials,
            dtype=trials_dtype,
        )
        metadata = Dataset_Metadata(
            trials=trials_tensor
        )
        return cls(
            metadata=metadata,
            features=features_tensor, 
            labels=labels_tensor,
        )
    
    @classmethod
    def from_pandas_parquet_file(
        cls, 
        path:Path|str, 
        features:list[str], 
        label:str,
        trial_index:str="trial_num",
        features_dtype:str|None=None,
        labels_dtype:str|None=None,
    ):
        df = read_parquet(path)
        trials = df.index.get_level_values(trial_index)
        return cls.from_pandas(
            df[features], 
            df[label], 
            trials,
            features_dtype=features_dtype, 
            labels_dtype=labels_dtype,
        )
    
    def to_device(
        self, 
        device:str,
    ) -> None:
        self.features = self.features.to(
            device
        )
        self.labels = self.labels.to(
            device
        ) 
    
    def std_scale_features(
        self, 
        std_scale_means:Tensor, 
        std_scale_stds:Tensor, 
        save_orig:bool=False,
    ):
        self.metadata.std_scale_means = std_scale_means
        self.metadata.std_scale_stds = std_scale_stds
        if save_orig:
            self.metadata.orig_features = self.features
        self.features = std_scale(
            self.features, 
            std_scale_means, 
            std_scale_stds,
        )

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

    def group_by_trial(self,):
        if len(self.metadata.trials) != len(self.labels):
            raise ValueError(
                "Mismatch in dataset array lengths."
            )
        selection = (
            self.metadata.trials.unique().unsqueeze(-1) 
            == self.metadata.trials
        )
        self.grouped_features = [
            self.features[trial_select] 
            for trial_select in selection
        ]
        self.grouped_labels = [
            self.labels[trial_select] 
            for trial_select in selection
        ]
        self.grouped_trials = [
            self.metadata.trials[trial_select] 
            for trial_select in selection
        ]


@dataclass
class Dataset_Set_File_Paths:
    train:str|Path
    val:str|Path
    test:str|Path|None = None

    def __post_init__(self):
        self.train = Path(self.train)
        self.val = Path(self.val)
        if self.test is not None:
            self.test = Path(self.test)


@dataclass
class Dataset_Set:
    train: Dataset
    val: Dataset
    test: Dataset

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
        features:list[str],
        label:str,
        features_dtype:str|None=None,
        labels_dtype:str|None=None,
    ):
        datasets = {
            split: (
                Dataset.from_pandas_parquet_file(
                    path, 
                    features, 
                    label, 
                    features_dtype=features_dtype, 
                    labels_dtype=labels_dtype
                ) if path is not None
                else Dataset()
            )
            for split, path in asdict(paths).items()
        }
        return cls(**datasets)
    
    def apply_std_scale(
        self,
    ) -> None:
        std_scale_means = mean(
            self.train.features, 
            dim=0
        )
        std_scale_stds = std(
            self.train.features, 
            dim=0
        )
        for dset in self:
            if dset == Dataset():
                continue
            dset.std_scale_features(
                std_scale_means, 
                std_scale_stds
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



def plot_discrete_dist(
    ax, 
    bin_edges, 
    values, 
    **plot_kwargs,
) -> None:
    bin_edges = array(bin_edges)
    values = array(values)
    x = repeat(bin_edges, 2)[1:-1]
    y = repeat(values, 2)
    ax.plot(x, y, **plot_kwargs)