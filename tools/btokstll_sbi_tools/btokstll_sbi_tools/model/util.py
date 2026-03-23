
from pathlib import Path
from dataclasses import dataclass

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
    bucketize, 
    from_numpy
)
from torch.nn import Module
from pandas import read_parquet, DataFrame, Series, Index


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



def to_torch_tensor(
    x,
):
    """Convert to torch tensor."""
    
    def torch_tensor_from_pandas(dataframe):
        """
        Convert a pandas dataframe to a torch tensor.
        """
        tensor = from_numpy(dataframe.to_numpy(copy=True))
        return tensor

    if isinstance(x, DataFrame|Series|Index):
        return torch_tensor_from_pandas(x)
    elif isinstance(x, ndarray):
        return from_numpy(x)
    elif isinstance(x, Tensor):
        return x
    else: raise ValueError(f"Unsupported type: {type(x)}")



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


@dataclass
class Dataset:
    features: Tensor
    labels: Tensor
    
    def __postinit__(
        self,
    ):
        assert (
            len(self.features) 
            == len(self.labels)
        )

    def __len__(
        self,
    ) -> int: 
        return len(self.labels)
    
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
    
    @classmethod
    def from_pandas(
        cls,
        features:DataFrame|Series,
        labels:DataFrame|Series
    ):
        features_tensor = to_torch_tensor(features)
        labels_tensor = to_torch_tensor(labels)
        return cls(
            features=features_tensor, 
            labels=labels_tensor
        )
    
    @classmethod
    def from_dataframe_parquet_file(
        cls, 
        path:Path|str, 
        feature_names:list[str], 
        label_name:str
    ):
        df = read_parquet(path)
        features = to_torch_tensor(
            df[feature_names]
        )
        labels = to_torch_tensor(
            df[label_name]
        )
        return cls(
            features=features, 
            labels=labels,
        )


@dataclass
class Dataset_Set:
    
    train: Dataset
    eval: Dataset

    @classmethod
    def from_dataframe_parquet_files(
        cls,
        train_file_path:Path|str,
        eval_file_path:Path|str,
        feature_names:list[str],
        label_name:str,
    ):
        train = Dataset.from_dataframe_parquet_file(
            train_file_path, 
            feature_names=feature_names,
            label_name=label_name,
        )
        eval = Dataset.from_dataframe_parquet_file(
            eval_file_path, 
            feature_names=feature_names,
            label_name=label_name,
        )
        return cls(
            train=train, 
            eval=eval,
        )
    

def std_scale(
    data:Tensor, 
    reference:Tensor,
) -> Tensor:
    """
    Standard scale a dataset using 
    the mean and standard deviation
    of a reference dataset.
    """
    var_means = mean(reference, dim=0)
    var_stds = std(reference, dim=0)
    return (data - var_means) / var_stds


def bin_(
    data:Tensor,
    bin_edges:Tensor,
    eps:float=1e-2
) -> Tensor:
    
    if any((data < bin_edges[0]) | (data > bin_edges[-1])):
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