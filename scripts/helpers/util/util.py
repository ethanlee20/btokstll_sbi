
from typing import Any
from pathlib import Path
from json import load

from pandas import DataFrame, Series, Index
from torch import cuda, device, Tensor, randperm, from_numpy, stack, all, randint
from torch.nn import Module





def shuffle_tensor(a: Tensor, dim=0) -> Tensor:
    num_elements = a.shape[dim]
    shuffled_indices = randperm(num_elements)
    out = a.index_select(dim, shuffled_indices)
    out = out.detach().clone()
    return out


def bootstrap_tensor(a: Tensor, num=None, dim=0) -> Tensor:
    num_elements = a.shape[dim]
    bootstrap_size = num_elements if num is None else num
    bootstrapped_indices = randint(
        high=num_elements, 
        size=(bootstrap_size,), 
        device=a.device
    )
    out = a.index_select(dim, bootstrapped_indices)
    out = out.detach().clone()
    return out


def tensor_from_pandas(
    obj: DataFrame | Series | Index,
    dtype: str | None = None,
) -> Tensor:
    """
    Convert a pandas object to a torch tensor.
    """
    tensor = from_numpy(
        obj.to_numpy(
            dtype=dtype,
            copy=True,
        )
    )
    return tensor


def select_device(verbose=True) -> str:
    """
    Select a device to compute with.

    Return the name of the selected device.
    Prefer cuda over cpu.
    """
    device = "cuda" if cuda.is_available() else "cpu"
    if verbose:
        print("Device: ", device)
    return device


def get_model_device(model: Module) -> device:
    device = next(model.parameters()).device
    return device
