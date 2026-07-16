from typing import Any
from pathlib import Path
from json import load

from pandas import DataFrame, Series, Index
from torch import cuda, device, Tensor, randperm, from_numpy, stack, all
from torch.nn import Module


def read_json(path: Path | str) -> dict:
    with open(path) as file:
        return load(file)


def are_instance(
    objs: Any,
    classinfo: Any,
) -> bool:
    for o in objs:
        if not isinstance(o, classinfo):
            return False
    return True


def all_same(a: list):
    unique = set(a)
    if len(unique) == 1:
        return True
    return False


def shuffle_pandas(
    a: DataFrame | Series, keep_index: bool = False
) -> DataFrame | Series:
    """
    Shuffle a pandas series or dataframe.

    Parameters
    ----------
    a : Dataframe or Series
    keep_index : bool
        False shuffles the index with the data. True uses the original index.

    Returns
    -------
    Shuffled pandas dataframe or series
    """
    a_shuf = a.sample(frac=1, replace=False)
    if keep_index:
        a_shuf.index = a.index
    return a_shuf


def shuffle_tensor(a: Tensor) -> Tensor:
    num_rows = len(a)
    shuffled_indices = randperm(num_rows)
    a = a[shuffled_indices]
    return a


def pandas_from_tensor(t: Tensor, names: str | list[str]) -> DataFrame | Series:

    if t.dim() not in (1, 2):
        raise ValueError(
            "Input tensor must be one or two dimensional."
            f"\nGot {t.dim()} dimensions."
        )

    numpy_array = t.numpy(force=True)

    pandas_obj = (
        Series(numpy_array, name=names)
        if numpy_array.ndim == 1
        else DataFrame(numpy_array, columns=names)
    )

    return pandas_obj


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


def all_(
    input: Tensor,
    not_dim: int | tuple,
) -> Tensor:
    """
    torch.all but specify dimensions to not reduce.
    """
    if isinstance(not_dim, int):
        not_dim = (not_dim,)
    tensor_dims = range(input.dim())
    dim = tuple(d for d in tensor_dims if d not in not_dim)
    out = all(input, dim=dim)
    return out


def group(
    data: Tensor,
    by: Tensor,
) -> Tensor | list[Tensor]:
    uniques = by.unique(dim=0).unsqueeze(dim=1)
    overlap = by == uniques
    selection = all_(
        overlap,
        not_dim=(0, 1),
    )
    grouped_list = [data[i] for i in selection]
    num_per_group = [len(g) for g in grouped_list]
    if all_same(num_per_group):
        grouped_tensor = stack(grouped_list)
        return grouped_tensor
    return grouped_list


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
