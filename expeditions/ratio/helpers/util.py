from typing import Any

from pandas import DataFrame, Series, Index
from torch import cuda, device, Tensor, randperm, from_numpy
from torch.nn import Module


def are_instance(
    objs: Any,
    classinfo: Any,
) -> bool:
    for o in objs:
        if not isinstance(o, classinfo):
            return False
    return True


def shuffle_pandas(a: DataFrame | Series) -> DataFrame:
    a_shuf = a.sample(frac=1, replace=False)
    return a_shuf


def shuffle_tensor(a: Tensor) -> Tensor:
    num_rows = len(a)
    shuffled_indices = randperm(num_rows)
    a = a[shuffled_indices]
    return a


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


def pandas_from_tensor(t: Tensor, names: str | list[str]) -> DataFrame | Series:

    if t.dims not in (1, 2):
        raise ValueError(
            "Input tensor must be one or two dimensional." f"\nGot {t.dims} dimensions."
        )

    numpy_array = t.numpy(force=True)

    pandas_obj = (
        Series(numpy_array, name=names)
        if numpy_array.dims == 1
        else DataFrame(numpy_array, columns=names)
    )

    return pandas_obj


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
