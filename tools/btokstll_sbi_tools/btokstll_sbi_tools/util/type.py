
from numpy import ndarray
from torch import from_numpy, Tensor
from pandas import DataFrame, Series


def are_instance(
    objects:list, 
    classinfo,
):
    assert isinstance(objects, list)

    for obj in objects:
        if not isinstance(obj, classinfo):
            return False
    return True


def safer_convert_to_int(
    x,
):
    assert x.is_integer()
    return int(x)


def to_torch_tensor(
    x,
):
    """Convert to torch tensor."""
    
    def torch_tensor_from_pandas(dataframe):
        """
        Convert a pandas dataframe to a torch tensor.
        """
        tensor = from_numpy(dataframe.to_numpy())
        return tensor

    if isinstance(x, DataFrame|Series):
        return torch_tensor_from_pandas(x)
    elif isinstance(x, ndarray):
        return from_numpy(x)
    elif isinstance(x, Tensor):
        return x
    else: raise ValueError(f"Unsupported type: {type(x)}")









