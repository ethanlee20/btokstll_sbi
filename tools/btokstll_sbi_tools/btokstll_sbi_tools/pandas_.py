
from pandas import DataFrame, Series, Index
from torch import Tensor, from_numpy


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