
from dataclasses import dataclass
import pathlib

import pandas
import numpy
# import torch


def are_instance(objects:list, classinfo):

    assert isinstance(objects, list)

    for obj in objects:
        if not isinstance(obj, classinfo):
            return False
    return True


def safer_convert_to_int(x):
    assert x.is_integer()
    return int(x)


# def to_torch_tensor(x):

#     """Convert to torch tensor."""
    
#     def torch_tensor_from_pandas(dataframe):
#         """
#         Convert a pandas dataframe to a torch tensor.
#         """
#         tensor = torch.from_numpy(dataframe.to_numpy())
#         return tensor

#     if isinstance(x, pandas.DataFrame | pandas.Series):
#         return torch_tensor_from_pandas(x)
#     elif isinstance(x, numpy.ndarray):
#         return torch.from_numpy(x)
#     elif isinstance(x, torch.Tensor):
#         return x
#     else: raise ValueError(f"Unsupported type: {type(x)}")


@dataclass
class Interval:
    left:float
    right:float
    
    def __post_init__(self):
        if self.left > self.right:
            raise ValueError(
                "Interval left bound must be greater" 
                " than or equal to right bound."
            )



def to_pandas_interval(a:tuple|pandas.Interval):

    assert isinstance(a, tuple|pandas.Interval)

    if isinstance(a, tuple):
        if len(a) != 2:
            raise ValueError(
                f"Cannot convert length {len(a)} tuple to pandas interval."
                " Tuple must have a length of 2."
            )
        a = pandas.Interval(*a)

    return a


def append_to_stem(
    path:pathlib.Path, 
    s,
):
    return path.with_stem(f"{path.stem}{s}")


def get_nodes_nested_dict(
    nested:dict
):
    nodes = {}
    def recurse(dict_):
        for k, v in dict_.items():
            if not isinstance(v, dict):
                nodes[k] = v
            else: 
                recurse(v)
    recurse(nested)
    return nodes

