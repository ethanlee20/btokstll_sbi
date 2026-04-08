
from dataclasses import dataclass
from typing import Any


def merge_dicts(
    *dicts:dict,
):
    merged = {}
    for d in dicts:
        merged = merged | d
    return merged


@dataclass
class Node:
    key:Any
    value:Any

    def as_dict(self):
        return {
            self.key: self.value,
        }


def access_nested_dict(
    dict_:dict, 
    *keys:Any,
    key_sep:str|None=None
) -> Node:
    value = dict_.copy()
    for k in keys: 
        value = value[k]
    key = (
        keys if key_sep is None
        else key_sep.join(keys)
    )
    return Node(key, value)


def flatten_dict(
    dict_:dict, 
    *keys:Any,
) -> dict:
    node = access_nested_dict(
        dict_, 
        *keys,
        key_sep="_",
    )
    if not isinstance(
        node.value, 
        dict,
    ):
        return node.as_dict()
    subdicts = [
        flatten_dict(dict_, *keys, k) 
        for k in node.value.keys()
    ]
    return merge_dicts(*subdicts)