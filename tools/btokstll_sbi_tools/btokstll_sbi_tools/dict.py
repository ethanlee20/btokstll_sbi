
from typing import Any


def merge_dicts(
    *dicts:dict,
) -> dict:
    merged = {}
    for d in dicts:
        merged = merged | d
    return merged


def access_nested_dict(
    dict_:dict, 
    *keys:Any,
) -> Any:
    value = dict_
    for k in keys: 
        value = value[k]
    return value


def flatten_dict(
    dict_:dict, 
    *keys:str,
    sep:str="_",
) -> dict:
    value = access_nested_dict(
        dict_, 
        *keys,
    )
    if not isinstance(
        value, 
        dict,
    ):
        key = sep.join(keys)
        return {key: value}
    subdicts = [
        flatten_dict(dict_, *keys, subkey) 
        for subkey in value.keys()
    ]
    return merge_dicts(*subdicts)