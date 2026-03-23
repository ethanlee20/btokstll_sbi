
from typing import Any
from pathlib import Path
from json import load, dump


def are_instance(
    objects:list, 
    classinfo:Any,
)-> bool:
    if not isinstance(objects, list):
        raise ValueError("Input must be list.")
    for obj in objects:
        if not isinstance(obj, classinfo):
            return False
    return True


def to_int(
    x:float|int,
) -> int:
    try:
        x.is_integer()
    except AttributeError:
        raise ValueError(
            "Input must define is_integer method."
        )
    if not x.is_integer():
        raise ValueError(
            "Input is not an integer"
        )
    return int(x)


def load_json(
    path:Path
) -> dict:
    with open(path) as file:
        return load(file)
    

def dump_json(
    obj:Any,
    path:Path,
) -> None:
    with open(path, 'x') as f:
        dump(
            obj, 
            f, 
            indent=4,
        )
    

def access_nested_dict(
    d:dict, 
    *keys:Any
) -> Any:
    for k in keys: 
        d = d[k] 
    return d
    

def flatten_dict(
    d:dict, 
    *keys:str,
) -> dict:
    if not isinstance(
        v := access_nested_dict(d, *keys), 
        dict
    ):
        return {"_".join(keys): v}
    d_ = {}
    for k in v:
        d_ = d_ | flatten_dict(d, *keys, k)
    return d_
