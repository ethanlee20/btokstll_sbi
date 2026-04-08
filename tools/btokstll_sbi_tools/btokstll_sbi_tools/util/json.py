
from pathlib import Path
from json import load, dump
from typing import Any


def load_json(
    path:Path|str
) -> dict:
    with open(path) as file:
        return load(file)
    

def dump_json(
    obj:Any,
    path:Path|str,
    indent=4,
) -> None:
    with open(path, 'x') as file:
        dump(
            obj, 
            file, 
            indent=indent,
        )