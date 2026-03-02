
from json import load, dump
from pathlib import Path


def load_json(
    path:Path
):
    with open(path) as file:
        return load(file)