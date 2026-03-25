
from dataclasses import dataclass
from typing import Any, Iterator
from pathlib import Path
from json import load, dump

from matplotlib.pyplot import style, rcParams, savefig, close



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


def setup_high_quality_plotting(
) -> None:
    rcParams.update(
        {
            "figure.dpi": 400, 
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Computer Modern",
            "text.latex.preamble": r"\usepackage{array} \usepackage{tabularx}"
        }
    )


def setup_dark_plotting(
) -> None:
    style.use("dark_background")
    

def setup_plotting(
    dark:bool=True,
    high_quality:bool=True,
) -> None:
    if dark:
        setup_dark_plotting()
    if high_quality:
        setup_high_quality_plotting()


def save_plot_and_close(
    path:Path|str,
) -> None:
    savefig(
        path,
        bbox_inches="tight"
    )
    close()


@dataclass
class Interval:
    left: float|int
    right: float|int

    def __post_init__(self):
        if self.left > self.right:
            raise ValueError(
                "Interval left bound must be"
                " less than or equal to right bound."
            )

    def __iter__(
        self,
    ) -> Iterator[float|int]:
        return (
            self.left, 
            self.right
        ).__iter__()

