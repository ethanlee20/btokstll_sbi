
from pathlib import Path
from json import load

from pandas import DataFrame, concat, read_parquet
from uproot import open


def remove_dot_underscore_files(dir_: str | Path):
    dir_ = Path(dir_)
    pattern = "._*"
    for path in dir_.rglob(pattern):
        path.unlink()


def read_json(path: Path | str) -> dict:
    with open(path) as file:
        out = load(file)
    return out


def open_root_file_as_pandas(path: Path | str) -> DataFrame:
    """
    Open a root file as a pandas dataframe.
    Each tree is labeled by a pandas multi-index.
    """
    unwanted_keys = ["persistent;1", "persistent;2"]
    with open(path) as file:
        keys = [key.split(";")[0] for key in file.keys() if key not in unwanted_keys]
        dataframes = [file[key].arrays(library="pd") for key in keys]
    out = concat(dataframes, keys=keys, names=["sim_type"])
    return out


def combine_pandas_parquet_files(input_paths: list[Path|str], out_path: Path|str):
    dataframes = [read_parquet(path) for path in input_paths]
    combined = concat(dataframes)
    combined.to_parquet(out_path)
