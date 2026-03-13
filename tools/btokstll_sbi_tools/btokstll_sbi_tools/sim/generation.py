
from pathlib import Path

from pandas import DataFrame, concat
import uproot
from tqdm import tqdm
from ..util import (
    flatten_dict,
    load_json,
    read_parquets
)


def _open_simulated_data_root_file(
    path:Path|str, 
    unwanted_keys:list[str]=[
        "persistent;1", 
        "persistent;2"
    ],
) -> DataFrame:
    
    """
    Open a simulated data root file as a pandas dataframe.
    Each tree will be labeled by a pandas multi-index.
    """

    with uproot.open(path) as file:

        keys = [
            key.split(";")[0] for key in file.keys() 
            if key not in unwanted_keys
        ]
        tree_dataframes = [
            file[key].arrays(library="pd") 
            for key in keys
        ]

    dataframe = concat(
        tree_dataframes, 
        keys=keys,
        names=["sim_type",]
    )
    return dataframe


def _root_to_parquet(
    root_file_path:Path|str,
) -> None:
    
    root_file_path = Path(root_file_path)
    if not root_file_path.is_file():
        raise FileNotFoundError(
            f"File not found: {root_file_path}"
        )
    dataframe = _open_simulated_data_root_file(
        root_file_path
    ).drop(
        columns="__eventType__"
    )
    save_path = root_file_path.with_suffix(
        ".parquet"
    )
    dataframe.to_parquet(save_path)


def _root_files_to_parquet(
    paths:list[Path],
    lazy:bool=True,
) -> None:
    
    to_convert = (
        paths if not lazy 
        else [
            path for path in paths
            if not path.with_suffix(".parquet").is_file()
        ]
    )
    for path in to_convert:
        _root_to_parquet(path)


def combine_files(
    dirs:list[Path],
    out_file_path:Path|str,
    index_names:list[str]=[
        "trial_num", 
        "lepton_flavor", 
        "split",
    ]
) -> None:
    
    for dir_ in dirs:
        if not dir_.is_dir():
            raise ValueError(
                "Input paths must be directories."
                f" {dir_} is not a directory."
            )
        nested_data_file_paths = (
            list(dir_.glob("*.root")) + 
            list(dir_.glob("*.parquet"))
        )
        if not nested_data_file_paths:
            raise ValueError(
                f"No data file in directory: {dir_}"
            )
        if not dir_.joinpath("metadata.json").is_file():
            raise ValueError(
                f"No metadata file in directory: {dir_}"
            )
    
    for dir_ in (
        pbar := tqdm(
            dirs, 
            desc="Converting"
        )
    ):
        pbar.set_postfix_str(dir_.name)
        root_file_paths = list(dir_.glob("*.root"))
        _root_files_to_parquet(
            paths=root_file_paths,
            lazy=True,
        )

    metadata_file_paths = [
        dir_.joinpath("metadata.json") 
        for dir_ in dirs
    ]
    metadatas = [
        load_json(path) 
        for path in metadata_file_paths
    ]
    metadatas = [
        flatten_dict(metadata)
        for metadata in metadatas
    ]

    nested_data_file_paths = [
        list(dir_.glob("*.parquet"))
        for dir_ in dirs
    ]
    dataframes = [
        read_parquets(paths)
        for paths in nested_data_file_paths
    ]
    
    index = [
        {
            name: metadata.pop(name) 
            for name in index_names
        } 
        for metadata in metadatas
    ]
    dataframes = [
        df.assign(**metadata) 
        for df, metadata in zip(
            dataframes, 
            metadatas
        )
    ]

    keys = [
        tuple(i.values())
        for i in index
    ]
    data = concat(
        dataframes, 
        keys=keys,
        names=index_names, 
    )
    data = data.sort_index() # check this for memory usage
    
    data.to_parquet(out_file_path)