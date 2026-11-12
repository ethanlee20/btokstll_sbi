
from pathlib import Path

from tqdm import tqdm
from pandas import read_parquet, DataFrame, concat
from uproot import open

from .util import read_json
from .file_handling import (
    remove_dot_underscore_files,
    open_root_file_as_pandas,
    combine_pandas_parquet_files
)
from .physics import calc_vars
from .selection import (
    select_first_candidates,
    select_variables
)


def preprocess_root_file(
    path: Path | str, 
    parameters: list[str],
    observables: list[str],
    lepton_flavor="mu",
) -> Path:

    path = Path(path)
    
    dataframe = open_root_file_as_pandas(path)
    dataframe = calc_vars(dataframe, lepton_flavor)
    
    metadata_path = path.with_name("metadata.json")
    metadata = read_json(metadata_path)
    
    parameters = metadata["parameter_values"]
    dataframe = dataframe.assign(**parameters)
    
    trial_num = metadata["trial_num"]
    dataframe = dataframe.assign(trial_num=trial_num).set_index("trial_num", append=True)
    dataframe.index = dataframe.index.reorder_levels(["trial_num", "sim_type", None])
    
    dataframe = select_first_candidates(dataframe)
    variables = parameters + observables
    dataframe = select_variables(dataframe, variables)
    
    out_path = path.with_suffix(".parquet")
    dataframe.to_parquet(out_path)
    return out_path


def preprocess_dir(
    dir_: str | Path, 
    out_file_name: str, 
    parameters: list[str],
    observables: list[str],
    lepton_flavor="mu"
):

    dir_ = Path(dir_)
    
    remove_dot_underscore_files(dir_)

    root_file_paths = dir_.rglob("*.root")

    parquet_file_paths = [
        preprocess_root_file(
            path=path,
            parameters=parameters,
            observables=observables, 
            lepton_flavor=lepton_flavor
        )
        for path in tqdm(
            list(root_file_paths), 
            desc="Preprocessing files"
        )
    ]

    print("Combining files...")
    out_path = dir_.joinpath(out_file_name)
    combine_pandas_parquet_files(
        parquet_file_paths,
        out_path
    )
