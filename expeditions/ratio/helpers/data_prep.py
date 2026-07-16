from pathlib import Path

from torch import ones, zeros, float32
from pandas import read_parquet, DataFrame, concat
from uproot import open

from .physics import calc_vars
from .util import shuffle_pandas, read_json
from .dataset import (
    concat_datasets,
    dataset_from_dataframe,
    dataset_from_dataframe_parquet,
    group_dataset_by_label,
)


def open_root_data_file(path: Path | str) -> DataFrame:
    """
    Open a simulated data root file as a pandas dataframe.
    Each tree will be labeled by a pandas multi-index.
    """
    unwanted_keys = ["persistent;1", "persistent;2"]
    with open(path) as file: 
        keys = [key.split(";")[0] for key in file.keys() if key not in unwanted_keys]
        dataframes = [file[key].arrays(library="pd") for key in keys]
    out = concat(dataframes, keys=keys, names=["sim_type"])
    return out


def calc_vars_root_file(path: Path | str, lepton_flavor="mu") -> DataFrame:
    """
    Save the output DataFrame to a file.
    """
    dataframe = open_root_data_file(path)
    dataframe = calc_vars(dataframe, lepton_flavor)
    metadata_path = Path(path).with_name("metadata.json")
    parameters = read_json(metadata_path)["parameter_values"]
    out = dataframe.assign(**parameters)
    return out


def select_first_candidates(dataframe: DataFrame) -> DataFrame:
    out = dataframe[dataframe["__candidate__"] == 0].copy()
    return out


def prep_root_file(path: Path | str):
    dataframe = calc_vars_root_file(path)
    dataframe = select_first_candidates(dataframe)
    columns = [
        "q_sq",
        "q_sq_mc",
        "cos_theta_lepton",
        "cos_theta_lepton_mc",
        "cos_theta_k",
        "cos_theta_k_mc",
        "chi",
        "chi_mc",
        "dC_7",
        "dC_9",
        "dC_10",
    ]
    save_path = Path(path).with_suffix(".parquet")
    dataframe[columns].to_parquet(save_path)
    return save_path


def remove_unnecessary_files(dir_:str|Path):
    dir_ = Path(dir_)
    unnecessary_file = "._*"
    for path in dir_.rglob(unnecessary_file):
        path.unlink()


def prep_data_dir(dir_: str | Path):
    remove_unnecessary_files(dir_)    
    root_file_paths = Path(dir_).rglob("*.root")
    parquet_file_paths = [prep_root_file(path) for path in root_file_paths]
    dataframes = [read_parquet(path) for path in parquet_file_paths]
    out = concat(dataframes)
    out.to_parquet(Path(dir_).joinpath("combo.parquet"))


def prep_train_data(
    path: Path | str,
):
    feature_names = [
        "q_sq",
        "cos_theta_lepton",
        "cos_theta_k",
        "chi",
    ]
    dataframe = read_parquet(path)

    dataframe["dC_9_shuf"] = shuffle_pandas(dataframe["dC_9"], keep_index=True)

    original_dset = dataset_from_dataframe(
        dataframe,
        feature_names=feature_names + ["dC_9"],
        features_dtype="float32",
    )
    shuffled_dset = dataset_from_dataframe(
        dataframe,
        feature_names=feature_names + ["dC_9_shuf"],
        features_dtype="float32",
    )

    original_dset.labels = ones(len(original_dset), dtype=float32).unsqueeze(-1)
    shuffled_dset.labels = zeros(len(shuffled_dset), dtype=float32).unsqueeze(-1)
    out = concat_datasets([original_dset, shuffled_dset])
    return out


def prep_eval_data(path: Path | str):
    feature_names = [
        "q_sq",
        "cos_theta_lepton",
        "cos_theta_k",
        "chi",
    ]
    label_names = ["dC_9"]
    dataset = dataset_from_dataframe_parquet(
        path,
        feature_names=feature_names,
        label_names=label_names,
        features_dtype="float32",
        labels_dtype="float32",
    )
    out = group_dataset_by_label(dataset)
    return out
