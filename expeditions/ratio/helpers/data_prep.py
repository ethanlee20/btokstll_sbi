from pathlib import Path

from torch import ones, zeros, float32
from pandas import read_parquet

from .util import shuffle_pandas
from .dataset import (
    concat_datasets,
    dataset_from_dataframe,
    dataset_from_dataframe_parquet,
    group_dataset_by_label,
)


def prep_train_data(
    path: Path | str,
):
    feature_names = [
        "q_squared",
        "cos_theta_mu",
        "cos_theta_k",
        "chi",
    ]
    dataframe = read_parquet(path)

    dataframe["dc9_shuf"] = shuffle_pandas(dataframe["dc9"], keep_index=True)

    original_dset = dataset_from_dataframe(
        dataframe,
        feature_names=feature_names + ["dc9"],
        features_dtype="float32",
    )

    shuffled_dset = dataset_from_dataframe(
        dataframe,
        feature_names=feature_names + ["dc9_shuf"],
        features_dtype="float32",
    )

    original_dset.labels = ones(len(original_dset), dtype=float32).unsqueeze(-1)
    shuffled_dset.labels = zeros(len(shuffled_dset), dtype=float32).unsqueeze(-1)

    combo_dset = concat_datasets([original_dset, shuffled_dset])

    return combo_dset


def prep_eval_data(path: Path | str):

    feature_names = [
        "q_squared",
        "cos_theta_mu",
        "cos_theta_k",
        "chi",
    ]

    label_names = ["dc9"]

    dataset = dataset_from_dataframe_parquet(
        path,
        feature_names=feature_names,
        label_names=label_names,
        features_dtype="float32",
        labels_dtype="float32",
    )

    grouped_dataset = group_dataset_by_label(dataset)

    return grouped_dataset
