from pathlib import Path

from pandas import read_parquet
from torch import ones, zeros, int64

from .util import shuffle_pandas
from .dataset import concat_datasets, dataset_from_dataframe


def prep_data(
    path: Path | str,
    features_dtype="float32",
):

    feature_names = [
        "q_squared",
        "cos_theta_mu",
        "cos_theta_k",
        "chi",
    ]

    dataframe = read_parquet(path)
    dataframe["dc9_shuf"] = shuffle_pandas(dataframe["dc9"])

    original_dset = dataset_from_dataframe(
        dataframe,
        feature_names=feature_names + ["dc9"],
        features_dtype=features_dtype,
    )

    shuffled_dset = dataset_from_dataframe(
        dataframe,
        feature_names=feature_names + ["dc9_shuf"],
        features_dtype=features_dtype,
    )

    original_dset.labels = ones(len(original_dset), dtype=int64)
    shuffled_dset.labels = zeros(len(shuffled_dset), dtype=int64)

    combo_dset = concat_datasets([original_dset, shuffled_dset])

    return combo_dset
