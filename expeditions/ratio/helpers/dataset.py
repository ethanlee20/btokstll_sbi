from pathlib import Path
from dataclasses import dataclass

from torch import Tensor, cat
from pandas import DataFrame, Series, read_parquet

from .util import tensor_from_pandas


@dataclass
class Dataset:
    features: Tensor
    labels: Tensor | None = None

    def __len__(self) -> int:
        if self.labels is not None:
            assert len(self.features) == len(self.labels)
        return len(self.features)


def concat_datasets(dsets: list[Dataset]) -> Dataset:

    combined_features = cat([d.features for d in dsets])

    there_are_labels = set([d.labels for d in dsets]) != {None}
    if there_are_labels:
        combined_labels = cat([d.labels for d in dsets])

    combined_dset = (
        Dataset(features=combined_features, labels=combined_labels)
        if there_are_labels
        else Dataset(combined_features)
    )

    return combined_dset


def dataset_from_pandas(
    features: DataFrame | Series,
    labels: DataFrame | Series | None = None,
    features_dtype: str | None = None,
    labels_dtype: str | None = None,
) -> Dataset:

    features_tensor = tensor_from_pandas(features, dtype=features_dtype)

    if labels is not None:
        labels_tensor = tensor_from_pandas(labels, dtype=labels_dtype)

    dset = (
        Dataset(features=features_tensor, labels=labels_tensor)
        if labels is not None
        else Dataset(features_tensor)
    )

    return dset


def dataset_from_dataframe(
    dataframe: DataFrame,
    feature_names: str | list[str],
    label_names: str | list[str] | None = None,
    features_dtype: str | None = None,
    labels_dtype: str | None = None,
) -> Dataset:

    features_pandas = dataframe[feature_names]
    if label_names is not None:
        labels_pandas = dataframe[label_names]

    dset = (
        dataset_from_pandas(
            features=features_pandas,
            labels=labels_pandas,
            features_dtype=features_dtype,
            labels_dtype=labels_dtype,
        )
        if label_names is not None
        else dataset_from_pandas(
            features_pandas,
            features_dtype=features_dtype,
        )
    )

    return dset


def dataset_from_dataframe_parquet(
    path: Path | str,
    feature_names: str | list[str],
    label_names: str | list[str] | None = None,
    features_dtype: str | None = None,
    labels_dtype: str | None = None,
) -> Dataset:

    dataframe = read_parquet(path)

    dset = dataset_from_dataframe(
        dataframe,
        feature_names=feature_names,
        label_names=label_names,
        features_dtype=features_dtype,
        labels_dtype=labels_dtype,
    )

    return dset
