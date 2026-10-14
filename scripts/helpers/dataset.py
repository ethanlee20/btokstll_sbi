from math import floor
from pathlib import Path
from dataclasses import dataclass

from torch import Tensor, cat
from pandas import DataFrame, Series, read_parquet

from .util import tensor_from_pandas, group


@dataclass
class Dataset:
    observables: Tensor
    parameters : Tensor | None = None

    def __post_init__(self):
        self._assert_reasonable()

    def __len__(self) -> int:
        self._assert_reasonable()
        return len(self.observables)

    def _assert_reasonable(self):
        assert len(self.observables) == len(self.parameters)

    @classmethod
    def from_pandas(cls, observables, parameters, dtype="float32"):
        observables = tensor_from_pandas(observables, dtype=dtype) 
        parameters = tensor_from_pandas(parameters, dtype=dtype) 
        out = cls(observables=observables, parameters=parameters)
        return out

    def iloc(self, start, stop):
        observables = self.observables[start:stop]
        parameters = self.parameters[start:stop]
        out = Dataset(observables=observables, parameters=parameters)
        return out 

    def copy(self):
        observables = self.observables.detach().clone()
        parameters = self.parameters.detach().clone()
        out = Dataset(observables=observables, parameters=parameters)
        return out

    def train_val_split(self, val_frac): 
        num_total = len(self)
        num_val = floor(num_total * val_frac)
        val = self.iloc(0, num_val).copy()
        train = self.iloc(num_val, num_total).copy()
        return train, val

    def means(self):
        observables_mean = self.observables.mean(dim=0)
        parameters_mean = self.parameters.mean(dim=0)
        out = cat([observables_mean, parameters_mean], dim=0)
        return out

    def stdevs(self):
        observables_stdev = self.observables.std(dim=0)
        parameters_stdev = self.parameters.std(dim=0)
        out = cat([observables_stdev, parameters_stdev], dim=0)
        return out


# def _check_all_labeled(datasets: list[Dataset]) -> bool:
#     assert datasets
#     labels_set = set([d.labels for d in datasets])
#     if None in labels_set:
#         return False
#     return True


# def _check_some_labeled_some_not(datasets: list[Dataset]) -> bool:
#     assert datasets
#     labels_set = set([d.labels for d in datasets])
#     if len(labels_set) > 1 and None in labels_set:
#         return True
#     return False


# def concat_datasets(datasets: list[Dataset]) -> Dataset:
#     some_labeled_some_not = _check_some_labeled_some_not(datasets)
#     if some_labeled_some_not:
#         raise ValueError("Datasets must be either all labeled or all unlabeled.")
#     
#     combined_features = cat([d.features for d in datasets])
#     all_labeled = _check_all_labeled(datasets)
#     if all_labeled:
#         combined_labels = cat([d.labels for d in datasets])
# 
#     combined = (
#         Dataset(features=combined_features, labels=combined_labels)
#         if all_labeled
#         else Dataset(combined_features)
#     )
#     return combined


# def dataset_from_pandas(
#     features: DataFrame | Series,
#     labels: DataFrame | Series | None = None,
#     features_dtype: str | None = None,
#     labels_dtype: str | None = None,
# ) -> Dataset:
# 
#     features_tensor = tensor_from_pandas(features, dtype=features_dtype)
#     if labels is not None:
#         labels_tensor = tensor_from_pandas(labels, dtype=labels_dtype)
# 
#     dataset = (
#         Dataset(features=features_tensor, labels=labels_tensor)
#         if labels is not None
#         else Dataset(features_tensor)
#     )
#     return dataset


# def dataset_from_dataframe(
#     dataframe: DataFrame,
#     feature_names: str | list[str],
#     label_names: str | list[str] | None = None,
#     features_dtype: str | None = None,
#     labels_dtype: str | None = None,
# ) -> Dataset:
# 
#     features_pandas = dataframe[feature_names]
#     if label_names is not None:
#         labels_pandas = dataframe[label_names]
# 
#     dataset = (
#         dataset_from_pandas(
#             features=features_pandas,
#             labels=labels_pandas,
#             features_dtype=features_dtype,
#             labels_dtype=labels_dtype,
#         )
#         if label_names is not None
#         else dataset_from_pandas(
#             features_pandas,
#             features_dtype=features_dtype,
#         )
#     )
#     return dataset


# def dataset_from_dataframe_parquet(
#     path: Path | str,
#     feature_names: str | list[str],
#     label_names: str | list[str] | None = None,
#     features_dtype: str | None = None,
#     labels_dtype: str | None = None,
# ) -> Dataset:
# 
#     dataframe = read_parquet(path)
# 
#     dataset = dataset_from_dataframe(
#         dataframe,
#         feature_names=feature_names,
#         label_names=label_names,
#         features_dtype=features_dtype,
#         labels_dtype=labels_dtype,
#     )
#     return dataset


# def group_dataset_by_label(dataset: Dataset):
#     assert dataset.labels is not None
#     out = dataset.copy()
#     out.features = group(dataset.features, by=dataset.labels)
#     out.labels = group(dataset.labels, by=dataset.labels)
#     out.labels = out.labels.unique(dim=1).squeeze()
#     assert out.labels.dim() == 1
#     assert len(out.features) == len(out.labels)
#     return out
