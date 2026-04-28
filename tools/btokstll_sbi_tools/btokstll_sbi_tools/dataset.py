
from typing import Any
from pathlib import Path
from dataclasses import dataclass, asdict

from pandas import DataFrame, Series, Index, read_parquet
from torch import (
    Tensor,
)

from .scaling import std_scale
from .interval import Interval
from .pandas_ import torch_tensor_from_pandas
from .binning import make_bins, to_bins, calc_binned_label_reweights


@dataclass
class Dataset:
    features: Tensor|None = None
    labels: Tensor|None = None

    @classmethod
    def from_pandas(
        cls,
        features:DataFrame|Series|None=None,
        labels:DataFrame|Series|None=None,
        features_dtype:str|None=None,
        labels_dtype:str|None=None,
    ):
        if features is not None:
            features = torch_tensor_from_pandas(
                features,
                dtype=features_dtype
            )
        if labels is not None:
            labels = torch_tensor_from_pandas(
                labels,
                dtype=labels_dtype
            )
        return cls(
            features=features, 
            labels=labels
        )
    
    @classmethod
    def from_pandas_parquet_file(
        cls, 
        path:Path|str, 
        feature_names:list[str]|None=None, 
        label_names:list[str]|None=None,
        features_dtype:str|None=None,
        labels_dtype:str|None=None,
    ):
        dataframe = read_parquet(path)
        features = (
            None 
            if feature_names is None 
            else dataframe[feature_names]
        )
        labels = (
            None
            if label_names is None
            else dataframe[label_names]
        )
        return cls.from_pandas(
            features=features,
            labels=labels,
            features_dtype=features_dtype, 
            labels_dtype=labels_dtype,
        )

    def _ensure_lengths_match(
        self
    ) -> None:
        if (
            self.features is not None 
            and self.labels is not None
        ):
            if (
                len(self.features) 
                != len(self.labels)
            ): 
                raise ValueError(
                    "Features and labels"
                    " must be the same length."
                )

    def __len__(
        self,
    ) -> int: 
        self._ensure_lengths_match()
        return len(self.labels)
    
    def __eq__(
        self,
        other,
    ) -> bool:
        if isinstance(other, Dataset):
            return (
                (self.features == other.features).all() 
                and (self.labels == other.labels).all()
            )
        return False


@dataclass
class Dataset_Set_File_Paths:
    train:str|Path|None = None
    val:str|Path|None = None
    test:str|Path|None = None

    def __post_init__(self):
        if self.train is not None:
            self.train = Path(self.train)
        if self.val is not None:
            self.val = Path(self.val)
        if self.test is not None:
            self.test = Path(self.test)


@dataclass
class DatasetSet:
    train: Dataset|None = None
    val: Dataset|None = None
    test: Dataset|None = None

    @classmethod
    def from_pandas_parquet_files(
        cls,
        train_path:str|Path|None,
        val_path:str|Path|None,
        test_path:str|Path|None,
        feature_names:list[str]|None=None,
        label_names:list[str]|None=None,
        features_dtype:str|None=None,
        labels_dtype:str|None=None,
    ):
        paths = {
            "train": train_path,
            "val": val_path,
            "test": test_path,
        }
        datasets = {
            split: Dataset.from_pandas_parquet_file(
                path, 
                feature_names=feature_names, 
                label_names=label_names, 
                features_dtype=features_dtype, 
                labels_dtype=labels_dtype
            ) 
            for split, path in paths.items()
        }
        return cls(
            train=datasets["train"], 
            val=datasets["val"], 
            test=datasets["test"],
        )
    
    def apply_std_scale(
        self,
        scale_features:bool=True,
        scale_labels:bool=False,
        dim:int=0,
    ) -> None:
        
        if self.val is not None:
            self.val.std_scale(
                self.train, 
                scale_features=scale_features, 
                scale_labels=scale_labels, 
                dim=dim,
            )
        if self.train is not None:
            self.train.std_scale(
                self.train,
                scale_features=scale_features,
                scale_labels=scale_labels,
                dim=dim,
            )