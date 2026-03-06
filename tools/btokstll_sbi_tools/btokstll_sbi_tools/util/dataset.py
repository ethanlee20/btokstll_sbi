
from pathlib import Path
from dataclasses import dataclass

from torch import Tensor
from pandas import read_parquet, DataFrame, Series
from .type import to_torch_tensor


@dataclass
class Dataset:

    features: Tensor
    labels: Tensor

    def __postinit__(
        self,
    ):
        assert (
            len(self.features) 
            == len(self.labels)
        )

    def __len__(
        self,
    ) -> int: 
        return len(self.labels)
    
    def to_device(
        self, 
        device:str,
    ) -> None:
        self.features = self.features.to(
            device
        )
        self.labels = self.labels.to(
            device
        ) 
    
    @classmethod
    def from_pandas(
        cls,
        features:DataFrame|Series,
        labels:DataFrame|Series
    ):
        features = to_torch_tensor(features)
        labels = to_torch_tensor(labels)
        return cls(
            features=features, 
            labels=labels
        )
    
    @classmethod
    def from_dataframe_parquet_file(
        cls, 
        path:Path|str, 
        feature_names:list[str], 
        label_name:str
    ):
        df = read_parquet(path)
        features = to_torch_tensor(
            df[feature_names]
        )
        labels = to_torch_tensor(
            df[label_name]
        )
        return cls(
            features=features, 
            labels=labels,
        )


@dataclass
class Dataset_Set:
    
    train: Dataset
    eval: Dataset

    @classmethod
    def from_dataframe_parquet_files(
        cls,
        train_file_path:Path|str,
        eval_file_path:Path|str,
        feature_names:list[str],
        label_name:str,
    ):
        train = Dataset.from_dataframe_parquet_file(
            train_file_path, 
            feature_names=feature_names,
            label_name=label_name,
        )
        eval = Dataset.from_dataframe_parquet_file(
            eval_file_path, 
            feature_names=feature_names,
            label_name=label_name,
        )
        return cls(
            train=train, 
            eval=eval,
        )