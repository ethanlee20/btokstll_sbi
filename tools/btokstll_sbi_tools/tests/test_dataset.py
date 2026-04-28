
from pytest import mark, fixture
from torch import tensor, float
from pandas import Series

from btokstll_sbi_tools.dataset import Dataset, DatasetSet


class TestDataset:

    @fixture
    def basic_dataset(self,):
        features = tensor([1,2,3], dtype=float)
        labels = tensor([1,2,1], dtype=float)
        return Dataset(
            features=features, 
            labels=labels,
        )
    
    def test_basic(self, basic_dataset):
        assert (basic_dataset.features == tensor([1,2,3], dtype=float)).all()
        assert (basic_dataset.labels == tensor([1,2,1], dtype=float)).all()

    def test_len(self, basic_dataset):
        assert len(basic_dataset) == 3
    
    @fixture
    def pandas_features(self):
        return Series([1,2,3], dtype="float32")
    
    @fixture 
    def pandas_labels(self):
        return Series([1,2,1], dtype="float32")

    def test_from_pandas(self, pandas_features, pandas_labels, basic_dataset):
        dataset_from_pandas = Dataset.from_pandas(
            pandas_features, 
            pandas_labels,
        )
        assert dataset_from_pandas == basic_dataset


class TestDatasetSet:

    @fixture
    def basic_train_dataset(self,):
        return Dataset(
            features=tensor([1,2,3], dtype=float),
            labels=tensor([4,5,6], dtype=float),
        )
    
    @fixture
    def basic_val_dataset(self,):
        return Dataset(
            features=tensor([4,3,3], dtype=float),
            labels=tensor([3,6,1], dtype=float),
        )
    
    @fixture
    def basic_test_dataset(self,):
        return Dataset(
            features=tensor([0,-2,3], dtype=float),
            labels=tensor([-3,5,0], dtype=float),
        )

    @fixture
    def basic_dataset_set(
        self, 
        basic_train_dataset, 
        basic_val_dataset, 
        basic_test_dataset, 
    ):
        return DatasetSet(
            train=basic_train_dataset,
            val=basic_val_dataset,
            test=basic_test_dataset,
        )
    
    def test_basic_dataset_set(
        self, 
        basic_dataset_set, 
        basic_train_dataset,
        basic_val_dataset, 
        basic_test_dataset,
    ):
        assert basic_dataset_set.train == basic_train_dataset
        assert basic_dataset_set.val == basic_val_dataset
        assert basic_dataset_set.test == basic_test_dataset
