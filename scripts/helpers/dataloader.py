from math import floor
from dataclasses import dataclass, field

from torch import Tensor, reshape, arange, randperm, meshgrid, randint

#from helpers.util import shuffle_tensor
#from helpers.dataset import Dataset



@dataclass
class SetDataLoader:
    features: Tensor
    labels: Tensor
    batch_size: int
    feature_indices: Tensor = field(init=False)
    label_indices: Tensor = field(init=False)

    def __post_init__(self):
        self.feature_indices, self.label_indices = _batched_set_indices(
            self.features.shape[0],
            self.features.shape[1],
            self.batch_size
        )

    def __len__(self,):
        return len(self.label_indices)

    def __iter__(self):
        self.batch_index = 0
        return self

    def __next__(self):
        if self.batch_index >= len(self):
            self.feature_indices, self.label_indices = _batched_set_indices(
                self.features.shape[0],
                self.features.shape[1],
                self.batch_size
            )
            raise StopIteration

        feature_batch_indices = (indices[self.batch_index] for indices in self.feature_indices)
        label_batch_indices = self.label_indices[self.batch_index]

        batch_features = self.features[*feature_batch_indices]
        batch_labels = self.labels[label_batch_indices]
    
        self.batch_index += 1
        return batch_features, batch_labels

#
#@dataclass
#class DataLoader:
#
#    dataset: Dataset
#    batch_size: int
#    shuffle: bool = True
#    batched_indices: Tensor = field(init=False)
#
#    def __post_init__(self):
#        self.batched_indices = _generate_batched_indices(
#            len(self.dataset), self.batch_size, self.shuffle
#        )
#
#    def __len__(
#        self,
#    ):
#        return len(self.batched_indices)
#
#    def __iter__(
#        self,
#    ):
#        self.index = 0
#        return self
#
#    def __next__(
#        self,
#    ):
#        if self.index >= len(self):
#            self.batched_indices = _make_batched_indices(
#                len(self.dataset), self.batch_size, self.shuffle
#            )
#            raise StopIteration
#
#        batch_indices = self.batched_indices[self.index]
#        batch_features = self.dataset.features[batch_indices]
#        batch_labels = (
#            None if self.dataset.labels is None else self.dataset.labels[batch_indices]
#        )
#
#        self.index += 1
#        return batch_features, batch_labels
#

# def _make_batched_indices(
#    dataset_size: int,
#    batch_size: int,
#    shuffle: bool,
#) -> Tensor:
#
#    assert isinstance(dataset_size, int) 
#    assert isinstance(batch_size, int)
#    assert isinstance(shuffle, bool)
#    assert dataset_size >= batch_size
#
#    indices = arange(dataset_size)
#    if shuffle:
#        indices = shuffle_tensor(indices)
#
#    num_batches = floor(dataset_size / batch_size)
#    truncated_dataset_size = num_batches * batch_size
#    truncated_indices = indices[:truncated_dataset_size]
#
#    batched_indices = reshape(truncated_indices, shape=(num_batches, batch_size))
#    return batched_indices


def _batched_set_indices(num_sets, num_per_set, batch_size):

    if num_sets < batch_size:
        raise ValueError
    
    set_indices = arange(num_sets)
    event_indices = arange(num_per_set)

    shuffled_set_indices = set_indices[randperm(len(set_indices))]
    bootstrapped_event_indices = event_indices[randint(len(event_indices), (len(event_indices),))]

    batching_remainder = num_sets % batch_size
    num_sets_reduced = num_sets - batching_remainder
    reduced_set_indices = shuffled_set_indices[:num_sets_reduced+1]

    set_indices_grid, event_indices_grid = meshgrid(reduced_set_indices, bootstrapped_event_indices, indexing='ij')
    batched_set_indices_grid = set_indices_grid.reshape(-1, batch_size, num_per_set)
    batched_event_indices_grid = event_indices_grid.reshape(-1, batch_size, num_per_set)
    feature_indices = (batched_set_indices_grid, batched_event_indices_grid)
    
    label_indices = reduced_set_indices.reshape(-1, batch_size)

    return feature_indices, label_indices


def test__batched_set_indices():

    batch_index = _batched_set_indices(5, 3, 2)
