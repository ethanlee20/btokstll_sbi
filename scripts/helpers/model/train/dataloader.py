from math import floor
from dataclasses import dataclass, field

from torch import Tensor, reshape, arange, randperm, meshgrid, randint, zeros, hstack, cat, float32, equal

from helpers.util import shuffle_tensor, bootstrap_tensor
from helpers.dataset import Dataset


@dataclass
class NreCDataLoader:
    dataset: Dataset
    batch_size: int
    k: int
    shuffle: bool = True
    batched_indices: Tensor = field(init=False)

    def __post_init__(self):
        self.batched_indices = _make_batched_indices(
            len(self.dataset), self.batch_size, self.shuffle
        )

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self):
            self.batched_indices = _make_batched_indices(
                len(self.dataset), self.batch_size, self.shuffle
            )
            raise StopIteration

        def make_batch(parameters, observables):
            assert len(parameters) == len(observables)
            batch_size = len(parameters)
            marginal_parameters = bootstrap_tensor(
                parameters, num=2*batch_size*self.k
            )
            marginal_parameters = marginal_parameters.reshape(
                (2*batch_size, self.k, -1)
            )
            batch_parameters = marginal_parameters.detach().clone()
            batch_parameters[0:batch_size, self.k-1] = parameters.squeeze()
            batch_observables = observables.tile((2,1))
            batch_observables = batch_observables.unsqueeze(dim=1).expand(-1, self.k, -1)
            assert batch_observables.shape[:-1] == batch_parameters.shape[:-1]
            is_joint = zeros(len(batch_observables), dtype=float32)  
            is_joint[0:batch_size] = 1
            is_joint = is_joint.unsqueeze(dim=-1)
            return batch_parameters, batch_observables, is_joint
        
        batch_indices = self.batched_indices[self.index]
        raw_batch_parameters = self.dataset.parameters[batch_indices]
        raw_batch_observables = self.dataset.observables[batch_indices]
   
        batch_parameters, batch_observables, is_joint = make_batch(
            parameters=raw_batch_parameters, 
            observables=raw_batch_observables
        )

        self.index += 1
        return batch_parameters, batch_observables, is_joint

    @property
    def num_batches(self,):
        return len(self.batched_indices) 


def _make_batched_indices(
    dataset_size: int,
    batch_size: int,
    shuffle: bool,
) -> Tensor:

    assert isinstance(dataset_size, int) 
    assert isinstance(batch_size, int)
    assert isinstance(shuffle, bool)
    assert dataset_size >= batch_size

    indices = arange(dataset_size)
    if shuffle:
        indices = shuffle_tensor(indices)

    num_batches = floor(dataset_size / batch_size)
    truncated_dataset_size = num_batches * batch_size
    truncated_indices = indices[:truncated_dataset_size]

    batched_indices = reshape(truncated_indices, shape=(num_batches, batch_size))
    return batched_indices
