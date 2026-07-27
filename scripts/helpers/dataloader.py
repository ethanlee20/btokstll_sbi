from math import floor
from dataclasses import dataclass, field

from torch import Tensor, reshape, arange

from .util import are_instance, shuffle_tensor
from .dataset import Dataset


@dataclass
class DataLoader:

    dataset: Dataset
    batch_size: int
    shuffle: bool = True
    batched_indices: Tensor = field(init=False)

    def __post_init__(self):
        self.batched_indices = _generate_batched_indices(
            len(self.dataset), self.batch_size, self.shuffle
        )

    def __len__(
        self,
    ):
        return len(self.batched_indices)

    def __iter__(
        self,
    ):
        self.index = 0
        return self

    def __next__(
        self,
    ):
        if self.index >= len(self):
            self.batched_indices = _generate_batched_indices(
                len(self.dataset), self.batch_size, self.shuffle
            )
            raise StopIteration

        batch_indices = self.batched_indices[self.index]

        batch_features = self.dataset.features[batch_indices]

        batch_labels = (
            None if self.dataset.labels is None else self.dataset.labels[batch_indices]
        )

        self.index += 1

        return batch_features, batch_labels


def _generate_batched_indices(
    dataset_size: int,
    batch_size: int,
    shuffle: bool,
) -> Tensor:

    assert are_instance([dataset_size, batch_size], int)
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
