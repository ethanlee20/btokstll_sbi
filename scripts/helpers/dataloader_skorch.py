from math import floor
from dataclasses import dataclass, field

import numpy as np
from torch.utils.data import Subset
from skorch.dataset import Dataset


@dataclass
class DataLoader:

    data: Subset | Dataset
    batch_size: int
    shuffle: bool = True
    batched_indices: np.ndarray = field(init=False)

    def __post_init__(self):
        self.batched_indices = _generate_batched_indices(
            self.data, self.batch_size, self.shuffle
        )

    def __len__(self):
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
                self.data, self.batch_size, self.shuffle
            )
            raise StopIteration

        batch_indices = self.batched_indices[self.index]
        dataset = self.data.dataset if isinstance(self.data, Subset) else self.data
        batch_features = dataset.X[batch_indices]
        batch_labels = dataset.y[batch_indices] if dataset.y is not None else None

        self.index += 1
        return batch_features, batch_labels


def _generate_batched_indices(
    data: Subset | Dataset,
    batch_size: int,
    shuffle: bool,
) -> np.ndarray:

    indices = (
        data.indices if isinstance(data, Subset) 
        else np.arange(len(data)) if isinstance(data, Dataset) 
        else None
    )
    if indices is None: 
        raise ValueError
    
    if shuffle:
        indices = np.random.permutation(indices)

    num_batches = floor(len(data) / batch_size)
    truncated_dataset_size = num_batches * batch_size
    truncated_indices = indices[:truncated_dataset_size]

    batched_indices = np.reshape(truncated_indices, shape=(num_batches, batch_size))
    return batched_indices