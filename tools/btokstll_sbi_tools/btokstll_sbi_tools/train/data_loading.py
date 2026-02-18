
import numpy
import torch
import pandas


def generate_batched_indices(dataset_size, batch_size, shuffle):

    assert are_instance([dataset_size, batch_size], int)
    assert isinstance(shuffle, bool)
    assert dataset_size > batch_size

    indices = torch.arange(dataset_size)
    if shuffle: 
        indices = indices[torch.randperm(len(indices))]
    num_batches = int(numpy.floor(dataset_size / batch_size))
    batched_indices = torch.reshape(
        indices[:num_batches*batch_size], 
        shape=(num_batches, batch_size)
    )
    return batched_indices


class Data_Loader:

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.dataset_size = len(self.dataset)
        self.batched_indices = generate_batched_indices(
            self.dataset_size, 
            self.batch_size, 
            self.shuffle
        )

    def __len__(self):
        
        return len(self.batched_indices)
    
    def __iter__(self):
        
        self.index = 0
        return self
    
    def __next__(self):

        if self.index >= len(self):
            self.batched_indices = generate_batched_indices(
                self.dataset_size, 
                self.batch_size, 
                self.shuffle
            )
            raise StopIteration
        
        batch_indices = self.batched_indices[self.index]
        batch_features = self.dataset.features[batch_indices]
        batch_labels = self.dataset.labels[batch_indices]

        self.index += 1

        return batch_features, batch_labels