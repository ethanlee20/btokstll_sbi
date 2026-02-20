
import torch

from btokstll_sbi_tools.util.types import to_torch_tensor, are_instance


class Dataset:

    def __init__(self, features, labels):

        features = to_torch_tensor(features)
        labels = to_torch_tensor(labels)

        assert are_instance([features, labels], torch.Tensor)
        assert len(features) == len(labels)

        self.features = features
        self.labels = labels

    def to(self, arg):

        return Dataset(self.features.to(arg), self.labels.to(arg))

    def __len__(self): 

        return len(self.labels)