
from pandas import Series

from ..util import to_torch_tensor


def calculate_class_weights_for_uniform_prior(
    labels:Series,
):
    """Calculate class weights for reweighting classes to uniform distribution."""

    normalized_class_counts = to_torch_tensor(
        labels.value_counts(normalize=True).sort_index()
    )
    weights = 1 / normalized_class_counts
    return weights

