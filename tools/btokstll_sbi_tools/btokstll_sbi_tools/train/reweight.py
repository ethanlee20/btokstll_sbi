
from torch import Tensor, bincount, sum


def calculate_reweights_uniform(
    binned_labels:Tensor,
    num_bins:int,
) -> Tensor:
    """
    Calculate class weights for reweighting 
    classes to uniform distribution.
    """

    bin_counts = bincount(
        input=binned_labels, 
        minlength=num_bins,
    )
    inverse_bin_counts = 1 / bin_counts
    return inverse_bin_counts / sum(inverse_bin_counts)

