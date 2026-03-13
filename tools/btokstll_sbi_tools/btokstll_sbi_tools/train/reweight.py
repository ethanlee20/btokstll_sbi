
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
    if (bin_counts == 0).any():
        raise ValueError(
            f"Some bins are empty!"
            f" Bin counts:\n{bin_counts}"
        )
    inverse_bin_counts = 1 / bin_counts
    return inverse_bin_counts / sum(inverse_bin_counts)

