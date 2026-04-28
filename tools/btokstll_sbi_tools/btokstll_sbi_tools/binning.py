
from torch import (
    Tensor, 
    linspace, 
    bucketize,
    bincount,
)

from .interval import Interval


def make_bins(
    interval:Interval,
    num_bins:int,
) -> tuple[Tensor, Tensor]:
    bin_edges = linspace(
        interval.left,
        interval.right,
        num_bins+1,
    )
    bin_mids = (
        bin_edges[:-1] + 0.5 
        * (bin_edges[1] - bin_edges[0])
    )
    return bin_edges, bin_mids


def to_bins(
    data:Tensor,
    bin_edges:Tensor,
    eps:float=1e-2
) -> Tensor:
    if any(
        (data < bin_edges[0]) 
        | (data > bin_edges[-1])
    ):
        raise ValueError(
            "Data outside of binned interval."
        )
    bin_edges[0] -= abs(Tensor([eps])).item()
    binned_data = bucketize(
        input=data, 
        boundaries=bin_edges, 
        out_int32=False, 
        right=False
    ) - 1
    return binned_data


def calc_binned_label_reweights(
    labels:Tensor,
    num_labels:int,
) -> Tensor:
    """
    Calculate class weights for reweighting 
    classes to uniform distribution.
    """
    bin_counts = bincount(
        input=labels, 
        minlength=num_labels,
    )
    if (bin_counts == 0).any():
        raise ValueError(
            f"Some bins are empty!"
            f" Bin counts:\n{bin_counts}"
        )
    inverse_bin_counts = 1 / bin_counts
    return inverse_bin_counts / sum(inverse_bin_counts)

