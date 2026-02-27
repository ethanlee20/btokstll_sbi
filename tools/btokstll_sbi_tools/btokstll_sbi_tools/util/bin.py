
from pandas import Series, DataFrame, cut
from torch import Tensor, all, abs, bucketize


def bin_(
    data:Series, 
    bins,
):
    """Bin data using given bins."""

    binned_indices = cut(
        data,
        bins,
        labels=False,
        include_lowest=True
    )
    binned_intervals = cut(
        data, 
        bins, 
        labels=None, 
        include_lowest=True
    )
    binned_mids = binned_intervals.apply(lambda interval : interval.mid)
    binned = DataFrame(
        {
            "original": data,
            "bin_index": binned_indices, 
            "bin_interval": binned_intervals, 
            "bin_mid": binned_mids
        }
    )
    return binned


def bin_(
    data:Tensor,
    bin_edges:Tensor,
    eps:float=1e-2
) -> Tensor:
    
    if any((data < bin_edges[0]) & (data > bin_edges[-1])):
        raise ValueError(
            "Data outside of binned interval."
        )
    
    bin_edges[0] -= abs(Tensor(eps)).item()

    binned_data = bucketize(
        input=data, 
        boundaries=bin_edges, 
        out_int32=True, 
        right=False
    ) - 1

    return binned_data

    