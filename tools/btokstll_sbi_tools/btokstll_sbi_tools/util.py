
from torch import Tensor, any, abs, bucketize


def bin_(
    data:Tensor,
    bin_edges:Tensor,
    eps:float=1e-2
) -> Tensor:
    
    if any((data < bin_edges[0]) | (data > bin_edges[-1])):
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