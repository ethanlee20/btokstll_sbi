
from torch import Tensor, all


def _to_tuple_of_ints(
    input:int|tuple[int, ...]|None
)-> tuple[int, ...]:
    if not isinstance(input, int|tuple|None):
        raise ValueError
    output = (
        () if input is None
        else (input,) if isinstance(input, int)
        else input if isinstance(input, tuple)
        else None
    )
    if output is None:
        raise ValueError(f"Bad input: {input}")
    for i in output:
        if not isinstance(i, int):
            raise ValueError(
                f"Input not converted to tuple of ints\n"
                f"Input: {input}\n"
                f"Output: {output}"
            )
    return output


def _all(
    tensor:Tensor, 
    keep_dims:int|tuple|None=None,
) -> Tensor:
    """
    torch.all but specify dimensions to not reduce.
    """
    keep_dims = _to_tuple_of_ints(keep_dims)
    num_dims = tensor.dim()
    dims = range(num_dims)
    for d in keep_dims:
        if d not in dims:
            raise ValueError(
                "Given non-existant dims"
                f" for this tensor: {keep_dims}"
            )
    reduce_dims = tuple(
        d for d in dims 
        if d not in keep_dims
    )
    result = all(tensor, dim=reduce_dims)
    return result


def group(
    data:Tensor, 
    by:Tensor,
) -> list[Tensor]:
    uniques = by.unique(dim=0)
    uniques = uniques.unsqueeze(dim=1)
    selection = uniques == by
    selection = _all(
        selection, 
        keep_dims=(0,1),
    )
    grouped = [data[i] for i in selection]
    return grouped

