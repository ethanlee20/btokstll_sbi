from pytest import mark
from torch import Tensor

from btokstll_sbi_tools.group import (
    _to_tuple_of_ints, 
    _all, 
    group,
)

@mark.parametrize(
    "input, expected_output", 
    [
        ((1,2), (1,2)),
        (1, (1,)),
        (None, ()),
    ]
)
def test_to_tuple_of_ints(
    input, 
    expected_output,
):
    output = _to_tuple_of_ints(input)
    assert output == expected_output


@mark.parametrize(
    "tensor, keep_dims, expected_output",
    [
        (
            Tensor([True, False, True]),
            0, 
            Tensor([True, False, True])
        ),
        (
            Tensor(
                [
                    [True, False],
                    [False, False], 
                    [True, True],
                ]
            ),
            0,
            Tensor([False, False, True])
        ),
        (
            Tensor(
                [
                    [
                        [True, False], 
                        [True, True],
                    ],
                    [
                        [False, False],
                        [False, False],
                    ], 
                    [
                        [True, True],
                        [True, True],
                    ],
                ]
            ),
            (0, 1),
            Tensor(
                [
                    [False, True], 
                    [False, False], 
                    [True, True],
                ]
            )
        ),
    ]
)
def test_all(
    tensor,
    keep_dims, 
    expected_output,
):
    output = _all(
        tensor=tensor, 
        keep_dims=keep_dims,
    )
    assert (output == expected_output).all()



@mark.parametrize(
    "data, by, expected_output",
    [
        (
            Tensor([1,2,3]), 
            Tensor([0,1,0]), 
            [Tensor([1,3]), Tensor([2])]
        ),
        (
            Tensor([1,2,3]), 
            Tensor([[1,0],[0,1],[1,0]]), 
            [Tensor([2]), Tensor([1,3])]
        ),
    ]
)
def test_group(
    data, 
    by, 
    expected_output,
):
    output = group(data, by)
    for i, j in zip(output, expected_output):
        assert (i == j).all()
