
from torch import Tensor, mean, std


def std_scale(
    data:Tensor, 
    reference:Tensor,
) -> Tensor:
    """
    Standard scale a dataset using 
    the mean and standard deviation
    of a reference dataset.
    """

    var_means = mean(reference, dim=0)
    var_stds = std(reference, dim=0)
    return (data - var_means) / var_stds






    

