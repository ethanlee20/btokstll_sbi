
from pandas import Series, DataFrame


def std_scale(
    data:DataFrame|Series, 
    reference:DataFrame|Series,
):
    """Standard scale a dataset using the mean and standard deviation of a reference dataset."""

    means = reference.mean()
    stds = reference.std()
    normalized = (data - means) / stds
    return normalized






    

