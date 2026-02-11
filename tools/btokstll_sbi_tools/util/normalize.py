
import torch
import pandas


def normalize_using_reference_data(data:pandas.DataFrame|pandas.Series, reference:pandas.DataFrame|pandas.Series):

    """Standard scale a dataset using the mean and standard deviation of a reference dataset."""

    means = reference.mean()
    stds = reference.std()
    normalized = (data - means) / stds
    return normalized






    

