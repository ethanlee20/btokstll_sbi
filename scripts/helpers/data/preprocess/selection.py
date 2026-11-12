
from pandas import DataFrame


def select_first_candidates(dataframe: DataFrame) -> DataFrame:
    out = dataframe[dataframe["__candidate__"] == 0].copy()
    return out


def select_variables(dataframe: DataFrame, names: list[str]) -> DataFrame: 
    out = dataframe[variables].copy()
    return out 
