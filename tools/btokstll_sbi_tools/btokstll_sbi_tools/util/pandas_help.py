
from pathlib import Path

from pandas import read_parquet, concat, DataFrame, Series


def read_parquets(
    paths:list[Path|str],
) -> DataFrame|Series:
    
    dfs = [
        read_parquet(path) 
        for path in paths
    ]
    return concat(dfs)
