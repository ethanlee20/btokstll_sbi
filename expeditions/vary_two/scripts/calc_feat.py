
from pathlib import Path

from pandas import read_parquet

from btokstll_sbi_tools.sim import calculate_B_to_K_star_l_l_features


data_file_path = Path("../data/combo.parquet")

data = read_parquet(data_file_path)

data = calculate_B_to_K_star_l_l_features(data, "mu")

data.to_parquet(data_file_path)
