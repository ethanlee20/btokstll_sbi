
from pathlib import Path

from btokstll_sbi_tools.preprocess import prep_data


dir_names = ("vary_dc9_train", "vary_dc9_val")

for d in dir_names:
    prep_data(
        Path(f"data/raw/{d}"), 
        Path(f"data/{d}.parquet")
    )