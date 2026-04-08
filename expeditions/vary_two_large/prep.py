
from pathlib import Path

from btokstll_sbi_tools.prep import prep_data


in_data_dirs = Path("./data/raw/").glob("*/")
out_dir = Path("./data/")

for dir_ in in_data_dirs:
    prep_data(
        dir_, 
        out_dir.joinpath(f"{dir_.name}.parquet")
    )
