
from pathlib import Path

from btokstll_sbi_tools.data_prep import prep_data


in_data_dir = Path("./data/raw/")
out_file_path = Path("./data/combined.parquet")

prep_data(
    in_data_dir, 
    out_file_path
)
