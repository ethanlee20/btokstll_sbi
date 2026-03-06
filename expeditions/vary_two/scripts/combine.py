
from pathlib import Path

from btokstll_sbi_tools.sim import combine_files


out_file_path = Path("../data/combo.parquet")

dirs = list(Path("../data/raw").glob("*/"))

combine_files(
    dirs, 
    out_file_path=out_file_path
)