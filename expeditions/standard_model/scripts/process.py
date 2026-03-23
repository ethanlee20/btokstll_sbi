

from pathlib import Path

from pandas import read_parquet

from btokstll_sbi_tools.sim import combine_files, calculate_B_to_K_star_l_l_features


data_dir = Path("../data/raw")
out_path = Path("../data/combo.parquet")


combine_files(
    list(data_dir.glob("*/")),
    out_file_path=out_path,
)
df = read_parquet(out_path)
df = calculate_B_to_K_star_l_l_features(df, "mu")
df.to_parquet(out_path)