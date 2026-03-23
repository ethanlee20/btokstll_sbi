
from argparse import ArgumentParser
from pathlib import Path

from pandas import read_parquet

from .util import calculate_B_to_K_star_l_l_features, combine_files


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Process generator output."
    )
    parser.add_argument("in_dir", type=Path)
    parser.add_argument("out_dir", type=Path)

    args = parser.parse_args()

    for p in args.in_dir.rglob("._*"):
        p.unlink()
    
    for dir_ in args.in_dir.glob("*/"):
        out_path = args.out_dir.joinpath(f"{dir_.name}.parquet")
        combine_files(
            list(dir_.glob("*/")),
            out_file_path=out_path,
        )
        df = read_parquet(out_path)
        df = calculate_B_to_K_star_l_l_features(df, "mu")
        df.to_parquet(out_path)