
from pandas import (
    DataFrame, 
    read_parquet,
)


def filter(
    data: DataFrame,
    columns : list[string] = [
        "q_squared",
        "cos_theta_mu", 
        "cos_theta_k", 
        "chi",
        "delta_wc_values_dc9"
    ],
    reduce_candidates: bool = True
):
    if reduce_candidates:
        data = data[
            data["__candidate__"] == 0
        ]
    data = data[columns]
    return data


def main():
    df = read_parquet(
        "data/vary_dc9_train.parquet"
    )
    df = filter(df)
    df = df.rename(columns={"delta_wc_values_dc9": "dc9"})
    df.to_parquet(
        "data/train_small.parquet"
    )


if __name__ == "__main__":
    main()
