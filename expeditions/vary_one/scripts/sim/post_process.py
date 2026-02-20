
# run locally

from pathlib import Path

from btokstll_sbi_tools.simulation.calculation import calculate_B_to_K_star_l_l_features
from btokstll_sbi_tools.simulation.generation import combine_files, Uniform_Delta_Wilson_Coefficient_Distribution
from btokstll_sbi_tools.util.types import Interval


main_data_dir = Path("../../data/data")
lepton_flavor = "mu"

distributions = {
    "vary_c_7": Uniform_Delta_Wilson_Coefficient_Distribution(
        delta_c_7_bounds=Interval(-0.2, 0.2), 
        delta_c_9_bounds=Interval(0.0, 0.0), 
        delta_c_10_bounds=Interval(0.0, 0.0)
    ), 
    "vary_c_9": Uniform_Delta_Wilson_Coefficient_Distribution(
        delta_c_7_bounds=Interval(0.0, 0.0), 
        delta_c_9_bounds=Interval(-2.0, 1.0), 
        delta_c_10_bounds=Interval(0.0, 0.0)
    ), 
    "vary_c_10": Uniform_Delta_Wilson_Coefficient_Distribution(
        delta_c_7_bounds=Interval(0.0, 0.0), 
        delta_c_9_bounds=Interval(0.0, 0.0), 
        delta_c_10_bounds=Interval(-1.0, 1.0)
    ), 
}

for experiment, dist in distributions.items():

    for split in ("train", "val"):

        dataframe = combine_files(main_data_dir, dist=dist, split=split)

        dataframe = calculate_B_to_K_star_l_l_features(
            dataframe, 
            lepton_flavor,
        )

        output_file_name = f"combined_{experiment}_{split}.parquet"
        output_file_path = main_data_dir.parent.joinpath(
            output_file_name
        )
        dataframe.to_parquet(output_file_path)





