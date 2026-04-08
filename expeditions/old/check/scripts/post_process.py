
# run locally

from pathlib import Path

from btokstll_sbi_tools.sim.calculation import calculate_B_to_K_star_l_l_features
from btokstll_sbi_tools.sim.generation import combine_files, Uniform_Delta_Wilson_Coefficient_Distribution
from btokstll_sbi_tools.util.misc import Interval


main_data_dir = Path("../data/")
lepton_flavor = "mu"

distribution = Uniform_Delta_Wilson_Coefficient_Distribution(
    delta_c_7_bounds=Interval(0.0, 0.0), 
    delta_c_9_bounds=Interval(0.0, 0.0), 
    delta_c_10_bounds=Interval(0.0, 0.0)
)

dataframe = combine_files(main_data_dir.joinpath("raw"), dist=distribution, split="train")

dataframe = calculate_B_to_K_star_l_l_features(
    dataframe, 
    lepton_flavor,
)

output_file_name = f"combined.parquet"
output_file_path = main_data_dir.parent.joinpath(
    output_file_name
)
dataframe.to_parquet(output_file_path)





