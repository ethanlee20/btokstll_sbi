
# run locally

from pathlib import Path

from btokstll_sbi_tools.simulation.calculation import calculate_B_to_K_star_l_l_features
from btokstll_sbi_tools.simulation.generation import combine_files


main_data_dir = Path("../../data")
lepton_flavor = "mu"
output_file_name = "combined.parquet"

dataframe = combine_files(main_data_dir)
dataframe = calculate_B_to_K_star_l_l_features(
    dataframe, 
    lepton_flavor,
)

output_file_path = main_data_dir.joinpath(
    output_file_name
)
dataframe.to_parquet(output_file_path)





