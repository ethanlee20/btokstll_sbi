
# Run using Alexei's environment on kekcc

import btokstll_sbi_tools


parent_data_dir = "path_to_data_dir"
path_to_sim_steer_file = "path"
path_to_recon_steer_file = "path"

btokstll_sbi_tools.simulation.generate.run_jobs(
    parent_data_dir=parent_data_dir,
    path_to_sim_steer_file=path_to_sim_steer_file,
    path_to_recon_steer_file=path_to_recon_steer_file
)