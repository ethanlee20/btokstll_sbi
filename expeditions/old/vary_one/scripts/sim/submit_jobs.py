
# Run using Alexei's environment on kekcc

from pathlib import Path

from btokstll_sbi_tools.simulation.generation import Job_Submitter


main_data_dir = Path("../../data")
recon_steer_file_path=Path("../../../../tools/items/kekcc/steer_recon.py")
sim_steer_file_path=Path("../../../../tools/items/kekcc/steer_sim.py")

submitter = Job_Submitter(
    main_data_dir=main_data_dir,
    recon_steer_file_path=recon_steer_file_path,
    sim_steer_file_path=sim_steer_file_path,
)

submitter.submit_jobs()


