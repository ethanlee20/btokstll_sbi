
# run this using personal venv on kekcc

from pathlib import Path

from btokstll_sbi_tools.util.types import Interval
from btokstll_sbi_tools.simulation.generation import (
    Directory_Manager, 
    Sampler,
    Uniform_Delta_Wilson_Coefficient_Distribution
)


main_data_dir = Path("../../data")
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

num_trials = {
    "train": 3, 
    "val": 4
}

num_subtrials = {
    "train": 1,
    "val": 2,
}

num_events_per_trial = {
    "train": 3, 
    "val": 3,
}

for experiment in distributions.keys():

    dist = distributions[experiment]

    sampler = Sampler(dist)

    samples = {
        split: sampler.sample(num) 
        for split, num in num_trials.items() 
    }

    directory_managers = {
        split: Directory_Manager(
            main_data_dir=main_data_dir,
            num_trials=num_trials[split],
            num_subtrials=num_subtrials[split],
            num_events_per_trial=num_events_per_trial[split],
            split=split,
            lepton_flavor=lepton_flavor,
            delta_wilson_coefficient_set_samples=samples[split],
            delta_wilson_coefficient_distribution=dist,
        )
        for split in num_trials.keys()
    }

    for manager in directory_managers.values():
        manager.setup_dirs()

    