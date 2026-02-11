
# run this using personal venv on kekcc

import btokstll_sbi_tools


parent_data_dir = "path_to_data_dir"
lepton_flavor = "mu"

num_trials = {"train":..., "val": ...}
num_events_per_trial = {"train":..., "val": ...}
num_subtrials = {"train":..., "val": ...}

intervals = {
    "vary_dc7": {"interval_dc7":..., "interval_dc9":..., "interval_dc10":...},
    "vary_dc9": {"interval_dc7":..., "interval_dc9":..., "interval_dc10":...},
    "vary_dc10": {"interval_dc7":..., "interval_dc9":..., "interval_dc10":...},
}

setups = []
for split in ["train", "val"]:
    setup_vary_dc7 = dict(
        num_trials=num_trials[split],
        num_subtrials=num_subtrials[split],
        num_events_per_trial=num_events_per_trial[split],
        split=split,
        interval_dc7=(-0.2, 0.2),
        interval_dc9=(0, 0),
        interval_dc10=(0, 0),
    )
    setup_vary_dc9 = dict(
        num_trials=num_trials[split],
        num_subtrials=num_subtrials[split],
        num_events_per_trial=num_events_per_trial[split],
        split=split,
        interval_dc7=(0, 0),
        interval_dc9=(-2.0, 1.0),
        interval_dc10=(0, 0),
    )
    setup_vary_dc10 = dict(
        num_trials=num_trials[split],
        num_subtrials=num_subtrials[split],
        num_events_per_trial=num_events_per_trial[split],
        split=split,
        interval_dc7=(0, 0),
        interval_dc9=(0, 0),
        interval_dc10=(-1.0, 1.0),
    )
    setups.append(setup_vary_dc7)
    setups.append(setup_vary_dc9)
    setups.append(setup_vary_dc10)


for setup in setups:
    btokstll_sbi_tools.simulation.setup.setup(
        parent_data_dir=parent_data_dir,
        lepton_flavor=lepton_flavor,
        **setup
    )
