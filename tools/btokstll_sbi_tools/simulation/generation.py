
import pathlib
import subprocess
import json
import dataclasses

import pandas
import numpy
import uproot
import tqdm

from ..utilities.types import (
    safer_convert_to_int, 
    are_instance,
    to_pandas_interval,
    Interval
)

@dataclasses.dataclass
class Delta_Wilson_Coefficient_Set:
    delta_c_7: float
    delta_c_9: float
    delta_c_10: float


@dataclasses.dataclass
class Uniform_Delta_Wilson_Coefficient_Distribution:
    delta_c_7_bounds:Interval 
    delta_c_9_bounds:Interval 
    delta_c_10_bounds:Interval
    

def sample(
    self,
    num_samples:int, 
    rng_seed=None
):
    rng = numpy.random.default_rng(rng_seed)
    samples = rng.uniform(
        low=[
            self.delta_c_7_bounds.left, 
            self.delta_c_9_bounds.left, 
            self.delta_c_10_bounds.left
        ],
        high=[
            self.delta_c_7_bounds.right, 
            self.delta_c_9_bounds.right, 
            self.delta_c_10_bounds.right
        ],
        size=(num_samples, 3)
    )
    samples = [
        Delta_Wilson_Coefficient_Set(*i) 
        for i in samples
    ]
    return samples


class Trial_Metadata:

    def __init__(
        self,
        trial_num:int,
        num_events_per_trial:int,
        num_subtrials:int,
        split:str,
        lepton_flavor:str,
        delta_wilson_coefficient_set:Delta_Wilson_Coefficient_Set,
        delta_wilson_coefficient_distribution:Uniform_Delta_Wilson_Coefficient_Distribution,
    ):
        self.trial_num = trial_num
        self.num_events_per_trial = num_events_per_trial
        self.num_subtrials=num_subtrials
        self.split = split
        self.lepton_flavor = lepton_flavor
        self.delta_wilson_coefficient_set = delta_wilson_coefficient_set
        self.delta_wilson_coefficient_distribution = delta_wilson_coefficient_distribution
    
    def to_json_file(self, path):
        metadata_dict = {
            "trial_num":self.trial_num,
            "num_events_per_trial":self.num_events_per_trial,
            "num_subtrials":self.num_subtrials,
            "split":self.split,
            "lepton_flavor":self.lepton_flavor,
            "delta_wilson_coefficient_set":self.delta_wilson_coefficient_set,
            "delta_wilson_coefficient_distribution":self.delta_wilson_coefficient_distribution
        }
        with open(path, 'x') as file:
            json.dump(metadata_dict, file)

    @classmethod
    def from_json_file(
        cls, 
        path
    ):
        series = pandas.read_json(path, typ="series")
        delta_wilson_coefficient_set = Delta_Wilson_Coefficient_Set(
            **series[["delta_c_7", "delta_c_9", "delta_c_10"]]
        )
        delta_wilson_coefficient_distribution = Uniform_Delta_Wilson_Coefficient_Distribution(
            delta_c_7_bounds=(series["delta_c_7_bounds_left"], series["delta_c_7_bounds_right"]),
            delta_c_9_bounds=(series["delta_c_9_bounds_left"], series["delta_c_9_bounds_right"]),
            delta_c_10_bounds=(series["delta_c_10_bounds_left"], series["delta_c_10_bounds_right"]),
        )
        return cls(
            **series[[
                "trial_num", 
                "num_events_per_trial", 
                "num_subtrials", 
                "split", 
                "lepton_flavor"
            ]],
            delta_wilson_coefficient_set=delta_wilson_coefficient_set,
            delta_wilson_coefficient_distribution=delta_wilson_coefficient_distribution
        )


    


def make_metadata(
    trial, 
    num_events_per_trial,
    num_subtrials,
    split,
    lepton_flavor,
    dc7, 
    dc9, 
    dc10, 
    interval_dc7, 
    interval_dc9, 
    interval_dc10,
):
    assert num_events_per_trial % num_subtrials == 0 
    
    metadata = {
        "trial": trial,
        "num_subtrials": num_subtrials,
        "num_events_per_trial": num_events_per_trial,
        "num_events_per_subtrial": safer_convert_to_int(num_events_per_trial / num_subtrials),
        "split": split,
        "lepton_flavor": lepton_flavor, 
        "dc7": dc7,
        "dc9": dc9,
        "dc10": dc10,
        "interval_dc7_lb": interval_dc7.left,
        "interval_dc7_ub": interval_dc7.right,
        "interval_dc9_lb": interval_dc9.left,
        "interval_dc9_ub": interval_dc9.right,
        "interval_dc10_lb": interval_dc10.left,
        "interval_dc10_ub": interval_dc10.right,
    }
    metadata = pandas.Series(metadata)
    return metadata


def make_trial_dir_name(trial, split, num_events_per_trial, lepton_flavor, dc7, dc9, dc10):

    name = f"{trial}_{split}_{num_events_per_trial}_{lepton_flavor}_{dc7:.2f}_{dc9:.2f}_{dc10:.2f}"
    return name


def get_largest_existing_trial_num(dir_to_search):

    dir_to_search = pathlib.Path(dir_to_search)
    metadata_paths = dir_to_search.rglob("metadata.json")
    trials = [
        pandas.read_json(path, typ="series")["trial"] 
        for path in metadata_paths
    ]
    largest = 0 if trials == [] else max(trials)
    return largest


def setup(
    parent_data_dir:str|pathlib.Path, 
    num_trials:int, 
    num_subtrials:int, 
    num_events_per_trial:int, 
    split:str,
    lepton_flavor:str, 
    interval_dc7:tuple|pandas.Interval,
    interval_dc9:tuple|pandas.Interval,
    interval_dc10:tuple|pandas.Interval,
):
    
    assert are_instance(
        [interval_dc7, interval_dc9, interval_dc10], 
        tuple|pandas.Interval
    )
    if isinstance(interval_dc7, tuple): interval_dc7 = pandas.Interval(*interval_dc7)
    if isinstance(interval_dc9, tuple): interval_dc9 = pandas.Interval(*interval_dc9)
    if isinstance(interval_dc10, tuple): interval_dc10 = pandas.Interval(*interval_dc10)

    parent_data_dir = pathlib.Path(parent_data_dir)
    assert parent_data_dir.is_dir()

    start_trial = get_largest_existing_trial_num(parent_data_dir) + 1
    trial_range = range(start_trial, start_trial+num_trials)

    wc_samples = sample_uniform_wc_dist(
        i_dc7=interval_dc7, 
        i_dc9=interval_dc9, 
        i_dc10=interval_dc10, 
        num_sample=num_trials
    )
    wc_samples.index = trial_range

    dir_names = [
        make_trial_dir_name(
            trial=trial,
            split=split, 
            num_events_per_trial=num_events_per_trial,
            lepton_flavor=lepton_flavor, 
            dc7=row["dc7"], 
            dc9=row["dc9"], 
            dc10=row["dc10"]
        )
        for trial, row in wc_samples.iterrows()
    ]
    metadatas = [
        make_metadata(
            trial=trial, 
            num_events_per_trial=num_events_per_trial, 
            num_subtrials=num_subtrials, 
            split=split,
            lepton_flavor=lepton_flavor, 
            dc7=row["dc7"], 
            dc9=row["dc9"], 
            dc10=row["dc10"], 
            interval_dc7=interval_dc7, 
            interval_dc9=interval_dc9, 
            interval_dc10=interval_dc10
        )
        for trial, row in wc_samples.iterrows()
    ]

    for dir_name, metadata in zip(dir_names, metadatas):
        dir_path = parent_data_dir.joinpath(dir_name)
        dir_path.mkdir()
        metadata_file_path = dir_path.joinpath("metadata.json")
        metadata.to_json(metadata_file_path)


def make_dec_file(file_path, lepton_flavor, dc7, dc9, dc10):

    assert lepton_flavor in ("e", "mu")

    content = f"""
    Alias MyB0 B0
    Alias MyAntiB0 anti-B0
    ChargeConj MyB0 MyAntiB0

    Alias MyK*0 K*0
    Alias MyAnti-K*0 anti-K*0
    ChargeConj MyK*0 MyAnti-K*0

    Decay Upsilon(4S)
    0.500  MyB0 anti-B0    VSS;
    0.500  B0 MyAntiB0    VSS;
    Enddecay

    Decay MyB0
    1.000 MyK*0 {lepton_flavor}+ {lepton_flavor}- BTOSLLNPR 0 0 {dc7} 0 1 {dc9} 0 2 {dc10} 0;
    Enddecay

    CDecay MyAntiB0

    Decay MyK*0
    1.000 K+ pi-   VSS;
    Enddecay

    CDecay MyAnti-K*0

    End
    """

    with open(file_path, "w") as f:
        f.write(content)


def check_trial_completed(trial_dir):

    trial_dir = pathlib.Path(trial_dir)
    
    metadata_path = trial_dir.joinpath("metadata.json")
    metadata = pandas.read_json(metadata_path, typ="series")
    num_subtrials = metadata["num_subtrials"]

    num_recon_files = len(trial_dir.glob("recon*.root"))

    if num_subtrials == num_recon_files:
        return True
    return False


def run_subtrial_job(
    path_to_sim_steer_file, 
    path_to_recon_steer_file, 
    path_to_decay_file, 
    path_to_sim_out_file, 
    path_to_recon_out_file, 
    path_to_log_file, 
    num_events_per_subtrial, 
    lepton_flavor
):
    
    subprocess.run(
        f'bsub -q l "basf2 {path_to_sim_steer_file} -- {path_to_decay_file} {path_to_sim_out_file} {num_events_per_subtrial} &>> {path_to_log_file}'
        f' && basf2 {path_to_recon_steer_file} {lepton_flavor} {path_to_sim_out_file} {path_to_recon_out_file} &>> {path_to_log_file}'
        f' && rm {path_to_sim_out_file}"',
        shell=True,
    )


def run_jobs(parent_data_dir, path_to_sim_steer_file, path_to_recon_steer_file):

    all_trial_dirs = [
        metadata_path.parent for metadata_path 
        in parent_data_dir.rglob("metadata.json")
    ]

    trial_dirs_to_run = [
        dir_ for dir_ in all_trial_dirs
        if not check_trial_completed(dir_)
    ]

    print("Running these trials:")
    for dir_ in trial_dirs_to_run: print(dir_)

    for dir_ in trial_dirs_to_run:

        metadata = pandas.read_json(
            dir_.joinpath("metadata.json"),
            typ="series"
        )

        make_dec_file(
            file_path=dir_.joinpath("decay.dec"),
            **metadata[["lepton_flavor", "dc7", "dc9", "dc10"]]
        )

        for subtrial in range(metadata["num_subtrials"]):
            run_subtrial_job(
                path_to_sim_steer_file=path_to_sim_steer_file,
                path_to_recon_steer_file=path_to_recon_steer_file,
                path_to_decay_file=dir_.joinpath("decay.dec"),
                path_to_sim_out_file=dir_.joinpath(f"sim_{subtrial}.root"),
                path_to_recon_out_file = dir_.joinpath(f"recon_{subtrial}.root"),
                path_to_log_file=dir_.joinpath("log.log"),
                **metadata[["lepton_flavor", "num_events_per_subtrial"]]
            )


def open_simulated_data_root_file(path, unwanted_keys=["persistent;1", "persistent;2"]):
    
    """
    Open a simulated data root file as a pandas dataframe.

    Each tree will be labeled by a pandas multi-index.
    """

    with uproot.open(path) as file:

        keys = [
            key.split(";")[0] for key in file.keys() 
            if key not in unwanted_keys
        ]
        tree_dataframes = [
            file[key].arrays(library="pd") 
            for key in keys
        ]

    dataframe = pandas.concat(tree_dataframes, keys=keys, names=["sim_type",])
    return dataframe


def root_to_parquet(path_to_root_file):
    
    path_to_root_file = pathlib.Path(path_to_root_file)
    if not path_to_root_file.is_file():
        raise FileNotFoundError(f"File not found: {path_to_root_file}")
    dataframe = open_simulated_data_root_file(path_to_root_file).drop(columns="__eventType__")
    save_path = path_to_root_file.with_suffix(".parquet")
    dataframe.to_parquet(save_path)


def combine_files(path_to_parent_data_dir):

    path_to_parent_data_dir = pathlib.Path(path_to_parent_data_dir)
    metadata_file_paths = list(path_to_parent_data_dir.rglob("metadata.json"))
    data_dirs = []

    list_of_dataframes = []
    list_of_keys = []

    for path_to_metadata in (pbar:=tqdm.tqdm(metadata_file_paths, desc="Combining files")):

        data

        pbar.set_postfix_str(path_to_metadata.parent)

        metadata = pandas.read_json(path_to_metadata, typ="series")

        path_to_parquet_file = path_to_metadata.with_name(f"{path_to_metadata.stem}_re.parquet")
        if not path_to_parquet_file.is_file():
            path_to_root_file = path_to_parquet_file.with_suffix(".root")
            root_to_parquet(path_to_root_file)
        data = pandas.read_parquet(path_to_parquet_file)

        data = data.assign(**metadata.drop(labels=["trial", "sub_trial", "num_events"]))

        trial = safer_convert_to_int(metadata["trial"])
        sub_trial = safer_convert_to_int(metadata["sub_trial"])
        split = get_split(trial)
        keys = (trial, sub_trial, split)

        list_of_dataframes.append(data)
        list_of_keys.append(keys)

    data = pandas.concat(
        list_of_dataframes, 
        keys=list_of_keys, 
        names=["trial", "sub_trial", "split"], 
        verify_integrity=True
    )
    data = data.sort_index()
    return data