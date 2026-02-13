
import pathlib
import subprocess
import json
import dataclasses
import functools

import pandas
import numpy
import uproot
import tqdm

from ..utilities.types import (
    safer_convert_to_int, 
    are_instance,
    to_pandas_interval,
    Interval,
    append_to_stem
)


@dataclasses.dataclass(frozen=True)
class Delta_Wilson_Coefficient_Set:
    delta_c_7: float
    delta_c_9: float
    delta_c_10: float


@dataclasses.dataclass(frozen=True)
class Uniform_Delta_Wilson_Coefficient_Distribution:
    delta_c_7_bounds: Interval 
    delta_c_9_bounds: Interval 
    delta_c_10_bounds: Interval 
    

class Sampler:
    def __init__(
        self, 
        distribution:Uniform_Delta_Wilson_Coefficient_Distribution,
        rng_seed=None
    ):
        self.distribution = distribution
        self.rng = numpy.random.default_rng(rng_seed)

    def sample(
        self, 
        num_samples:int
    ):
        bounds = dataclasses.asdict(self.distribution).values()
        lower_bounds = [b.left for b in bounds]
        upper_bounds = [b.right for b in bounds]
        samples = self.rng.uniform(
            low=lower_bounds,
            high=upper_bounds,
            size=(num_samples, len(bounds))
        )
        samples = [
            Delta_Wilson_Coefficient_Set(*s) 
            for s in samples
        ]
        return samples


@dataclasses.dataclass(frozen=True)
class Trial_Metadata:
    trial_num: int
    num_events_per_trial: int
    num_subtrials: int
    split: str
    lepton_flavor: str
    delta_wilson_coefficient_set: Delta_Wilson_Coefficient_Set
    delta_wilson_coefficient_distribution: Uniform_Delta_Wilson_Coefficient_Distribution

    @functools.cached_property
    def num_events_per_subtrial(
        self
    ):
        return safer_convert_to_int(
            self.num_events_per_trial
            / self.num_subtrials
        )
    
    def to_json_file(
        self, 
        path
    ):
        metadata_dict = dataclasses.asdict(self)
        with open(path, 'x') as file:
            json.dump(metadata_dict, file)

    @classmethod
    def from_json_file(
        cls, 
        path
    ):
        # Requires manual object reconstruction.

        with open(path, 'r') as f:
            metadata_dict = json.load(f)
        
        def reconstruct(key, cls_):
            metadata_dict[key] = cls_(**metadata_dict[key])
        
        reconstruct("delta_wilson_coefficient_set", Delta_Wilson_Coefficient_Set)
        for key in metadata_dict["delta_wilson_coefficient_distribution"].keys():
            reconstruct(key, Interval)
        reconstruct("delta_wilson_coefficient_distribution", Uniform_Delta_Wilson_Coefficient_Distribution)
    
        return cls(**metadata_dict)


@dataclasses.dataclass(frozen=True)
class Directory_Manager:
    main_data_dir: pathlib.Path
    num_trials: int
    num_subtrials: int
    num_events_per_trial: int
    split: str
    lepton_flavor: str
    delta_wilson_coefficient_set_samples: list[Delta_Wilson_Coefficient_Set]
    delta_wilson_coefficient_distribution: Uniform_Delta_Wilson_Coefficient_Distribution
    metadata_file_name: str = "metadata.json"

    def __post_init__(
        self
    ):
        if not self.main_data_dir.is_dir():
            raise ValueError(
                "Main data directory not found"
                f" ({self.main_data_dir})."
            )
        if self.num_trials != len(self.delta_wilson_coefficient_set_samples):
            raise ValueError(
                "Number of samples" 
                f" ({len(self.delta_wilson_coefficient_set_samples)})" 
                " must match number of trials"
                f" ({self.num_trials})."
            )
        
    @functools.cached_property
    def trial_range(
        self
    ):
        start = self._largest_existing_trial_num + 1 
        end = start + self.num_trials
        return range(start, end)
    
    @functools.cached_property
    def metadata_list(
        self
    ):
        return [
            Trial_Metadata(
                trial_num=trial, 
                num_events_per_trial=self.num_events_per_trial, 
                num_subtrials=self.num_subtrials,
                split=self.split,
                lepton_flavor=self.lepton_flavor,
                delta_wilson_coefficient_set=sample,
                delta_wilson_coefficient_distribution=self.delta_wilson_coefficient_distribution,
            ) for trial, sample in zip(
                self.trial_range, 
                self.delta_wilson_coefficient_set_samples
            )
        ]

    @functools.cached_property
    def _largest_existing_trial_num(
        self
    ):
        paths_files_metadata = self.main_data_dir.rglob(
            self.metadata_file_name
        )
        metadatas = [
            Trial_Metadata.from_json_file(p) 
            for p in paths_files_metadata
        ]
        trial_nums = [m.trial_num for m in metadatas]
        largest = 0 if trial_nums == [] else max(trial_nums)
        return largest
    
    def setup_dirs(
        self
    ):
        def dir_name(m:Trial_Metadata):
            name = (
                f"{m.trial_num}"
                f"_{m.split}"
                f"_{m.num_events_per_trial}"
                f"_{m.lepton_flavor}"
            )
            for c in dataclasses.astuple(
                m.delta_wilson_coefficient_set
            ):
                name += f"_{c:.2f}"
            return name

        dir_names = [dir_name(m) for m in self.metadata_list]
        dir_paths = [self.main_data_dir.joinpath(n) for n in dir_names]
        for dir_, metadata in zip(dir_paths, self.metadata_list):
            dir_.mkdir()
            metadata.to_json_file(dir_)


class Job_Submitter:
    def __init__(
        self,
        main_data_dir:pathlib.Path,
        max_num_jobs:int, 
        metadata_file_name:pathlib.Path=pathlib.Path("metadata.json"),
        recon_file_name:pathlib.Path=pathlib.Path("recon.root")
    ):
        if not main_data_dir.is_dir():
            raise ValueError(
                "Main data directory not found"
                f" ({main_data_dir})."
            )
        self.main_data_dir = main_data_dir
        self.max_num_jobs = max_num_jobs
        self.metadata_file_name = metadata_file_name
        self.recon_file_name = recon_file_name
    
    @property
    def incomplete_dirs(
        self
    ):
        def is_incomplete(dir_:pathlib.Path):
            num_subtrials = Trial_Metadata.from_json_file(
                dir_.joinpath(self.metadata_file_name)
            ).num_subtrials
            num_recon_files = len(list(
                dir_.glob(str(
                    append_to_stem(self.recon_file_name, '*')
                ))
            ))
            if num_subtrials != num_recon_files:
                return True
            return False
        
        return [
            p for p in self._all_candidate_dirs 
            if is_incomplete(p)
        ] 
    
    @functools.cached_property
    def _all_candidate_dirs(
        self
    ):
        metadata_file_paths = self.main_data_dir.rglob(str(
            self.metadata_file_name
        ))
        return [p.parent for p in metadata_file_paths]



def make_dec_file(
    file_path, 
    lepton_flavor, 
    delta_c_7, 
    delta_c_9, 
    delta_c_10
):

    if lepton_flavor not in ("e", "mu"):
        raise ValueError(
            f"Lepton flavor ({lepton_flavor}) must be 'e' or 'mu'."
        )

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
    1.000 MyK*0 {lepton_flavor}+ {lepton_flavor}- BTOSLLNPR 0 0 {delta_c_7} 0 1 {delta_c_9} 0 2 {delta_c_10} 0;
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