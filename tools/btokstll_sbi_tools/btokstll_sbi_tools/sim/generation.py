
from pathlib import Path
import subprocess
import json
import dataclasses
import functools
import time
from typing import NoReturn

from pandas import DataFrame, read_parquet, concat
import numpy
import uproot
from tqdm import tqdm
from ..util import (
    safer_convert_to_int, 
    Interval,
    append_to_stem,
    flatten_dict,
    load_json,
    read_parquets
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
        lower_bounds = [b["left"] for b in bounds]
        upper_bounds = [b["right"] for b in bounds]
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
            json.dump(metadata_dict, file, indent=4)

    @classmethod
    def from_json_file(
        cls, 
        path
    ):
        # Requires manual object reconstruction.

        with open(path, 'r') as f:
            metadata_dict = json.load(f)
        
        def reconstruct(dict, key, cls_):
            dict[key] = cls_(**dict[key])
        
        reconstruct(
            metadata_dict, 
            "delta_wilson_coefficient_set", 
            Delta_Wilson_Coefficient_Set
        )
        for key in metadata_dict["delta_wilson_coefficient_distribution"].keys():
            reconstruct(
                metadata_dict["delta_wilson_coefficient_distribution"],
                key, 
                Interval
            )
        reconstruct(
            metadata_dict,
            "delta_wilson_coefficient_distribution", 
            Uniform_Delta_Wilson_Coefficient_Distribution
        )
        return cls(**metadata_dict)


@dataclasses.dataclass(frozen=True)
class Directory_Manager:
    main_data_dir: Path
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
        largest = -1 if trial_nums == [] else max(trial_nums)
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
            metadata.to_json_file(dir_.joinpath(self.metadata_file_name))


def write_dec_file(
    file_path: Path, 
    lepton_flavor: str, 
    delta_wilson_coefficient_set: Delta_Wilson_Coefficient_Set, 
):

    if lepton_flavor not in ("e", "mu"):
        raise ValueError(
            f"Lepton flavor ({lepton_flavor}) must be 'e' or 'mu'."
        )
    
    delta_c_7 = delta_wilson_coefficient_set.delta_c_7
    delta_c_9 = delta_wilson_coefficient_set.delta_c_9
    delta_c_10 = delta_wilson_coefficient_set.delta_c_10

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


def submit_job(
    lepton_flavor,
    num_events,
    sim_steer_file_path,
    recon_steer_file_path,
    decay_file_path,
    sim_file_path,
    recon_file_path,
    log_file_path,
):
    subprocess.run(
        f'bsub -q l "basf2 {sim_steer_file_path} -- {decay_file_path} {sim_file_path} {num_events} &>> {log_file_path}'
        f' && basf2 {recon_steer_file_path} {lepton_flavor} {sim_file_path} {recon_file_path} &>> {log_file_path}'
        f' && rm {sim_file_path}"',
        shell=True,
    )


@dataclasses.dataclass
class Job_Submitter:
    main_data_dir:Path
    recon_steer_file_path:Path
    sim_steer_file_path:Path
    metadata_file_name:Path = Path("metadata.json")
    recon_file_name:Path = Path("recon.root")
    sim_file_name:Path = Path("sim.root")
    decay_file_name:Path = Path("decay.dec")
    log_file_name:Path = Path("log.log")
    batch_size:int = 500
    batch_wait:int = 300

    def __post_init__(
        self,
    ):
        if not self.main_data_dir.is_dir():
            raise ValueError(
                "Main data directory not found"
                f" ({self.main_data_dir})."
            )
        
        self.num_submitted_jobs = 0
    
    @property
    def incomplete_dirs(
        self
    ):
        def is_incomplete(dir_:Path):
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

    def submit_jobs(
        self,
    ):
        for dir_ in self.incomplete_dirs:

            decay_file_path = dir_.joinpath(self.decay_file_name)
            log_file_path = dir_.joinpath(self.log_file_name)
            metadata_file_path = dir_.joinpath(self.metadata_file_name)

            metadata = Trial_Metadata.from_json_file(metadata_file_path)

            write_dec_file(
                decay_file_path, 
                metadata.lepton_flavor, 
                metadata.delta_wilson_coefficient_set
            )

            for subtrial in range(metadata.num_subtrials):

                sim_file_path = dir_.joinpath(
                    append_to_stem(self.sim_file_name, subtrial)
                )
                recon_file_path = dir_.joinpath(
                    append_to_stem(self.recon_file_name, subtrial)
                )

                submit_job(
                    lepton_flavor=metadata.lepton_flavor,
                    num_events=metadata.num_events_per_subtrial,
                    sim_steer_file_path=self.sim_steer_file_path,
                    recon_steer_file_path=self.recon_steer_file_path,
                    decay_file_path=decay_file_path,
                    sim_file_path=sim_file_path,
                    recon_file_path=recon_file_path,
                    log_file_path=log_file_path,
                )

                self.num_submitted_jobs += 1

                if self.num_submitted_jobs % self.batch_size == 0:
                    time.sleep(self.batch_wait)


def open_simulated_data_root_file(
    path:Path|str, 
    unwanted_keys:list[str]=[
        "persistent;1", 
        "persistent;2"
    ],
) -> DataFrame:
    
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

    dataframe = concat(
        tree_dataframes, 
        keys=keys,
        names=["sim_type",]
    )
    return dataframe


def root_to_parquet(
    root_file_path:Path|str,
) -> None:
    
    root_file_path = Path(root_file_path)
    if not root_file_path.is_file():
        raise FileNotFoundError(
            f"File not found: {root_file_path}"
        )
    dataframe = open_simulated_data_root_file(
        root_file_path
    ).drop(
        columns="__eventType__"
    )
    save_path = root_file_path.with_suffix(
        ".parquet"
    )
    dataframe.to_parquet(save_path)


def root_files_to_parquet(
    paths:list[Path],
    lazy:bool=True,
) -> None:
    
    to_convert = (
        paths if not lazy 
        else [
            path for path in paths
            if not path.with_suffix(".parquet").is_file()
        ]
    )
    for path in to_convert:
        root_to_parquet(path)


def combine_files(
    dirs:list[Path],
    out_file_path:Path|str,
    index_names:list[str]=[
        "trial_num", 
        "lepton_flavor", 
        "split",
    ]
) -> None:
    
    for dir_ in dirs:
        if not dir_.is_dir():
            raise ValueError(
                "Input paths must be directories."
                f" {dir_} is not a directory."
            )
        nested_data_file_paths = (
            list(dir_.glob("*.root")) + 
            list(dir_.glob("*.parquet"))
        )
        if not nested_data_file_paths:
            raise ValueError(
                f"No data file in directory: {dir_}"
            )
        if not dir_.joinpath("metadata.json").is_file():
            raise ValueError(
                f"No metadata file in directory: {dir_}"
            )
    
    for dir_ in (
        pbar := tqdm(
            dirs, 
            desc="Converting"
        )
    ):
        pbar.set_postfix_str(dir_.name)
        root_file_paths = list(dir_.glob("*.root"))
        root_files_to_parquet(
            paths=root_file_paths,
            lazy=True,
        )

    metadata_file_paths = [
        dir_.joinpath("metadata.json") 
        for dir_ in dirs
    ]
    metadatas = [
        load_json(path) 
        for path in metadata_file_paths
    ]
    metadatas = [
        flatten_dict(metadata)
        for metadata in metadatas
    ]

    nested_data_file_paths = [
        list(dir_.glob("*.parquet"))
        for dir_ in dirs
    ]
    dataframes = [
        read_parquets(paths)
        for paths in nested_data_file_paths
    ]
    
    index = [
        {
            name: metadata.pop(name) 
            for name in index_names
        } 
        for metadata in metadatas
    ]
    dataframes = [
        df.assign(**metadata) 
        for df, metadata in zip(
            dataframes, 
            metadatas
        )
    ]

    keys = [
        tuple(i.values())
        for i in index
    ]
    data = concat(
        dataframes, 
        keys=keys,
        names=index_names, 
    )
    data = data.sort_index() # check this for memory usage
    
    data.to_parquet(out_file_path)