
from pathlib import Path
from dataclasses import dataclass, field

from pandas import DataFrame, read_parquet

from .preprocess.preprocess import preprocess_dir


@dataclass
class Dataset:
    dir_: Path | str
    parameter_names: list[str]
    observable_names: list[str] 
    remake: bool = False
    data: DataFrame = field(init=False)

    def __post_init__(self):
        self.dir_ = Path(self.dir_)
        data_file_name = "data.parquet"
        data_file_path = self.dir_.joinpath(data_file_name)
        if not data_file_path.is_file() or remake:
            preprocess_dir(
                dir_=dir_,
                out_file_name=data_file_name,
                parameters=parameter_names,
                observables=observable_names,
                lepton_flavor="mu"
            )
        self.data = read_parquet(data_file_path)

    @property
    def parameters(self): -> DataFrame
        out = self.data[self.parameter_names]
        return out

    @property
    def observables(self): -> DataFrame
        out = self.data[self.observable_names]
        return out

    def trial_parameters(self, trial_num): -> DataFrame
        data = self.data.xs(trial_num, level="trial_num")
        out = data[self.parameter_names]
        return out

    def trial_observables(self, trial_num): -> DataFrame
        data = self.data.xs(trial_num, level="trial_num")
        out = data[self.observable_names]
        return out
