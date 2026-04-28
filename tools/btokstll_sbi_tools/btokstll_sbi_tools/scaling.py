
from dataclasses import dataclass

from torch import Tensor


def std_scale(
    data:Tensor, 
    means:Tensor, 
    stdevs:Tensor,
):
    return (data - means) / stdevs


def undo_std_scale(
    data:Tensor,
    means:Tensor,
    stdevs:Tensor,
):
    return data * stdevs + means


@dataclass
class Std_Scaler:
    means: Tensor
    stdevs: Tensor

    def std_scale(
        self, 
        data:Tensor,
    ):
        return std_scale(
            data, 
            means=self.means, 
            stdevs=self.stdevs,
        )
    
    def undo_std_scale(
        self, 
        data:Tensor,
    ):
        return undo_std_scale(
            data,
            means=self.means,
            stdevs=self.stdevs,
        )