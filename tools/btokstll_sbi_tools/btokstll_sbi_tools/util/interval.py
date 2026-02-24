
from dataclasses import dataclass


@dataclass
class Interval:
    left:float
    right:float
    
    def __post_init__(
        self
    ):
        if self.left > self.right:
            raise ValueError(
                "Interval left bound must be greater" 
                " than or equal to right bound."
            )