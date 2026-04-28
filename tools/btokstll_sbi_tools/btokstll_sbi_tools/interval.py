
from typing import Iterator
from dataclasses import dataclass


@dataclass
class Interval:
    left: float|int = 0
    right: float|int = 0

    def __post_init__(self):
        if self.left > self.right:
            raise ValueError(
                "Left bound must be"
                " less than or equal to"
                " right bound."
            )

    def __iter__(
        self,
    ) -> Iterator[float|int]:
        return (
            self.left, 
            self.right
        ).__iter__()