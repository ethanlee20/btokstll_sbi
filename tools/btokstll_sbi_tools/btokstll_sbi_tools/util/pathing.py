
from pathlib import Path


def append_to_stem(
    path:Path, 
    s,
):
    return path.with_stem(f"{path.stem}{s}")
