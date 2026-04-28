
from pathlib import Path

from torch import save, load
from torch.nn import Module


def save_torch_model_state_dict(
    model:Module, 
    path:Path|str,
):
    path = Path(path)
    if not path.parent.is_dir():
        raise ValueError(f"Parent directory doesn't exist: {path.parent}")
    if path.exists():
        raise ValueError(f"File exists: {path}")
    save(model.state_dict(), path)


def load_torch_model_state_dict( 
    path: Path|str
):
    state_dict = load(path, weights_only=True)
    return state_dict
