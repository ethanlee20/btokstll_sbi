
from pathlib import Path

from torch import load, save
from torch.nn import Module


def save_torch_model_state_dict(
    model:Module, 
    path:Path|str,
):
    save(model.state_dict(), path)


def load_torch_model_state_dict( 
    path: Path|str
):
    state_dict = load(path, weights_only=True)
    return state_dict