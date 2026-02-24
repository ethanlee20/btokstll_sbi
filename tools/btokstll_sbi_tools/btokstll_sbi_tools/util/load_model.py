
from torch import load


def open_torch_model_state_dict( 
    path
):
    state_dict = load(path, weights_only=True)
    return state_dict