
from torch import cuda, device
from torch.nn import Module


def select_device(
    verbose=True
) -> str:
    """
    Select a device to compute with.

    Returns the name of the selected device.
    "cuda" if cuda is available, otherwise "cpu".
    """
    device = "cuda" if cuda.is_available() else "cpu"
    if verbose: 
        print("Device: ", device)
    return device


def get_model_current_device(
    model:Module
) -> device:
    device = next(model.parameters()).device
    return device