
import torch


def select_device():

    """
    Select a device to compute with.

    Returns the name of the selected device.
    "cuda" if cuda is available, otherwise "cpu".
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)
    return device


def get_model_current_device(model):

    device = next(model.parameters()).device
    return device
