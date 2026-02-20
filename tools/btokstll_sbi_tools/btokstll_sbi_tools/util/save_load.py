
import torch


def save_torch_model(model, path):

    torch.save(model.state_dict(), path)


def open_torch_model_state_dict(path):
    
    state_dict = torch.load(path, weights_only=True)
    return state_dict