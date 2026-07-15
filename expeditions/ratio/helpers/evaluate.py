from torch import full, column_stack, sum, Tensor, cat
from torch.nn import Module

from .util import get_model_device


def append_parameter(parameter: float, features: Tensor):

    parameter_tensor = full((len(features),), parameter, dtype=features.dtype)
    parameter_tensor = parameter_tensor.to(features.device)
    out = column_stack([features, parameter_tensor])
    return out


def evaluate(model: Module, features: Tensor, parameter_samples: Tensor) -> Tensor:

    device = get_model_device(model)

    features = features.to(device)

    predictions = []

    for parameter in parameter_samples:
        parameter = parameter.item()
        model_input = append_parameter(parameter, features)
        event_predictions = model.log_likelihood_ratio(model_input)
        dataset_prediction = sum(event_predictions, dim=0)
        predictions.append(dataset_prediction)

    predictions = cat(predictions)

    return predictions
