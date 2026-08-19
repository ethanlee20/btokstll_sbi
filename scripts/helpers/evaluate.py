from torch import log, full, column_stack, sum, Tensor, cat, stack
from torch.nn import Module

from .util import get_model_device


def append_parameter(parameter: float | Tensor, features: Tensor) -> Tensor:
    if isinstance(parameter, Tensor):
        parameter = parameter.item()
    parameter_tensor = full((len(features),), parameter, dtype=features.dtype)
    parameter_tensor = parameter_tensor.to(features.device)
    out = column_stack([features, parameter_tensor])
    return out


def eval_prob(model, features, parameter, cal_fit=None):
    device = get_model_device(model)
    features = features.to(device)
    model.eval()

    model_input = append_parameter(parameter, features)
    out = (
        model.probability(model_input) if cal_fit is None
        else model.cal_prob(model_input, cal_fit)
    )
    return out


def eval_probs(model: Module, features: Tensor, parameters: Tensor, cal_fits:None|dict=None) -> Tensor:
    probs = [eval_prob(model, features, p, cal_fits[p]) for p in parameters]    
    out = column_stack(probs)
    return out


def log_likelihood_ratio(probabilities: Tensor):
    out = log(1 / probabilities - 1)
    return out


def eval_log_likelihood_ratio(model, features, parameters, cal_fits:None|dict=None):
    probs = eval_probs(model, features, parameters, cal_fits)
    out = log_likelihood_ratio(probs)
    return out


def eval_sum_log_likelihood_ratio(model, features, parameters, cal_fits:None|dict=None):
    ratios = eval_log_likelihood_ratio(model, features, parameters, cal_fits)
    out = ratios.sum(dim=0)
    return out


"""
def eval_probs_ensemble(models, features, parameters):
    probs = [eval_probs(model, features, parameters) for model in models]
    probs = stack(probs)
    out = probs.mean(dim=0)
    return out
"""

"""
def eval_log_likelihood_ratio_sum_ensemble(models, features, parameters):
    log_likelihood_ratio = eval_log_likelihood_ratio_ensemble(
        models, features, parameters
    )
    out = log_likelihood_ratio.sum(dim=0)
    return out
"""

"""
def eval_log_likelihood_ratio_ensemble(models, features, parameters):
    avg_probs = eval_probs_ensemble(models, features, parameters)
    out = log_likelihood_ratio(avg_probs)
    return out
"""

"""
def evaluate(model: Module, features: Tensor, parameter_samples: Tensor) -> Tensor:
    device = get_model_device(model)
    features = features.to(device)
    model.eval()

    predictions = []
    for parameter in parameter_samples:
        parameter = parameter.item()
        model_input = append_parameter(parameter, features)
        event_predictions = model.log_likelihood_ratio(model_input)
        dataset_prediction = sum(event_predictions, dim=0)
        predictions.append(dataset_prediction)

    predictions = cat(predictions)
    return predictions
"""