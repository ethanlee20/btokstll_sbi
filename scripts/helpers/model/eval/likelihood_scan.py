
from numpy.typing import ArrayLike
from matplotlib.pyplot import Axes

from torch import Tensor, zeros, sum


def scan_over_single(model: MLP, parameters: Tensor, observables: Tensor):
    predictions = zeros(parameters.shape[0])
    for i in range(len(parameters)):
        parameters_ = parameters[i].unsqueeze(dim=0).expand(observables.shape[0], -1) 
        predictions[i] = sum(model(parameters_, observables)).detach()
    return predictions


def scan_over(model: MLP, parameters: Tensor, observables: list[Tensor]|Tensor):
    assert (
        isinstance(observables, list) 
        or isinstance(observables, Tensor)
    )
    if isinstance(observables, Tensor):
        observables = [observables]
    predictions = [
        scan_over_single(
            model=model, 
            parameters=parameters, 
            observables=o
        ) for o in observables
    ]
    predictions = [p.unsqueeze(dim=0) for p in predictions]
    predictions = cat(predictions, dim=0)
    return predictions


def _plot_likelihood_scan(
    parameters: ArrayLike,
    log_probabilities: ArrayLike,
    color: str,
    mle_x=None, 
    mle_y=None,
    confidence_interval=None,
    true_value:float|None=None,
):
    
    ax.plot(parameters, log_probabilities, color=color)
    if true_value is not None:
        ax.axvline(true_value, color=color, linestyle="--", zorder=-100)
    if mle_x is not None and mle_y is not None:
        ax.scatter(mle_x, mle_y, color=color, s=25, zorder=100)
    if confidence_interval is not None:
        ax.hlines(mle_y, confidence_interval[0], confidence_interval[-1], color=color, linestyle="--", zorder=-101)
    ax.set_ylabel(r"$\log p(\textrm{data} \;|\; \delta C_9) + c$", fontsize=15)
    ax.set_xlabel(r"$\delta C_9$", fontsize=15)    

