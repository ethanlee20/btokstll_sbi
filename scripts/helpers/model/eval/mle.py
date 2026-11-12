
from torch import Tensor, max

from ..architecture import MLP
from .likelihood_scan import scan_over


def mle_by_scan(likelihoods):
    mle_likelihood, argmax_ = max(likelihoods, dim=0)
    mle_ = parameters[argmax_]
    return mle_, mle_likelihood
