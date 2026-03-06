
from torch import no_grad, sum, log, exp, logsumexp, Tensor
from torch.nn import Module
from torch.nn.functional import log_softmax

from ..util import to_torch_tensor, Dataset


class Predictor:

    def __init__(
        self, 
        model:Module, 
        dataset:Dataset, 
        device:str,
    ):
        self.device = device
        self.model = model.to(device)
        self.dataset = dataset
        self.dataset.features = self.dataset.features.to(device)

    def calc_log_probs(
        self,
    ) -> Tensor:
        with no_grad():
            event_logits = self.model(self.dataset.features)
            event_log_probs = log_softmax(event_logits, dim=2)
            set_logits = sum(event_log_probs, dim=1)
            set_log_probs = log_softmax(set_logits, dim=1)
            return set_log_probs
        
    def calc_expected_values(
        self, 
        set_log_probs:Tensor, 
        bin_mids:Tensor,
        bin_shift:int=5,
    ) -> Tensor:
        bin_mids = bin_mids.to(self.device)
        set_log_probs = set_log_probs.to(self.device)
        
        def calc_expectation(log_probs):
            log_bin_map = log(bin_mids + bin_shift)
            expectation = exp(logsumexp(log_bin_map + log_probs, dim=0)) - bin_shift
            return expectation
    
        with no_grad():
            expected_values = Tensor(
                [calc_expectation(log_p) for log_p in set_log_probs]
            )
            return expected_values