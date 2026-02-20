
import torch
from btokstll_sbi_tools.util.types import to_torch_tensor


class Predictor:

    def __init__(self, model, features, device):

        if not isinstance(features, torch.Tensor):
            raise ValueError("Features must be torch tensor.")

        self.device = device
        self.model = model.to(device)
        self.features = features.to(device)

    def calc_log_probs(self):

        with torch.no_grad():
            event_logits = self.model(self.features)
            event_log_probs = torch.nn.functional.log_softmax(event_logits, dim=2)
            set_logits = torch.sum(event_log_probs, dim=1)
            set_log_probs = torch.nn.functional.log_softmax(set_logits, dim=1)
            return set_log_probs
        
    def calc_expected_values(self, set_log_probs, bin_map):

        bin_shift = 5
        bin_map = to_torch_tensor(bin_map)
        bin_map = bin_map.to(self.device)
        set_log_probs = set_log_probs.to(self.device)
        
        def calc_expectation(log_probs):
            log_bin_map = torch.log(bin_map + bin_shift)
            expectation = torch.exp(torch.logsumexp(log_bin_map + log_probs, dim=0)) - bin_shift
            return expectation
    
        with torch.no_grad():
            expected_values = torch.tensor([calc_expectation(log_p) for log_p in set_log_probs])
            return expected_values