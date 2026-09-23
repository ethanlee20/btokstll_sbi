
from torch import Tensor, tensor, max, sum, ones, zeros, mean, linspace, nonzero
from torch.nn import (
    Module, 
    ReLU, 
    Linear, 
    Dropout, 
    Softmax, 
    Sequential
)
from torch.nn.parameter import Buffer


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.layers = Sequential(
            Linear(5, 500),
            ReLU(),
            Linear(500, 500),
            ReLU(),
            Linear(500, 500),
            ReLU(),
            Linear(500, 1),
        )
        self.train_means = Buffer(zeros(5))
        self.train_stds = Buffer(ones(5))

    def forward(self, x):
        rescaled_x = self.std_scale(x) 
        raw = self.layers(rescaled_x)
        out = sum(raw, dim=(1,2))
        return out

    def std_scale(self, x: Tensor):
        out = (x - self.train_means) / self.train_stds
        return out
    
    def set_std_scale(
        self, 
        train_means: Tensor, 
        train_stds: Tensor
    ):
        self.train_means = Buffer(train_means)
        self.train_stds = Buffer(train_stds)

    def scan_likelihood_ratio(self, scan_min, scan_max, scan_num, x):
        parameters = linspace(scan_min, scan_max, scan_num)
        ratios = []
        for p in parameters:
            x[:, :, 4] = p
            log_ratio_sum = self(x)
            ratios.append(log_ratio_sum.detach())
        ratios = tensor(ratios)
        return parameters, ratios

    def mle_and_confidence_interval(self, scan_min, scan_max, scan_num, x):
        parameters, ratios = self.scan_likelihood_ratio(scan_min, scan_max, scan_num, x)
        mle_x = parameters[ratios.argmax()]
        mle_y = max(ratios)
        parameters_within_confidence = parameters[nonzero(ratios > mle_y - 0.5, as_tuple=True)]        
        confidence_interval = (
            tensor(float['nan'], float['nan']) if parameters_within_confidence.size in (0, 1)
            else tensor(
                [parameters_within_confidence[0].item(), parameters_within_confidence[-1].item()]
            )
        )
        return mle_x, mle_y, confidence_interval, parameters, ratios



