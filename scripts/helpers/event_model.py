
from torch import (
    Tensor, 
    tensor, 
    max, 
    sum, 
    ones, 
    zeros, 
    mean, 
    linspace, 
    nonzero, 
    exp, 
    cat
)
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
    def __init__(self, param_size:int, feature_size:int):
        super().__init__()
        self.param_size = param_size
        self.feature_size = feature_size
        self.input_size = param_size + feature_size
        self.layers = Sequential(
            Linear(self.input_size, 100),
            ReLU(),
            Linear(100, 100),
            ReLU(),
            Linear(100, 100),
            ReLU(),
            Linear(100, 1),
        )
        self.train_means = Buffer(zeros(self.input_size))
        self.train_stds = Buffer(ones(self.input_size))

    def forward(self, params:Tensor, features:Tensor): 
        input = self.create_input(params=params, features=features)
        rescaled_input = self.std_scale(input) 
        out = self.layers(rescaled_input)
        return out
    
    def create_input(self, params:Tensor, features:Tensor):
        assert params.shape[-1] == self.param_size
        assert features.shape[-1] == self.feature_size
        out = cat([params, features], dim=-1)
        assert out.shape[-1] == self.input_size
        return out

    def std_scale(self, input: Tensor):
        out = (input - self.train_means) / self.train_stds
        return out
    
    def set_std_scale(
        self, 
        train_params: Tensor,
        train_features: Tensor,
    ):
        train_input = self.create_input(params=train_params, features=train_features)
        train_means = train_input.mean(dim=0)
        train_stds = train_input.std(dim=0)
        self.train_means = Buffer(train_means)
        self.train_stds = Buffer(train_stds)

    def scan_likelihood_ratio(self, min, max, num, features):
        parameters = linspace(min, max, num)
        predictions = zeros_like(parameters)
        for i in range(len(parameters)):
            predictions[i] = sum(self(parameters[i], features)).detach()
        return parameters, predictions

    def mle(self, scan_min, scan_max, scan_num, x):
        parameters, ratios = self.scan_likelihood_ratio(scan_min, scan_max, scan_num, x)
        mle_x = parameters[ratios.argmax()]
        mle_y = max(ratios)
        return mle_x, mle_y



