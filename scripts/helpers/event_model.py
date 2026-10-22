
from torch import (
    Tensor, 
    tensor, 
    max, 
    sum, 
    ones, 
    zeros, 
    zeros_like,
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
    def __init__(self, parameter_size:int, feature_size:int):
        super().__init__()
        self.parameter_size = parameter_size
        self.feature_size = feature_size
        self.input_size = parameter_size + feature_size
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

    def forward(self, parameters:Tensor, features:Tensor): 
        input = self.create_input(parameters=parameters, features=features)
        rescaled_input = self.std_scale(input) 
        out = self.layers(rescaled_input)
        return out
    
    def create_input(self, parameters:Tensor, features:Tensor):
        assert parameters.shape[-1] == self.parameter_size
        assert features.shape[-1] == self.feature_size
        out = cat([parameters, features], dim=-1)
        assert out.shape[-1] == self.input_size
        return out

    def std_scale(self, input: Tensor):
        out = (input - self.train_means) / self.train_stds
        return out
    
    def set_std_scale(
        self, 
        train_parameters: Tensor,
        train_features: Tensor,
    ):
        train_input = self.create_input(parameters=train_parameters, features=train_features)
        train_means = train_input.mean(dim=0)
        train_stds = train_input.std(dim=0)
        self.train_means = Buffer(train_means)
        self.train_stds = Buffer(train_stds)

    def scan_over(self, parameters, features):
        predictions = zeros(parameters.shape[0])
        for i in range(len(parameters)):
            parameters_ = parameters[i].unsqueeze(dim=0).expand(features.shape[0], -1) 
            predictions[i] = sum(self(parameters_, features)).detach()
        return predictions

    def mle(self, parameters, features):
        ratios = self.scan_likelihood_ratio(parameters, features)
        mle_x = parameters[ratios.argmax()]
        mle_y = max(ratios)
        return mle_x, mle_y



