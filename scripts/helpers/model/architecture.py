
from torch import (
    Tensor, 
    ones, 
    zeros, 
    cat
)
from torch.nn import (
    Module, 
    ReLU, 
    Linear, 
    Sequential
)
from torch.nn.parameter import Buffer


class MLP(Module):
    def __init__(
        self, 
        parameter_size:int, 
        observable_size:int
        num_hidden_layers:int=3, 
        hidden_layer_size:int=100,
        activation=ReLU
    ):
        super().__init__()

        self.parameter_size = parameter_size
        self.observable_size = observable_size
        self.input_size = parameter_size + observable_size

        self.layers = Sequential()
        layer_sizes = (
            [input_size] 
            + [hidden_layer_size] * num_hidden_layers 
            + [1]
        )
        weight_shapes = zip(layer_sizes[:-1], layer_sizes[1:]) 
        for shape in weight_shapes:
            self.layers.append(Linear(*shape))
        for activation_index in range(len(weight_shapes) - 1, 0, -1):
            self.layers.insert(activation_index, activation())
        
        self.train_means = Buffer(zeros(self.input_size))
        self.train_stds = Buffer(ones(self.input_size))

    def forward(self, parameters:Tensor, observables:Tensor): 
        input = self.create_input(
            parameters=parameters, 
            observables=observables
        )
        rescaled_input = self.std_scale(input) 
        out = self.layers(rescaled_input)
        return out
    
    def create_input(self, parameters:Tensor, observables:Tensor):
        assert parameters.shape[-1] == self.parameter_size
        assert observables.shape[-1] == self.observable_size
        out = cat([parameters, observables], dim=-1)
        assert out.shape[-1] == self.input_size
        return out

    def std_scale(self, input: Tensor):
        out = (input - self.train_means) / self.train_stds
        return out
    
    def set_std_scale(
        self, 
        train_parameters: Tensor,
        train_observables: Tensor,
    ):
        train_input = self.create_input(
            parameters=train_parameters, 
            observables=train_observables
        )
        train_means = train_input.mean(dim=0)
        train_stds = train_input.std(dim=0)
        self.train_means = Buffer(train_means)
        self.train_stds = Buffer(train_stds)


class NreCLoss(Module):
    def __init__(self, gamma, k):
        super().__init__()
        self.gamma = tensor(gamma)
        self.k = tensor(k)

    def forward(self, logits, is_joint):
        device = logits.device
        self.gamma = self.gamma.to(device)
        self.k = self.k.to(device)
        
        extended_logits = cat(
            [
                log(self.k).view(1,1,1).expand(logits.shape[0], -1, -1), 
                log(self.gamma) + logits
            ],
            dim=1
        )
        denominator = logsumexp(extended_logits, dim=1)

        numerator_marginal = log(self.k)
        numerator_joint = log(self.gamma) + logits[:, k-1, :]  

        marginal = numerator_marginal - denominator
        joint = numerator_joint - denominator

        marginal_weight = self.gamma / (1 + self.gamma)
        joint_weight = 1 / (1 + self.gamma)

        loss =  -((1 - is_joint) * marginal_weight * marginal + is_joint * joint_weight * joint) 
        expected_loss = 2 * mean(loss)
        return expected_loss
