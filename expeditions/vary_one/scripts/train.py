
import torch


class MLP(torch.nn.Module):

    def __init__(self):

        super().__init__()
        
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(4, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 30),
        )

    def forward(self, x):
        
        logits = self.layers(x)
        return logits