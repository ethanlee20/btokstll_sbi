
from torch.nn import ReLU
from torch.nn.optim import AdamW

from helpers.data.dataset import Dataset
from helpers.model.model import NreCModel


model = NreCModel(
    dir_="../models/test",
    device="cuda",
    parameter_size=2,
    observable_size=4,
    num_hidden_layers=3,
    hidden_layer_size=100,
    activation=ReLU,        
)

parameter_names = ["dc7", "dc9"]
observable_names=["q_sq", "cos_theta_mu", "cos_theta_k", "chi"]

train_dataset = Dataset(
    dir_=...,
    parameter_names=parameter_names,
    observable_names=observable_names    
)
val_dataset = Dataset(
    dir_=...,
    parameter_names=parameter_names,
    observable_names=observable_names,
)
val_set_dataset = Dataset(
    dir_=...,
    parameter_names=parameter_names,
    observable_names=observable_names
)

model.train(
    train_dataset=train_dataset,
    optimizer=AdamW(
        model.architecture.parameters(), 
        lr=3e-4, 
        betas=(0.95, 0.95)
    ),
    num_epochs=5,
    batch_size=10_000,
    k=10,
    gamma=1,
    val_dataset=val_dataset,
)

model.eval(
    dataset=val_set_dataset,
    scan_trial_nums=range(3),
    linearity_trial_nums=range(3),
    bounds=[(-2, 1), (-0.1, 0.1)],
    scan_size=10,
    rerun=True
)


