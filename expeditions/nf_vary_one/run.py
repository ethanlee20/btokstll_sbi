
import torch
import numpy as np
import normflows as nf
from matplotlib import pyplot as plt
from tqdm import tqdm

from btokstll_sbi_tools.model import (
    Dataset_Set_File_Paths,
    Dataset_Set,
    Data_Loader,
)
from btokstll_sbi_tools.util.misc import save_plot_and_close


# load data


data_paths = Dataset_Set_File_Paths(
    "data/vary_c9_train.parquet", 
    "data/vary_c9_val.parquet",
)
dset_set = Dataset_Set.from_pandas_parquet_files(
    data_paths, 
    features=["cos_theta_mu", "wc_set_d_c_9"], 
    features_dtype="float32",
)
dset_set.apply_std_scale()




# Set up model

# Define 2D Gaussian base distribution
base = nf.distributions.base.DiagGaussian(2)

# Define list of flows
num_layers = 32
flows = []
for i in range(num_layers):
    # Neural network with two hidden layers having 64 units each
    # Last layer is initialized by zeros making training more stable
    param_map = nf.nets.MLP([1, 64, 64, 2], init_zeros=True)
    # Add flow layer
    flows.append(nf.flows.AffineCouplingBlock(param_map))
    # Swap dimensions
    flows.append(nf.flows.Permute(2, mode='swap'))
    
# Construct flow model
model = nf.NormalizingFlow(base, flows)




# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
model = model.to(device)





# Plot target distribution

fig, ax = plt.subplots()
indices = torch.randint(
    low=0, 
    high=len(dset_set.train.features), 
    size=(100_000,)
)
ax.scatter(
    dset_set.train.features[indices][:,0],
    dset_set.train.features[indices][:,1],
    alpha=0.005
)
save_plot_and_close("plots/target.png")




# Plot initial flow distribution

grid_size = 200
xx, yy = torch.meshgrid(
    torch.linspace(-3, 3, grid_size), 
    torch.linspace(-3, 3, grid_size), 
    indexing="ij"
)
zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
zz = zz.to(device)

model.eval()
log_prob = model.log_prob(zz).to('cpu').view(*xx.shape)
model.train()
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0

plt.figure(figsize=(15, 15), dpi=50)
plt.pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
plt.gca().set_aspect('equal', 'box')
save_plot_and_close("plots/initial.png")






# Train model
epochs = 2
dloader = Data_Loader(
    dset_set.train, 
    batch_size=1_000, 
    shuffle=True
)


loss_hist = np.array([])

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

for ep in range(epochs):

    for features, _ in tqdm(dloader):
        optimizer.zero_grad()
        
        features = features.to(device)    
        # Compute loss
        loss = model.forward_kld(features)
        
        # Do backprop and optimizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()
        
        # Log loss
        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
        
    # Plot learned distribution
    model.eval()
    log_prob = model.log_prob(zz)
    model.train()
    prob = torch.exp(log_prob.to('cpu').view(*xx.shape))
    prob[torch.isnan(prob)] = 0

    plt.figure(figsize=(15, 15), dpi=50)
    plt.pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
    plt.gca().set_aspect('equal', 'box')
    save_plot_and_close(f"plots/model_epoch_{ep}.png")

# Plot loss
plt.figure(figsize=(10, 10), dpi=50)
plt.plot(loss_hist, label='loss')
plt.legend()
save_plot_and_close("plots/loss.png")




# # Plot target distribution
# f, ax = plt.subplots(1, 2, sharey=True, figsize=(15, 7))

# log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
# prob = torch.exp(log_prob)
# prob[torch.isnan(prob)] = 0

# ax[0].pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')

# ax[0].set_aspect('equal', 'box')
# ax[0].set_axis_off()
# ax[0].set_title('Target', fontsize=24)

# # Plot learned distribution
# model.eval()
# log_prob = model.log_prob(zz).to('cpu').view(*xx.shape)
# model.train()
# prob = torch.exp(log_prob)
# prob[torch.isnan(prob)] = 0

# ax[1].pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')

# ax[1].set_aspect('equal', 'box')
# ax[1].set_axis_off()
# ax[1].set_title('Real NVP', fontsize=24)

# plt.subplots_adjust(wspace=0.1)

# plt.show()