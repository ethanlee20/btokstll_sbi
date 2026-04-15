
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
from btokstll_sbi_tools.util.plot import (
    set_ax_bounds, 
    set_ax_labels,
    set_ax_ticks,
    turn_on_dark_plots,
    turn_on_hq_plots,
    save_fig_and_close,
)
from btokstll_sbi_tools.model.util import (
    select_device, 
    save_torch_model_state_dict, 
    load_torch_model_state_dict,
)


# device setup

device = select_device()


# plot setup

turn_on_dark_plots()
turn_on_hq_plots()
plot_interval = (-2.25, 2.25)
ticks = [-2, -1, 0, 1, 2]
xlabel = r"$\cos\theta_\mu$ (normalized)"
ylabel = r"$\delta C_9$ (normalized)"
label_fontsize = 18


# grid setup

grid_size = 200
xx, yy = torch.meshgrid(
    torch.linspace(*plot_interval, grid_size), 
    torch.linspace(*plot_interval, grid_size), 
    indexing="ij"
)
zz = torch.cat(
    [
        xx.unsqueeze(2), 
        yy.unsqueeze(2)
    ], 
    dim=2,
).view(-1, 2)
zz = zz.to(device)
breakpoint()

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


# setup model

base = nf.distributions.base.DiagGaussian(2)

num_layers = 32
flows = []
for i in range(num_layers):
    param_map = nf.nets.MLP([1, 64, 64, 2], init_zeros=True)
    flows.append(nf.flows.AffineCouplingBlock(param_map))
    flows.append(nf.flows.Permute(2, mode='swap')) 
   
model = nf.NormalizingFlow(base, flows)
model = model.to(device)


# Plot target distribution

fig, ax = plt.subplots()
ax.hist2d(
    dset_set.train.features[:,0], 
    dset_set.train.features[:,1], 
    density=True,
    bins=100,
    range=(plot_interval, plot_interval)
)
ax.set_aspect("equal")
set_ax_ticks(ax, ticks)
set_ax_bounds(ax, plot_interval)
set_ax_labels(
    ax, 
    xlabel, 
    ylabel, 
    fontsize=label_fontsize
)
save_fig_and_close("plots/target.png")


# Plot initial flow distribution

model.eval()
log_prob = model.log_prob(zz).to('cpu').view(*xx.shape)
model.train()
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0

fig, ax = plt.subplots()
ax.pcolormesh(xx, yy, prob.data.numpy())
ax.set_aspect("equal")
set_ax_labels(ax, xlabel, ylabel, fontsize=label_fontsize)
save_fig_and_close("plots/initial.png")


# Train model

retrain = False
if retrain:

    epochs = 3
    dloader = Data_Loader(
        dset_set.train, 
        batch_size=1_000, 
        shuffle=True
    )
    loss_hist = np.array([])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    it = 0
    pic_at = (0, 50, 100, 999, 1999, 2999)
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
            
            if it in pic_at:
                # Plot learned distribution
                model.eval()
                log_prob = model.log_prob(zz)
                model.train()
                prob = torch.exp(log_prob.to('cpu').view(*xx.shape))
                prob[torch.isnan(prob)] = 0

                fig, ax = plt.subplots()
                ax.pcolormesh(xx, yy, prob.data.numpy())
                ax.set_aspect('equal')
                set_ax_labels(ax, xlabel, ylabel, fontsize=label_fontsize)
                save_fig_and_close(f"plots/model_at_{it}.png")
            
            it += 1

    save_torch_model_state_dict(model, "models/model.pt")

    fig, ax = plt.subplots()
    ax.plot(loss_hist, label='loss')
    ax.legend()
    save_fig_and_close("plots/loss.png")

else:
    model.load_state_dict(
        load_torch_model_state_dict("models/model.pt")
    )


# Plot unnormalized p(y | x) for 3 training data points

num_samples = 3
idx = torch.randint(
    low=0, 
    high=len(dset_set.train.features), 
    size=(num_samples,)
)
data = dset_set.train.features[idx]
labels = data[:, 1]
features = data[:, 0]

input_to_model = torch.cartesian_prod(
    features, 
    yy[0,:],
)#.view(
#     num_samples, 
#     grid_size, 
#     -1
# )
input_to_model = input_to_model.to(device)

model.eval()
with torch.no_grad():
    log_prob = model.log_prob(input_to_model)
    prob = torch.exp(log_prob.to('cpu'))
    prob[torch.isnan(prob)] = 0
    prob = prob.view(num_samples, grid_size, -1)

fig, ax = plt.subplots()
colors = ['#377eb8', '#ff7f00', '#4daf4a'][:num_samples]
alpha=0.8
for dist, label, feat, color in zip(prob, labels, features, colors):
    ax.plot(
        yy[0, :].numpy(), 
        dist, 
        color=color, 
        alpha=alpha, 
        label=(
            r"$\cos\theta_\mu=" 
            f"{feat.item():.2f}"
            r"$"
        ),
    )
    ax.axvline(
        label.item(), 
        color=color, 
        linestyle="--", 
        alpha=alpha, 
        label=(
            r"$\delta C_9="
            f"{label.item():.2f}"
            r"$"
        ),
    )
set_ax_labels(
    ax, 
    ylabel, 
    r"$\propto p(\delta C_9 \,|\, \cos\theta_\mu)$", 
    fontsize=label_fontsize
)
ax.legend(loc="upper right")
ax.set_box_aspect(1)
save_fig_and_close(f"plots/model_dists.png")

with torch.no_grad():
    model.eval()
    log_prob = model.log_prob(zz)
    model.train()
    prob = torch.exp(log_prob.to('cpu').view(*xx.shape))
    prob[torch.isnan(prob)] = 0

    fig, ax = plt.subplots()
    ax.pcolormesh(xx, yy, prob.data.numpy())
    for label, feat, color in zip(labels, features, colors):
        ax.axvline(feat.item(), color=color,)
        ax.scatter(feat.item(), label.item(), color=color,)

    ax.set_aspect('equal')

    set_ax_labels(ax, xlabel, ylabel, fontsize=label_fontsize)
    save_fig_and_close(f"plots/final_model_annotated.png")







