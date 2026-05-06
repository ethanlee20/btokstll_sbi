
from pathlib import Path

from tqdm import tqdm
from matplotlib.pyplot import subplots, close
from torch import (
    tensor,
    meshgrid, 
    linspace, 
    cat, 
    exp, 
    isnan, 
    isinf, 
    randint, 
    cartesian_prod, 
    no_grad,
)
from torch.optim import AdamW
import normflows as nf
from normflows import NormalizingFlow
from normflows.distributions.base import DiagGaussian, GaussianMixture
from normflows.nets import MLP
from normflows.flows import AffineCouplingBlock, Permute

from btokstll_sbi_tools.dataset import (
    DatasetSet,
)
from btokstll_sbi_tools.train import Data_Loader
from btokstll_sbi_tools.plot import (
    turn_on_dark_plots,
    turn_on_hq_plots,
)
from btokstll_sbi_tools.hardware import (
    select_device, 
)
from btokstll_sbi_tools.state_dict import (
    save_torch_model_state_dict, 
    load_torch_model_state_dict,
)
from btokstll_sbi_tools.scaling import Std_Scaler


####################
### setup device ###
####################

device = select_device()


######################
### setup plotting ###
######################

turn_on_dark_plots()
turn_on_hq_plots()

fancy_var_names = {
    "cos_theta_mu": r"$\cos\theta_\mu$", 
    "delta_wc_values_dc9": r"$\delta C_9$",
}

label_fontsize = 18

plot_dir = Path("plots/model/")
plot_dir.mkdir(exist_ok=True)


#################
### load data ###
#################

train_data_path = "data/vary_dc9_train.parquet"
val_data_path = "data/vary_dc9_val.parquet"

feature_names = ["cos_theta_mu",]
label_names = ["delta_wc_values_dc9",]

features_dtype = "float32"
labels_dtype = "float32"

dset_set = DatasetSet.from_pandas_parquet_files(
    train_path=train_data_path, 
    val_path=val_data_path, 
    feature_names=feature_names, 
    label_names=label_names,
    features_dtype=features_dtype,
    labels_dtype=labels_dtype,
)


###########################
### standard scale data ###
###########################

features_std_scaler = Std_Scaler(
    means=dset_set.train.features.mean(dim=0), 
    stdevs=dset_set.train.features.std(dim=0),
)
labels_std_scaler = Std_Scaler(
    means=dset_set.train.labels.mean(dim=0), 
    stdevs=dset_set.train.labels.std(dim=0),
)

dset_set.train.features = features_std_scaler.std_scale(
    dset_set.train.features
)
dset_set.val.features = features_std_scaler.std_scale(
    dset_set.val.features
)
dset_set.train.labels = labels_std_scaler.std_scale(
    dset_set.train.labels
)
dset_set.val.labels = labels_std_scaler.std_scale(
    dset_set.val.labels
)


##################
### setup grid ###
##################

grid_size = 200
xx, yy = meshgrid(
    linspace(-3, 3, grid_size), 
    linspace(-3, 3, grid_size), 
    indexing="ij"
)
zz = cat(
    [
        xx.unsqueeze(2), 
        yy.unsqueeze(2)
    ], 
    dim=2,
).view(-1, 2)


###################
### setup model ###
###################

K = 16
# torch.manual_seed(0)

latent_size = 2
hidden_units = 128
hidden_layers = 2

flows = []
for i in range(K):
    flows += [
        nf.flows.AutoregressiveRationalQuadraticSpline(
            latent_size, 
            hidden_layers, 
            hidden_units,
        )
    ]
    flows += [nf.flows.LULinearPermute(latent_size)]

# Set base distribuiton
low = dset_set.train.features.min().item()
high = dset_set.train.features.max().item()
q0 = nf.distributions.Uniform(
    2, 
)

# base = DiagGaussian(2)

# num_layers = 4
#flows = []
#for i in range(num_layers):
#    param_map = MLP([1, 32, 32, 2], init_zeros=True)
#    flows.append(AffineCouplingBlock(param_map))
#    flows.append(Permute(2, mode='swap')) 
   
model = NormalizingFlow(q0, flows)


###################
### train model ###
###################

retrain = True

epochs = 40
batch_size = 10_000
lr = 3e-4

if retrain:

    model = model.to(device)

    dloader = Data_Loader(
        dset_set.train, 
        batch_size=batch_size, 
        shuffle=True
    )
    optimizer = AdamW(model.parameters(), lr=lr)

    epoch_losses = []
    for ep in range(epochs):
        batch_losses = []
        for features, labels in dloader:
            model_input = cat([features, labels], dim=1,)
            optimizer.zero_grad()
            batch_loss = model.forward_kld(model_input.to(device))
            if ~(isnan(batch_loss) | isinf(batch_loss)):
                batch_loss.backward()
                optimizer.step()
            batch_losses.append(batch_loss.to('cpu').item())
        epoch_loss = tensor(batch_losses).mean().item()
        epoch_losses.append(epoch_loss)
    
        model.eval()
        log_prob = model.log_prob(zz.to(device))
        model.train()
        prob = exp(log_prob.view(*xx.shape))
        prob[isnan(prob)] = 0

        fig, ax = subplots(layout="constrained")
        ax.pcolormesh(
            features_std_scaler.undo_std_scale(xx), 
            labels_std_scaler.undo_std_scale(yy), 
            prob.data.cpu()
        )
        ax.set_ylim(-10.1, 0.1)
        ax.set_xlim(-1.01, 1.01)
        ax.set_box_aspect(1)
        ax.set_xlabel(
            fancy_var_names["cos_theta_mu"], 
            fontsize=label_fontsize
        )
        ax.set_ylabel(
            fancy_var_names["delta_wc_values_dc9"], 
            fontsize=label_fontsize
        )
        plot_path = plot_dir.joinpath(f"{ep}.png")
        fig.savefig(plot_path, bbox_inches="tight")
        close(fig)

        fig, axs = subplots(2,1, layout="constrained")
        axs.flat[0].plot(epoch_losses, label="loss")
        axs.flat[1].semilogy(epoch_losses, label="loss")
        axs.flat[0].legend()
        plot_path = plot_dir.joinpath("loss.png")
        fig.savefig(plot_path, bbox_inches="tight")
        close(fig)
        
    model_path = "models/model.pt"
    save_torch_model_state_dict(
        model, 
        model_path, 
        overwrite_ok=True,
    )

else:
    model_path = "models/model.pt"
    model.load_state_dict(
        load_torch_model_state_dict(model_path)
    )
    model = model.to(device)


########################
### plot predictions ###
########################

num_samples = 3
sample_indices = randint(
    low=0, 
    high=len(dset_set.train), 
    size=(num_samples,)
)
labels = dset_set.train.labels[sample_indices].squeeze()
features = dset_set.train.features[sample_indices].squeeze()

input_to_model = cartesian_prod(
    features, 
    yy[0,:],
).to(device)

model.eval()
with no_grad():
    log_prob = model.log_prob(input_to_model)
    prob = exp(log_prob.to('cpu'))
    prob[isnan(prob)] = 0
    prob = prob.view(num_samples, grid_size, -1)

fig, ax = subplots(layout="constrained")
colors = ['#377eb8', '#ff7f00', '#4daf4a']
alpha=0.8
for dist, label, feat, color in zip(prob, labels, features, colors):
    ax.plot(
        labels_std_scaler.undo_std_scale(yy[0, :]).numpy(), 
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
        labels_std_scaler.undo_std_scale(label).item(), 
        color=color, 
        linestyle="--", 
        alpha=alpha, 
        label=(
            r"$\delta C_9="
            f"{labels_std_scaler.undo_std_scale(label).item():.2f}"
            r"$"
        ),
    )
ax.set_xlabel(fancy_var_names["delta_wc_values_dc9"], fontsize=label_fontsize)
ax.set_ylabel(r"$\propto p(\delta C_9 \,|\, \cos\theta_\mu)$", fontsize=label_fontsize)
# ax.legend(loc="upper right")
ax.set_box_aspect(1)
fig.savefig(plot_dir.joinpath("dists.png"), bbox_inches="tight")
close(fig)





