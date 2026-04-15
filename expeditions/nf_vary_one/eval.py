
from matplotlib.pyplot import subplots

from torch import linspace, cartesian_prod, no_grad

from normflows import NormalizingFlow
from normflows.flows import AffineCouplingBlock, Permute
from normflows.distributions.base import DiagGaussian
from normflows.nets import MLP

from btokstll_sbi_tools.model.util import select_device, load_torch_model_state_dict
from btokstll_sbi_tools.util.plot import (
    turn_on_dark_plots, 
    turn_on_hq_plots, 
    set_ax_labels, 
    set_ax_bounds,
    set_ax_ticks,
    save_fig_and_close,
)
from btokstll_sbi_tools.model.util import Dataset_Set_File_Paths, Dataset_Set


# plot setup

turn_on_dark_plots()
turn_on_hq_plots()
plot_interval = (-2.25, 2.25)
ticks = [-2, -1, 0, 1, 2]
xlabel = r"$\cos\theta_\mu$ (normalized)"
ylabel = r"$\delta C_9$ (normalized)"
label_fontsize = 18


# device setup

device = select_device()


# load data

data_paths = Dataset_Set_File_Paths(
    "data/vary_c9_train.parquet", 
    "data/vary_c9_val.parquet",
)
dset_set = Dataset_Set.from_pandas_parquet_files(
    data_paths, 
    features=["cos_theta_mu",],
    label="wc_set_d_c_9", 
    features_dtype="float32",
    labels_dtype="float32",
)
dset_set.apply_std_scale()
dset_set.val.group_by_trial()


# setup model

base = DiagGaussian(2)

num_layers = 32
flows = []
for i in range(num_layers):
    param_map = MLP([1, 64, 64, 2], init_zeros=True)
    flows.append(AffineCouplingBlock(param_map))
    flows.append(Permute(2, mode='swap')) 
   
model = NormalizingFlow(base, flows)
model = model.to(device)
model.load_state_dict(
    load_torch_model_state_dict(
        "models/model.pt"
    )
)
model.eval()


# Setup sampling

num_samples = 200
sample_at = linspace(*plot_interval, num_samples)


# Input to model

features = dset_set.val.grouped_features[0].squeeze()
label = dset_set.val.grouped_labels[0].unique().item()

input_to_model = cartesian_prod(
    features, 
    sample_at,
)
input_to_model = input_to_model.to(device)

model.eval()
with no_grad():
    event_log_probs = model.log_prob(input_to_model)
    event_log_probs = event_log_probs.view( 
        -1, 
        num_samples,
    )


# Plot first 3 event distributions

fig, ax = subplots()
colors = ['#377eb8', '#ff7f00', '#4daf4a']
alpha=0.8

for log_p, feat, color in zip(
    event_log_probs[:3], 
    features, 
    colors,
):
    ax.plot(
        sample_at, 
        log_p.cpu(), 
        color=color, 
        alpha=alpha, 
        label=(
            r"$\cos\theta_\mu=" 
            f"{feat.item():.2f}"
            r"$"
        ),
    )
    ax.axvline(
        label, 
        color=color, 
        linestyle="--", 
        alpha=alpha, 
        label=(
            r"$\delta C_9="
            f"{label:.2f}"
            r"$"
        ),
    )

set_ax_labels(
    ax, 
    ylabel, 
    r"$\propto p(\delta C_9 \,|\, \cos\theta_\mu)$", 
    fontsize=label_fontsize
)
set_ax_bounds(
    ax, 
    xbounds=plot_interval, 
    ybounds=(-5, 0.2),
)
set_ax_ticks(
    ax, 
    xticks=ticks,
)
ax.legend(loc="upper right")
ax.set_box_aspect(1)
save_fig_and_close(f"plots/eval_model_dists.png")
