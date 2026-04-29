
from matplotlib.pyplot import subplots
from pandas import read_parquet
from torch import (
    tensor,
    linspace, 
    cartesian_prod, 
    no_grad, 
    sum,
)
from normflows import NormalizingFlow
from normflows.flows import AffineCouplingBlock, Permute
from normflows.distributions.base import DiagGaussian
from normflows.nets import MLP

from btokstll_sbi_tools.hardware import select_device
from btokstll_sbi_tools.state_dict import load_torch_model_state_dict
from btokstll_sbi_tools.plot import (
    turn_on_dark_plots, 
    turn_on_hq_plots, 
    set_ax_labels, 
    set_ax_bounds,
    set_ax_ticks,
    save_fig_and_close,
)
from btokstll_sbi_tools.dataset import DatasetSet
from btokstll_sbi_tools.scaling import Std_Scaler
from btokstll_sbi_tools.group import group


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

dset_set = DatasetSet.from_pandas_parquet_files(
    train_path="data/vary_dc9_train.parquet", 
    val_path="data/vary_dc9_val.parquet", 
    features=["cos_theta_mu",],
    labels=["wc_set_d_c_9",], 
    features_dtype="float32",
    labels_dtype="float32",
)

val_trial_nums = tensor(
    read_parquet("data/vary_dc9_val.parquet")
    .index.get_level_values("trial_num")
)


# standard scale

unscaled_train_feature_means = dset_set.train.features.mean(dim=0)
unscaled_train_feature_stds = dset_set.train.features.std(dim=0)
unscaled_train_label_means = dset_set.train.labels.mean(dim=0)
unscaled_train_label_stds = dset_set.train.labels.std(dim=0)

features_std_scaler = Std_Scaler(
    means=unscaled_train_feature_means, 
    stdevs=unscaled_train_feature_stds,
)
labels_std_scaler = Std_Scaler(
    means=unscaled_train_label_means,
    stdevs=unscaled_train_label_stds,
)

dset_set.val.features = features_std_scaler.std_scale(dset_set.val.features)
dset_set.val.labels = features_std_scaler.std_scale(dset_set.val.labels)


# group by trial

grouped_val_features = group(
    data=dset_set.val.features, 
    by=val_trial_nums
)
grouped_val_labels = group(
    data=dset_set.val.labels, 
    by=val_trial_nums
)


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
fig, ax = subplots()
alpha = 0.8
for i in range(5):
    features = grouped_val_features[i].squeeze()
    label = grouped_val_labels[i].unique().item()

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
        set_log_probs = sum(event_log_probs, dim=0)


    # plot set log probs

    ax.plot(
        sample_at,
        set_log_probs.cpu(),
        alpha=alpha,
        label=f"Set Log Probs: {label}"
    )
    # ax.axvline(
    #     label, 
    #     alpha=alpha, 
    #     linestyle="--", 
    #     label=(
    #         r"$\delta C_9="
    #         f"{label:.2f}"
    #         r"$"
    #     ),
    # )

set_ax_labels(
    ax, 
    ylabel, 
    r"$\log p(\delta C_9 \,|\, \textrm{Dataset}) + \textrm{Const.}$", 
    fontsize=label_fontsize
)
set_ax_bounds(
    ax, 
    xbounds=plot_interval, 
    ybounds=(-1e5, -4e4),
)
set_ax_ticks(
    ax, 
    xticks=ticks,
)
ax.legend(loc="upper right")
ax.set_box_aspect(1)
save_fig_and_close(f"plots/eval_model_set_dist_many.png")



# # Plot first 3 event distributions

# fig, ax = subplots()
# colors = ['#377eb8', '#ff7f00', '#4daf4a']
# alpha=0.8

# for log_p, feat, color in zip(
#     event_log_probs[:3], 
#     features, 
#     colors,
# ):
#     ax.plot(
#         sample_at, 
#         log_p.cpu(), 
#         color=color, 
#         alpha=alpha, 
#         label=(
#             r"$\cos\theta_\mu=" 
#             f"{feat.item():.2f}"
#             r"$"
#         ),
#     )
#     ax.axvline(
#         label, 
#         color=color, 
#         linestyle="--", 
#         alpha=alpha, 
#         label=(
#             r"$\delta C_9="
#             f"{label:.2f}"
#             r"$"
#         ),
#     )

# set_ax_labels(
#     ax, 
#     ylabel, 
#     r"$\log p(\delta C_9 \,|\, \textrm{Event}) + \textrm{Const.}$", 
#     fontsize=label_fontsize
# )
# set_ax_bounds(
#     ax, 
#     xbounds=plot_interval, 
#     ybounds=(-5, 0.2),
# )
# set_ax_ticks(
#     ax, 
#     xticks=ticks,
# )
# ax.legend(loc="upper right")
# ax.set_box_aspect(1)
# save_fig_and_close(f"plots/eval_model_dists.png")
