
from matplotlib.pyplot import subplots
from pandas import read_parquet
from torch import (
    exp,
    isnan,
    tensor,
    linspace, 
    cartesian_prod, 
    no_grad, 
    sum,
    fmax,
)
import normflows as nf
from normflows import NormalizingFlow
from normflows.flows import AffineCouplingBlock, Permute
from normflows.distributions.base import DiagGaussian, GaussianMixture
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
    feature_names=["cos_theta_mu",],
    label_names=["delta_wc_values_dc9",], 
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
dset_set.val.labels = labels_std_scaler.std_scale(dset_set.val.labels)


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


models = []
for k in (16, 8):
    K = k
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
    q0 = nf.distributions.DiagGaussian(2, trainable=False)

    # base = DiagGaussian(2)

    # num_layers = 4
    #flows = []
    #for i in range(num_layers):
    #    param_map = MLP([1, 32, 32, 2], init_zeros=True)
    #    flows.append(AffineCouplingBlock(param_map))
    #    flows.append(Permute(2, mode='swap')) 
    
    model = NormalizingFlow(q0, flows)
    models.append(model)


# base = DiagGaussian(2)

#num_layers = 4
#flows = []
#for i in range(num_layers):
#    param_map = MLP([1, 32, 32, 2], init_zeros=True)
#    flows.append(AffineCouplingBlock(param_map))
#    flows.append(Permute(2, mode='swap')) 
   
#model = NormalizingFlow(base, flows)

models[0] = models[0].to(device)
models[0].load_state_dict(
    load_torch_model_state_dict(
        "models/model_0.pt"
    )
)
models[0].eval()

models[1] = models[1].to(device)
models[1].load_state_dict(
    load_torch_model_state_dict(
        "models/model_1.pt"
    )
)
models[1].eval()


# Setup sampling

num_samples = 200
sample_at = linspace(*plot_interval, num_samples)


# Input to model

fig, ax = subplots()
alpha = 0.8

colors = ['#377eb8', '#ff7f00', '#4daf4a']

features = grouped_val_features[0:3]
labels = grouped_val_labels[0:3]

for feat, label, color in zip(features, labels, colors):
    
    feat = feat.squeeze()
    label = label.unique().item()

    input_to_model = cartesian_prod(
        feat, 
        sample_at,
    )
    input_to_model = input_to_model.to(device)

    models[0].eval()
    models[1].eval()
    with no_grad():
        set_log_probs_s = []
        for model in models:
            event_log_probs = model.log_prob(input_to_model)
            event_log_probs = event_log_probs.view( 
                -1, 
                num_samples,
            )
            set_log_probs = sum(event_log_probs, dim=0)
            set_log_probs_s.append(set_log_probs)

        set_log_probs = set_log_probs_s[0] + set_log_probs_s[1]

    # plot set log probs

    #set_probs = exp(set_log_probs - set_log_probs[~isnan(set_log_probs)].max())
    #set_probs[isnan(set_probs)] = 0

    ax.plot(
        labels_std_scaler.undo_std_scale(sample_at[25:-25]),
        set_log_probs.cpu()[25:-25],
        alpha=alpha,
        color=color,
    )

    ax.axvline(
        labels_std_scaler.undo_std_scale(label), 
        alpha=alpha, 
        color=color,
        linestyle="--", 
    )

set_ax_labels(
    ax, 
    r"$\delta C_9$", 
    r"$\log p(\delta C_9 \,|\, \textrm{Dataset}) + \textrm{Const.}$", 
    fontsize=label_fontsize
)
# ax.set_xbound(*plot_interval)
# set_ax_ticks(
#     ax, 
#     xticks=ticks,
# )
# ax.legend(loc="upper right")
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
