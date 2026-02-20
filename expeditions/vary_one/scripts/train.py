


from pathlib import Path

import pandas
import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm, Colormap, Normalize
from matplotlib.cm import ScalarMappable
import torch

from btokstll_sbi_tools.train.training import Trainer, calculate_class_weights_for_uniform_prior
from btokstll_sbi_tools.evaluation.predicting import Predictor
from btokstll_sbi_tools.util.normalize import normalize_using_reference_data
from btokstll_sbi_tools.util.hardware import select_device
from btokstll_sbi_tools.util.binning import bin
from btokstll_sbi_tools.util.dataset import Dataset
from btokstll_sbi_tools.util.save_load import open_torch_model_state_dict


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


train = False
name = "test_dc10_30_bins"
sim_type = "gen"
label_name = "delta_c_10"
feature_names = ["q_squared", "cos_theta_mu", "cos_theta_k", "chi"]
dc9_interval = (-1, 1)
num_bins = 30
device = select_device()
optimizer = "adam"
loss_fn = "cross_entropy"
learn_rate = 3e-4
learn_rate_sched = "reduce_lr_on_plateau"
learn_rate_sched_reduce_factor = 0.95
learn_rate_sched_patience = 0
learn_rate_sched_threshold = 0
learn_rate_sched_eps = 0
epochs = 250
epochs_checkpoint = 20
batch_size_train = 10_000
batch_size_eval = 10_000
path_to_parent_dir = "../models"
path_to_train_data = "../data/combined_vary_c_10_train.parquet"
path_to_val_data = "../data/combined_vary_c_10_val.parquet"

data_train = pandas.read_parquet(path_to_train_data).xs(sim_type, level="sim_type")
data_val = pandas.read_parquet(path_to_val_data).xs(sim_type, level="sim_type")


bins = numpy.linspace(*dc9_interval, num_bins+1)
binned_labels_train = bin(pandas.Series(data_train.index.get_level_values(label_name)), bins)
binned_labels_val = bin(pandas.Series(data_val.index.get_level_values(label_name)), bins)
bin_map = binned_labels_train[["bin_index", "bin_mid"]].drop_duplicates(subset="bin_index").sort_values("bin_index")["bin_mid"]
# breakpoint()
assert len(bin_map) == num_bins

normalized_features_train = normalize_using_reference_data(
    data_train[feature_names], 
    data_train[feature_names]
)
normalized_features_val = normalize_using_reference_data(
    data_val[feature_names], 
    data_train[feature_names], 
)

dataset_train = Dataset(
    features=normalized_features_train.astype("float32"), 
    labels=binned_labels_train["bin_index"]
)
dataset_val = Dataset(
    features=normalized_features_val.astype("float32"), 
    labels=binned_labels_val["bin_index"]
)

print(f"Num train examples: {len(dataset_train)}")
print(f"Num validation examples: {len(dataset_val)}")
    
loss_label_weights = calculate_class_weights_for_uniform_prior(
    binned_labels_train["bin_index"]
).to(torch.float32).to(device)

params = {
    "name": name,
    "parent_dir": path_to_parent_dir,
    "optimizer": optimizer,
    "optimizer_params": {"lr": learn_rate},
    "loss_fn": loss_fn,
    "loss_fn_params": {"weight": loss_label_weights},
    "batch_sizes": {"train": batch_size_train, "eval": batch_size_eval},
    "epochs": epochs,
    "checkpoint_epochs": epochs_checkpoint,
    "lr_scheduler": learn_rate_sched,
    "lr_scheduler_params": {
        "factor":learn_rate_sched_reduce_factor, 
        "patience":learn_rate_sched_patience, 
        "threshold":learn_rate_sched_threshold, 
        "eps":learn_rate_sched_eps
    },
}

model = MLP()

if train:
    trainer = Trainer(dataset_train, dataset_val, model, params)
    trainer.train(device)
else:
    path_to_final_model_state_dict = Path(path_to_parent_dir).joinpath(f"{name}/final.pt")
    model.load_state_dict(open_torch_model_state_dict(path_to_final_model_state_dict))


eval_sets_features = numpy.concatenate(
    [
        numpy.expand_dims(trial_data.iloc[:50_000].to_numpy(), 0)
        for _, trial_data in normalized_features_val.astype("float32")
        .groupby(level="trial_num")
    ]
)
breakpoint()
eval_sets_labels = binned_labels_val.groupby("original").first()
eval_sets_dataset = Dataset(eval_sets_features, eval_sets_labels["bin_index"])

predictor = Predictor(model, eval_sets_dataset.features, device)
log_probs = predictor.calc_log_probs()
expected_values = predictor.calc_expected_values(log_probs, bin_map)
breakpoint()
# print(log_probs)
# print(eval_sets_labels)

plt.style.use("dark_background")
plt.rcParams.update({
    "figure.dpi": 400, 
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern",

})
alpha = 0.85
fig, axs = plt.subplots(1, 2, figsize=(7,3), layout="constrained")
fig.get_layout_engine().set(wspace=0.06)
ax_dist = axs[0]
ax_expected = axs[1]
# norm = Normalize(vmin=-0.5, vmax=0.5)
norm = CenteredNorm(vcenter=0, halfrange=2)
cmap = mpl.colormaps["coolwarm"]
for l, label in zip(log_probs, eval_sets_labels["bin_index"].to_list()):
    color = cmap(norm(label))
    ax_dist.plot(bin_map, l.cpu(), color=color, alpha=alpha)
ax_expected.plot([-1.9, 0.9], [-1.9, 0.9], color="grey", zorder=-10, alpha=0.5, linestyle="--")
ax_expected.scatter(eval_sets_labels["bin_index"], expected_values, color=cmap(norm(eval_sets_labels["bin_index"])), alpha=alpha)
ax_dist.set_xticks([-2, -1, 0, 1])
ax_expected.set_xticks([-2, -1, 0, 1])
ax_expected.set_yticks([-2, -1, 0, 1])
# cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax_dist)
# cbar.set_ticks(range(num_bins))
# cbar.set_label(r"Actual $\delta C_9$ Bin", fontsize=15)
ax_dist.set_xlabel(r"$\delta C_9$", fontsize=13)
ax_dist.set_ylabel(r"$\log P(\delta C_9 \, | \, \textrm{dataset})$", fontsize=13)
ax_expected.set_xlabel(r"Actual $\delta C_9$", fontsize=13)
ax_expected.set_ylabel(r"Predicted $\delta C_9$", fontsize=13)
plt.savefig(Path(path_to_parent_dir).joinpath(f"{name}/predictions_sets_val.png"), bbox_inches="tight")
plt.close()





