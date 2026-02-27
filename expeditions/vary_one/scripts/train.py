
from pathlib import Path

from pandas import read_parquet
import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm, Colormap, Normalize
from matplotlib.cm import ScalarMappable
from torch import Tensor, linspace
from torch.nn import Module, Sequential, Linear, ReLU

from btokstll_sbi_tools.train import (
    train,
    calculate_reweights_uniform,
    Adam_Hyperparams,
    CrossEntropyLoss_Hyperparams,
    ReduceLROnPlateau_Hyperparams,
    Hyperparams,
    Loss_Table
)
from btokstll_sbi_tools.util import (
    bin_,
    Dataset, 
    Dataset_Set,
    load_torch_model_state_dict,
    save_torch_model_state_dict,
    select_device,
    std_scale,
)


### Config
retrain = True
name = "2026-02-24_test"
main_models_dir = Path("../models")
train_file_path = Path("../data/combined_vary_c_10_train.parquet")
eval_file_path = Path("../data/combined_vary_c_10_val.parquet")
feature_names = ["q_squared", "cos_theta_mu", "cos_theta_k", "chi"]
label_name = "delta_c_10" 
binned_interval = (-1.0, +1.0)
num_bins = 30
lr = 3e-4
train_batch_size = 10_000
eval_batch_size = 10_000
epochs = range(0, 300)
lr_scheduler_factor = 0.95
lr_scheduler_patience = 0
lr_scheduler_treshold = 0
lr_scheduler_eps = 0
shuffle = True
###


class MLP(Module):

    def __init__(
        self,
    ):
        super().__init__()  
        self.layers = Sequential(
            Linear(4, 16),
            ReLU(),
            Linear(16, 32),
            ReLU(),
            Linear(32, 32),
            ReLU(),
            Linear(32, num_bins),
        )

    def forward(
        self, 
        x:Tensor,
    ) -> Tensor:
        
        logits = self.layers(x)
        return logits


model = MLP()

device = select_device()

train_dataframe = read_parquet(train_file_path)
train_dataset = Dataset.from_pandas(
    features=train_dataframe[feature_names],
    labels=train_dataframe.index.get_level_values(level=label_name)
)
eval_dataframe = read_parquet(eval_file_path)
eval_dataset = Dataset.from_pandas(
    features=eval_dataframe[feature_names],
    labels=eval_dataframe.index.get_level_values(level=label_name)
)

bin_edges = linspace(*binned_interval, num_bins+1)
train_dataset.labels = bin_(
    data=train_dataset.labels, 
    bin_edges=bin_edges
)
eval_dataset.labels = bin_(
    data=eval_dataset.labels, 
    bin_edges=bin_edges
)

reweights = calculate_reweights_uniform(
    binned_labels=train_dataset.lables,
    num_bins=num_bins
)

train_dataset.features = std_scale(
    data=train_dataset.features, 
    reference=train_dataset.features
)
eval_dataset.features = std_scale(
    data=eval_dataset.features, 
    reference=train_dataset.features
)

hyperparams = Hyperparams(
    optimizer=Adam_Hyperparams(
        lr=lr
    ),
    train_batch_size=train_batch_size,
    eval_batch_size=eval_batch_size,
    shuffle=shuffle,
    epochs=epochs,
    loss_fn=CrossEntropyLoss_Hyperparams(
        weight=reweights,
    ),
    lr_scheduler=ReduceLROnPlateau_Hyperparams(
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        threshold=lr_scheduler_treshold,
        eps=lr_scheduler_eps,
    ),
    num_bins=num_bins,
    binned_interval_left=binned_interval[0],
    binned_interval_right=binned_interval[1],
)

model_dir = main_models_dir.joinpath(name)
model_dir.mkdir()

if retrain:
    loss_table = train(
        model=model, 
        hyperparams=hyperparams, 
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset, 
        device=device,
    )
    main_models_dir.mkdir(name)
    save_torch_model_state_dict(
        model=model, 
        path=model_dir.joinpath("model.pt")
    )
    loss_table.save_table_as_json(model_dir.joinpath("loss.json"))
    hyperparams....
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
eval_sets_labels = binned_labels_val.groupby("original").first()
eval_sets_dataset = Dataset(eval_sets_features, eval_sets_labels["bin_index"])

predictor = Predictor(model, eval_sets_dataset.features, device)
log_probs = predictor.calc_log_probs()
expected_values = predictor.calc_expected_values(log_probs, bin_map)

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





