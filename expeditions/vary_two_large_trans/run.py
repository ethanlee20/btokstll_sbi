
from pathlib import Path
from dataclasses import asdict

from pandas import read_parquet
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig, subplots, close, plot, show
from matplotlib.colors import Normalize
from torch import Tensor, linspace, cat, unsqueeze, float32, ones, zeros, ones_like, zeros_like, int64
from torch.nn import Module, Sequential, Linear, ReLU

from btokstll_sbi_tools.model import (
    train,
    AdamW_Hyperparams,
    CrossEntropyLoss_Hyperparams,
    ReduceLROnPlateau_Hyperparams,
    CosineAnnealingLR_Hyperparams,
    Hyperparams,
    plot_predictions,
    select_device,
    Dataset,
    Dataset_Metadata,
    Dataset_Set,
    Dataset_Set_File_Paths,
    save_torch_model_state_dict,
    load_torch_model_state_dict,
    calc_log_probs,
    calc_set_log_probs,
    calc_expected_value,
)
from btokstll_sbi_tools.util.misc import (
    Interval,
    save_plot_and_close, 
    setup_plotting,
    to_int,
)


# general stuff

train_vary_one = False
train_vary_two = True
models_dir = Path("models")
feature_names = ["q_squared", "cos_theta_mu", "cos_theta_k", "chi"]
wc_name = lambda wc: f"wc_set_d_c_{wc}"
bound_name = lambda wc, left_or_right: f"wc_dist_d_c_{wc}_{left_or_right}"
binned_interval = Interval(-10, 0)
num_bins = 30
shuffle = True
device = select_device()
setup_plotting()

class Base_MLP(Module):
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
    

# datasets

vary_one_dset_set = Dataset_Set.from_pandas_parquet_files(
    Dataset_Set_File_Paths(
        Path("data/vary_c9_train.parquet"), 
        Path("data/vary_c9_val.parquet"), 
    ), 
    feature_names, 
    wc_name(9), 
    features_dtype="float32",
)
vary_two_dset_set = Dataset_Set.from_pandas_parquet_files(
    Dataset_Set_File_Paths(
        "data/vary_c7_c9_train.parquet", 
        "data/vary_c7_c9_val.parquet",
    ), 
    feature_names, 
    wc_name(9), 
    features_dtype="float32"
)
for dset_set in (vary_one_dset_set, vary_two_dset_set):
    dset_set.apply_std_scale()
    dset_set.apply_binning(binned_interval, num_bins)
    dset_set.calc_label_reweights(num_labels=num_bins)


# hyperparameters

vary_one_hyperparams = Hyperparams(
    optimizer=AdamW_Hyperparams(
        lr=3e-4
    ),
    train_batch_size=10_000,
    eval_batch_size=10_000,
    shuffle=shuffle,
    epochs=Interval(0, 200),
    loss_fn=CrossEntropyLoss_Hyperparams(
        weight=vary_one_dset_set.train.metadata.bin_reweights.to(device),
    ),
    # lr_scheduler=Ex
    # lr_scheduler=None,
    # lr_scheduler=CosineAnnealingLR_Hyperparams(
    #     T_max=to_int(epochs.right-epochs.left),
    # ),
    lr_scheduler=ReduceLROnPlateau_Hyperparams(
        factor=0.95,
        patience=0,
        threshold=0,
        eps=0,
    ),
    num_bins=num_bins,
    binned_interval=binned_interval,
)
vary_two_hyperparams = Hyperparams(
    optimizer=AdamW_Hyperparams(
        lr=3e-6
    ),
    train_batch_size=10_000,
    eval_batch_size=10_000,
    shuffle=shuffle,
    epochs=Interval(200, 1000),
    loss_fn=CrossEntropyLoss_Hyperparams(
        weight=vary_one_dset_set.train.metadata.bin_reweights.to(device),
    ),
    # lr_scheduler=Ex
    # lr_scheduler=None,
    # lr_scheduler=CosineAnnealingLR_Hyperparams(
    #     T_max=to_int(epochs.right-epochs.left),
    # ),
    lr_scheduler=ReduceLROnPlateau_Hyperparams(
        factor=0.994,
        patience=0,
        threshold=0,
        eps=0,
    ),
    num_bins=num_bins,
    binned_interval=binned_interval,
)

# model paths

vary_one_model_name = "pred_c9_vary_c9"
vary_one_model_dir = models_dir.joinpath(vary_one_model_name)
vary_one_model_file_path = vary_one_model_dir.joinpath("model.pt")

vary_two_model_name = "pred_c9_vary_c7_c9"
vary_two_model_dir = models_dir.joinpath(vary_two_model_name)
vary_two_model_file_path = vary_two_model_dir.joinpath("model.pt")


# models

vary_one_model = Base_MLP().to(device)
vary_two_model = Base_MLP().to(device)


# training

if train_vary_one:
    vary_one_model_dir.mkdir()
    loss_table = train(
        model=vary_one_model, 
        hyperparams=vary_one_hyperparams, 
        train_dataset=vary_one_dset_set.train, 
        eval_dataset=vary_one_dset_set.val, 
        device=device,
    )
    save_torch_model_state_dict(
        model=vary_one_model, 
        path=vary_one_model_file_path,
    )
    loss_table.save_table_as_json(vary_one_model_dir.joinpath("loss.json"))
    # hyperparams_dict = asdict(hyperparams)
    # with open(model_dir.joinpath("hyperparams.json"), 'x') as f:
    #     dump(hyperparams_dict, f)
    loss_dict = loss_table.as_lists()
    loss_dict["epochs"] = [int(ep) for ep in loss_dict["epochs"]]
    fig, ax = subplots(figsize=(5,4))
    colors = {"train": "goldenrod", "eval": "skyblue"}
    for split in ("train", "eval"):
        ax.scatter(loss_dict["epochs"], loss_dict[split], label=split, s=1, color=colors[split])
    ax.set_ylabel("Cross Entropy Loss", fontsize=13)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.legend(fontsize=13, markerscale=5)
    savefig(Path("plots/").joinpath(f"loss_{vary_one_model_name}"), bbox_inches="tight")
    close()
else:
    vary_one_model.load_state_dict(
        load_torch_model_state_dict(
            vary_one_model_file_path
        )
    )
if train_vary_two:
    vary_two_model_dir.mkdir()
    vary_two_model.load_state_dict(
        load_torch_model_state_dict(
            vary_one_model_file_path
        )
    )
    loss_table = train(
        model=vary_two_model, 
        hyperparams=vary_two_hyperparams, 
        train_dataset=vary_two_dset_set.train, 
        eval_dataset=vary_two_dset_set.val, 
        device=device,
    )
    save_torch_model_state_dict(
        model=vary_two_model, 
        path=vary_two_model_file_path,
    )
    loss_table.save_table_as_json(vary_two_model_dir.joinpath("loss.json"))
    # hyperparams_dict = asdict(hyperparams)
    # with open(model_dir.joinpath("hyperparams.json"), 'x') as f:
    #     dump(hyperparams_dict, f)
    loss_dict = loss_table.as_lists()
    loss_dict["epochs"] = [int(ep) for ep in loss_dict["epochs"]]
    fig, ax = subplots(figsize=(5,4))
    colors = {"train": "goldenrod", "eval": "skyblue"}
    for split in ("train", "eval"):
        ax.scatter(loss_dict["epochs"], loss_dict[split], label=split, s=1, color=colors[split])
    ax.set_ylabel("Cross Entropy Loss", fontsize=13)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.legend(fontsize=13, markerscale=5)
    savefig(Path("plots/").joinpath(f"loss_{vary_two_model_name}"), bbox_inches="tight")
    close()
else:
    vary_two_model.load_state_dict(
        load_torch_model_state_dict(
            vary_two_model_file_path
        )
    )


# eval

vary_one_dset_set.val.to(device)
vary_two_dset_set.val.to(device)

vary_one_dset_set.val.group_by_trial()
vary_two_dset_set.val.group_by_trial()

# # vary one on vary one
logits = cat(
    [
        vary_one_model(set_).unsqueeze(0) 
        for set_ in vary_one_dset_set.val.grouped_features
    ]
)
log_probs = calc_set_log_probs(logits) 
expected_values = calc_expected_value(
    log_probs, 
    vary_one_dset_set.val.metadata.bin_mids.to(device),
)
labels = Tensor(
    [
        labels.unique().item() 
        for labels in vary_one_dset_set.val.grouped_labels
    ]
).to(int64)
unbinned_labels = vary_one_dset_set.val.metadata.bin_mids[labels]
plot_file_path = Path(f"plots/{vary_one_model_name}_preds_vary_one.png")
plot_predictions(
    vary_one_dset_set.val.metadata.bin_edges.cpu().detach(), 
    log_probs.cpu().detach(), 
    expected_values.cpu().detach(), 
    unbinned_labels, 
    9, 
    plot_file_path, 
)

# # vary one on vary two
logits = cat(
    [
        vary_one_model(set_).unsqueeze(0) 
        for set_ in vary_two_dset_set.val.grouped_features
    ]
)
log_probs = calc_set_log_probs(logits) 
expected_values = calc_expected_value(
    log_probs, 
    vary_two_dset_set.val.metadata.bin_mids.to(device),
)
labels = Tensor(
    [
        labels.unique().item() 
        for labels in vary_two_dset_set.val.grouped_labels
    ]
).to(int64)
unbinned_labels = vary_two_dset_set.val.metadata.bin_mids[labels]
plot_file_path = Path(f"plots/{vary_one_model_name}_preds_vary_two.png")
plot_predictions(
    vary_two_dset_set.val.metadata.bin_edges.cpu().detach(), 
    log_probs.cpu().detach(), 
    expected_values.cpu().detach(), 
    unbinned_labels, 
    9, 
    plot_file_path, 
)

# # vary two
logits = cat(
    [
        vary_two_model(set_).unsqueeze(0) 
        for set_ in vary_two_dset_set.val.grouped_features
    ]
)
log_probs = calc_set_log_probs(logits) 
expected_values = calc_expected_value(
    log_probs, 
    vary_two_dset_set.val.metadata.bin_mids.to(device),
)
labels = Tensor(
    [
        labels.unique().item() 
        for labels in vary_two_dset_set.val.grouped_labels
    ]
).to(int64)
unbinned_labels = vary_two_dset_set.val.metadata.bin_mids[labels]
plot_file_path = Path(f"plots/{vary_two_model_name}_preds.png")
plot_predictions(
    vary_two_dset_set.val.metadata.bin_edges.cpu().detach(), 
    log_probs.cpu().detach(), 
    expected_values.cpu().detach(), 
    unbinned_labels, 
    9, 
    plot_file_path, 
)