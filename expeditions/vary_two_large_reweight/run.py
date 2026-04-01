
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
from btokstll_sbi_tools.util import (
    Interval,
    save_plot_and_close, 
    setup_plotting
)


# general stuff #

setup_plotting()

models_dir = Path("models")
feature_names = ["q_squared", "cos_theta_mu", "cos_theta_k", "chi"]
wc_name = lambda wc: f"wc_set_d_c_{wc}"
bound_name = lambda wc, left_or_right: f"wc_dist_d_c_{wc}_{left_or_right}"
binned_interval = Interval(-10, 0)
num_bins = 8
shuffle = True
device = select_device()


# base vary one model #

retrain = False
model_name = "pred_c9_vary_c9"
data_file_path = lambda split: Path(f"data/vary_c9_{split}.parquet")
lr = 3e-4
train_batch_size = 10_000
eval_batch_size = 10_000
epochs = Interval(0, 200)
lr_scheduler_factor = 0.95
lr_scheduler_patience = 0
lr_scheduler_treshold = 0
lr_scheduler_eps = 0

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

model = Base_MLP()

data_paths = Dataset_Set_File_Paths(
    data_file_path("train"), 
    data_file_path("val"),
)

dset_set = Dataset_Set.from_pandas_parquet_files(
    data_paths, 
    feature_names, 
    wc_name(9), 
    features_dtype="float32",
)

dset_set.apply_std_scale()

dset_set.apply_binning(
    binned_interval, 
    num_bins
)
dset_set.calc_label_reweights(
    num_labels=num_bins
)

hyperparams = Hyperparams(
    optimizer=AdamW_Hyperparams(
        lr=lr
    ),
    train_batch_size=train_batch_size,
    eval_batch_size=eval_batch_size,
    shuffle=shuffle,
    epochs=epochs,
    loss_fn=CrossEntropyLoss_Hyperparams(
        weight=dset_set.train.metadata.bin_reweights.to(device),
    ),
    lr_scheduler=ReduceLROnPlateau_Hyperparams(
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        threshold=lr_scheduler_treshold,
        eps=lr_scheduler_eps,
    ),
    num_bins=num_bins,
    binned_interval=binned_interval,
)

model_dir = models_dir.joinpath(model_name)
model_file_path = model_dir.joinpath("model.pt")

if retrain:
    model_dir.mkdir()
    loss_table = train(
        model=model, 
        hyperparams=hyperparams, 
        train_dataset=dset_set.train, 
        eval_dataset=dset_set.val, 
        device=device,
    )
    save_torch_model_state_dict(
        model=model, 
        path=model_file_path,
    )
    loss_table.save_table_as_json(model_dir.joinpath("loss.json"))
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
    savefig(Path("plots/").joinpath(f"loss_{model_name}"), bbox_inches="tight")
    close()

else:
    model.load_state_dict(
        load_torch_model_state_dict(
            model_file_path
        )
    )

dset_set.val.group_by_trial()
logits = cat(
    [
        model(set_).unsqueeze(0) 
        for set_ in dset_set.val.grouped_features
    ]
)
log_probs = calc_set_log_probs(logits) 
expected_values = calc_expected_value(
    log_probs, 
    dset_set.val.metadata.bin_mids,
)
labels = Tensor(
    [
        labels.unique().item() 
        for labels in dset_set.val.grouped_labels
    ]
).to(int64)
unbinned_labels = dset_set.val.metadata.bin_mids[labels]

plot_file_path = Path(f"plots/{model_name}_preds.png")
plot_predictions(
    dset_set.val.metadata.bin_edges.cpu().detach(), 
    log_probs.cpu().detach(), 
    expected_values.cpu().detach(), 
    unbinned_labels, 
    9, 
    plot_file_path, 
)



### l ratio ###

retrain = False
model_name = lambda bin_index: f"classify_bin_{bin_index}"
lr = 3e-4
train_batch_size = 5_000
eval_batch_size = 5_000
epochs = Interval(0, 200)
lr_scheduler_factor = 0.95
lr_scheduler_patience = 0
lr_scheduler_treshold = 0
lr_scheduler_eps = 0

vary_one_dset_paths = Dataset_Set_File_Paths(
    "data/vary_c9_train.parquet", 
    "data/vary_c9_val.parquet",
)
vary_two_dset_paths = Dataset_Set_File_Paths(
    "data/vary_c7_c9_train.parquet", 
    "data/vary_c7_c9_val.parquet",
)
vary_one_dset_set = Dataset_Set.from_pandas_parquet_files(
    vary_one_dset_paths, 
    feature_names, 
    wc_name(9), 
    features_dtype="float32"
)
vary_two_dset_set = Dataset_Set.from_pandas_parquet_files(
    vary_two_dset_paths, 
    feature_names, 
    wc_name(9), 
    features_dtype="float32"
)

for dset_set in (
    vary_two_dset_set, 
    vary_one_dset_set, 
):
    dset_set.apply_binning(
        binned_interval, 
        num_bins
    )

def mask_dataset(dset, mask):
    metadata = Dataset_Metadata(
        trials=dset.metadata.trials[mask], 
        bin_mids=dset.metadata.bin_mids, 
        bin_edges=dset.metadata.bin_edges,
        orig_labels=dset.metadata.orig_labels[mask]
    )
    return Dataset(
        metadata,
        dset.features[mask],
        dset.labels[mask],
    )

def mask_to_bin(
    dset_set:Dataset_Set, 
    bin_:int
) -> Dataset_Set:
    return Dataset_Set(
        **{
            split: (
                mask_dataset(
                    dset, 
                    dset.labels==bin_
                )
            )
            for split, dset in dset_set.__dict__.items()
        }
    )

vary_one_per_bin_dset_sets = [
    mask_to_bin(vary_one_dset_set, bin_) 
    for bin_ in range(num_bins)
]
vary_two_per_bin_dset_sets = [
    mask_to_bin(vary_two_dset_set, bin_)
    for bin_ in range(num_bins)
]

for dset_set in vary_one_per_bin_dset_sets:
    for split in dset_set:
        split.labels = zeros_like(
            split.labels, 
            dtype=int64
        )

for dset_set in vary_two_per_bin_dset_sets:
    for split in dset_set:
        split.labels = ones_like(
            split.labels, 
            dtype=int64
        )

def cat_dsets(dsets:list[Dataset]) -> Dataset:
    metadata = Dataset_Metadata( # fix this
        trials=cat([dset.metadata.trials for dset in dsets]),
        bin_mids=dsets[0].metadata.bin_mids, 
        bin_edges=dsets[0].metadata.bin_edges,
        orig_labels=cat([dset.metadata.orig_labels for dset in dsets]),
        orig_features=cat([dset.metadata.orig_features for dset in dsets]),
    )
    return Dataset(
        metadata,
        cat([dset.features for dset in dsets]),
        cat([dset.labels for dset in dsets]),
    )

def cat_dset_sets(dset_sets:list[Dataset_Set]) -> Dataset_Set:
    return Dataset_Set(
        cat_dsets([dset_set.train for dset_set in dset_sets]),
        cat_dsets([dset_set.val for dset_set in dset_sets]),
        cat_dsets([dset_set.test for dset_set in dset_sets])
    )

binary_dset_sets = [
    cat_dset_sets(
        [
            vary_one, 
            vary_two
        ]
    ) for vary_one, vary_two in zip(
        vary_one_per_bin_dset_sets,
        vary_two_per_bin_dset_sets,
    )
]


for dset_set in binary_dset_sets:
    dset_set.apply_std_scale()
    dset_set.calc_label_reweights(
        num_labels=2
    )

class Binary_MLP(Module):
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
            Linear(32, 2),
        )
    def forward(
        self, 
        x:Tensor,
    ) -> Tensor:
        logits = self.layers(x)
        return logits

if retrain:
    for bin_ in range(num_bins):
        dset_set = binary_dset_sets[bin_]
        hyperparams = Hyperparams(
            optimizer=AdamW_Hyperparams(
                lr=lr
            ),
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            shuffle=shuffle,
            epochs=epochs,
            loss_fn=CrossEntropyLoss_Hyperparams(
                weight=dset_set.train.metadata.bin_reweights.to(device),
            ),
            lr_scheduler=ReduceLROnPlateau_Hyperparams(
                factor=lr_scheduler_factor,
                patience=lr_scheduler_patience,
                threshold=lr_scheduler_treshold,
                eps=lr_scheduler_eps,
            ),
            num_bins=num_bins,
            binned_interval=binned_interval,
        )

        model_dir = models_dir.joinpath(model_name(bin_))
        model_file_path = model_dir.joinpath("model.pt")
        model = Binary_MLP()
        model_dir.mkdir()
        loss_table = train(
            model=model, 
            hyperparams=hyperparams, 
            train_dataset=dset_set.train, 
            eval_dataset=dset_set.val, 
            device=device,
        )
        save_torch_model_state_dict(
            model=model, 
            path=model_file_path,
        )
        loss_table.save_table_as_json(model_dir.joinpath("loss.json"))
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
        save_plot_and_close(f"plots/loss_{model_name(bin_)}.png")

vary_two_dset_set.val.to_device(device)
vary_two_dset_set.val.group_by_trial()

models = [Binary_MLP().to(device) for _ in range(num_bins)]
for bin_ in range(num_bins):
    models[bin_].load_state_dict(
        load_torch_model_state_dict(
            f"models/classify_bin_{bin_}/model.pt"
        )
    )

l_ratios = zeros(
    (
        len(vary_two_dset_set.val.grouped_features), 
        num_bins
    )
).to(device)
for bin_ in range(num_bins):
    logits = cat(
        [
            models[bin_](
                (
                    set_.to(device)
                    - binary_dset_sets[bin_].train.metadata.std_scale_means.to(device)
                ) / binary_dset_sets[bin_].train.metadata.std_scale_stds.to(device)
            ).unsqueeze(0) 
            for set_ in vary_two_dset_set.val.grouped_features
        ]
    )
    log_probs = calc_set_log_probs(logits) 
    l_ratios[:, bin_] = log_probs[:,1] - log_probs[:,0]


fig, ax = plt.subplots()
for r in l_ratios:
    ax.plot(r.detach().cpu())
show()


base_model = Base_MLP().to(device)
base_model.load_state_dict(
    load_torch_model_state_dict(
        f"models/pred_c9_vary_c9/model.pt"
    )
)


base_logits = cat(
    [
        base_model(
            (set_.to(device) - dset_set.train.metadata.std_scale_means.to(device))
            / dset_set.train.metadata.std_scale_stds.to(device)
        ).unsqueeze(0) 
        for set_ in vary_two_dset_set.val.grouped_features
    ]
)
base_log_probs = calc_set_log_probs(base_logits) 

base_expected_values = calc_expected_value(
    base_log_probs, 
    vary_two_dset_set.val.metadata.bin_mids.to(device)
)

labels = Tensor(
    [
        labels.unique().item() 
        for labels in vary_two_dset_set.val.grouped_labels
    ]
).to(int64)
unbinned_labels = vary_two_dset_set.val.metadata.bin_mids[labels]

fig, ax = plt.subplots()
for r in base_log_probs:
    ax.plot(r.detach().cpu())
show()

plot_file_path = Path(f"plots/base_model_preds.png")
plot_predictions(
    vary_two_dset_set.val.metadata.bin_edges.detach().cpu(), 
    base_log_probs.detach().cpu(), 
    base_expected_values.detach().cpu(), 
    unbinned_labels, 
    9, 
    plot_file_path, 
)

total = base_log_probs + l_ratios

total_p = calc_log_probs(total, dim=1)

fig, ax = plt.subplots()
for r in total_p:
    ax.plot(r.detach().cpu())
show()

expected_values = calc_expected_value(total_p, vary_two_dset_set.val.metadata.bin_mids.to(device))
plot_file_path = Path(f"plots/reweighted_preds.png")
plot_predictions(
    vary_two_dset_set.val.metadata.bin_edges.detach().cpu(), 
    total_p.detach().cpu(), 
    expected_values.detach().cpu(), 
    unbinned_labels.detach().cpu(), 
    9, 
    plot_file_path, 
)