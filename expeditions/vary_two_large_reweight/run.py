
from pathlib import Path
from dataclasses import asdict

from pandas import read_parquet
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig, subplots, close, plot, show
from matplotlib.colors import Normalize
from torch import Tensor, linspace, cat, unsqueeze, float32, ones, zeros, int64
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
dset_set.calc_bin_reweights()

hyperparams = Hyperparams(
    optimizer=AdamW_Hyperparams(
        lr=lr
    ),
    train_batch_size=train_batch_size,
    eval_batch_size=eval_batch_size,
    shuffle=shuffle,
    epochs=epochs,
    loss_fn=CrossEntropyLoss_Hyperparams(
        weight=dset_set.train.bin_reweights.to(device),
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
    dset_set.val.bin_mids,
)
labels = Tensor(
    [
        labels.unique().item() 
        for labels in dset_set.val.grouped_labels
    ]
).to(int64)
unbinned_labels = dset_set.val.bin_mids[labels]

plot_file_path = Path(f"plots/{model_name}_preds.png")
plot_predictions(
    dset_set.val.bin_edges.cpu().detach(), 
    log_probs.cpu().detach(), 
    expected_values.cpu().detach(), 
    unbinned_labels, 
    9, 
    plot_file_path, 
)



### l ratio ###

retrain=True
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

breakpoint()

mask_dataset = lambda dset, mask: Dataset(
    **{
        key: array[mask] 
        for key, array in asdict(dset).items()
    }
)
mask_to_bin = lambda dset_set, bin_: Dataset_Set(
    **{
        split: mask_dataset(dset, dset.labels==bin_)
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

# breakpoint()
cat_dsets = lambda dsets: Dataset(
    cat([dset.features for dset in dsets]),
    cat([dset.labels for dset in dsets]),
    cat([dset.trials for dset in dsets]),
)
cat_dset_sets = lambda dset_sets: Dataset_Set(
    cat_dsets([dset_set.train for dset_set in dset_sets]),
    cat_dsets([dset_set.val for dset_set in dset_sets]),
    cat_dsets([dset_set.test for dset_set in dset_sets])
)
binary_dset_sets = []
for vary_one, vary_two in zip(
    vary_one_per_bin_dset_sets,
    vary_two_per_bin_dset_sets, 
):
    binary_dset_sets.append(
        cat_dset_sets(
            [
                vary_one, 
                vary_two,
            ]
        )
    )

for dset_set in binary_dset_sets:
    dset_set.apply_std_scale()
    dset_set.calc_bin_reweights()

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
    for bin_index in range(num_bins):
        dset_set = binary_dset_sets[bin_index]
        hyperparams = Hyperparams(
            optimizer=AdamW_Hyperparams(
                lr=lr
            ),
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            shuffle=shuffle,
            epochs=epochs,
            loss_fn=CrossEntropyLoss_Hyperparams(
                weight=dset_set.train.bin_reweights,
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

        model_dir = models_dir.joinpath(model_name(bin_index))
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
        save_plot_and_close(f"plots/loss_{model_name(bin_index)}.png")

        

# vary_two_eval_sets_features = cat(
#     [
#         unsqueeze(
#             torch_tensor_from_pandas(
#                 trial_df.iloc[:num_events_eval_set]
#             ), 
#             dim=0
#         )
#         for _, trial_df in vary_two_eval_dataframe[feature_names]
#         .groupby(level="trial_num")
#     ]
# )
# vary_two_eval_sets_labels = torch_tensor_from_pandas(
#     vary_two_eval_dataframe[wc_name(9)].groupby(
#         level="trial_num"
#     ).first()
# )
# vary_two_eval_sets_dataset = Dataset(
#     features=vary_two_eval_sets_features, 
#     labels=vary_two_eval_sets_labels
# )
# vary_two_eval_sets_dataset.features = vary_two_eval_sets_dataset.features.to(
#     float32
# )
# vary_two_eval_sets_dataset.features = std_scale(
#     data=vary_two_eval_sets_dataset.features, 
#     reference=reference_train_features
# )
# vary_two_eval_sets_dataset.features = vary_two_eval_sets_dataset.features.to(
#     device
# )

# models = [Binary_MLP() for _ in range(num_bins)]
# for bin_index in range(num_bins):
#     models[bin_index].load_state_dict(
#         load_torch_model_state_dict(
#             f"models/classify_bin_{bin_index}/model.pt"
#         )
#     )


# l_ratios = zeros((20, num_bins)).to(device)
# for bin_index in range(num_bins):
#     predictor = Predictor(
#         models[bin_index], 
#         vary_two_eval_sets_dataset, 
#         device,
#     )
#     log_probs = predictor.calc_log_probs()
#     l_ratios[:, bin_index] = log_probs[:,0] - log_probs[:,1]



# fig, ax = plt.subplots()
# for r in l_ratios:
#     ax.plot(r.cpu())
# show()


# base_model = Base_MLP()
# base_model.load_state_dict(
#     load_torch_model_state_dict(
#         f"models/pred_c9_vary_c9/model.pt"
#     )
# )

# predictor = Predictor(base_model, vary_two_eval_sets_dataset, device)
# base_log_probs = predictor.calc_log_probs()
# base_expected_values = predictor.calc_expected_values(base_log_probs, bin_mids)

# fig, ax = plt.subplots()
# for r in base_log_probs:
#     ax.plot(r.cpu())
# show()

# plot_file_path = Path(f"plots/{model_name(1)}_preds.png")
# plot_predictions(
#     bin_edges.cpu(), 
#     base_log_probs.cpu(), 
#     base_expected_values.cpu(), 
#     vary_two_eval_sets_dataset.labels.cpu(), 
#     9, 
#     plot_file_path, 
# )

# total = base_log_probs + l_ratios

# total_p = calc_log_probs(total, dim=1)

# fig, ax = plt.subplots()
# for r in total_p:
#     ax.plot(r.cpu())
# show()

# expected_values = predictor.calc_expected_values(total_p, bin_mids)

# plot_file_path = Path(f"plots/{model_name(0)}_preds.png")
# plot_predictions(
#     bin_edges.cpu(), 
#     total_p.cpu(), 
#     expected_values.cpu(), 
#     vary_two_eval_sets_dataset.labels.cpu(), 
#     9, 
#     plot_file_path, 
# )