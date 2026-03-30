
from pathlib import Path

from pandas import read_parquet
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig, subplots, close, plot, show
from matplotlib.colors import Normalize
from torch import Tensor, linspace, concatenate, unsqueeze, float32, ones, zeros, int64
from torch.nn import Module, Sequential, Linear, ReLU

from btokstll_sbi_tools.model import (
    train,
    calculate_reweights_uniform,
    AdamW_Hyperparams,
    CrossEntropyLoss_Hyperparams,
    ReduceLROnPlateau_Hyperparams,
    Hyperparams,
    Predictor, 
    plot_predictions,
    select_device,
    torch_tensor_from_pandas,
    Dataset,
    Dataset_Set,
    Dataset_Set_File_Paths,
    make_bins,
    to_bins,
    std_scale,
    save_torch_model_state_dict,
    load_torch_model_state_dict,
    calc_log_probs
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
num_events_eval_set = 25_000
device = select_device()


# base vary one model #

retrain = True
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


predictor = Predictor(
    model, 
    dset_set.val.features, 
    device,
)
log_probs = predictor.calc_log_probs()
expected_values = predictor.calc_expected_values(log_probs, bin_mids)


plot_file_path = Path(f"plots/{model_name}_preds.png")
plot_predictions(
    bin_edges.cpu(), 
    log_probs.cpu(), 
    expected_values.cpu(), 
    eval_sets_dataset.labels.cpu(), 
    9, 
    plot_file_path, 
)



### l ratio ###



columns = feature_names + [wc_name(9)]
vary_two_train_dataframe = read_parquet(
    "data/vary_c7_c9_train.parquet"
)[columns]
vary_two_eval_dataframe = read_parquet(
    "data/vary_c7_c9_val.parquet"
)[columns]
vary_one_train_dataframe = read_parquet(
    "data/vary_c9_train.parquet"
)[columns]
vary_one_eval_dataframe = read_parquet(
    "data/vary_c9_val.parquet"
)[columns]

vary_two_train_dataset = Dataset.from_pandas(
    features=vary_two_train_dataframe[feature_names],
    labels=vary_two_train_dataframe[wc_name(9)],
    features_dtype="float32",
)
vary_two_eval_dataset = Dataset.from_pandas(
    features=vary_two_eval_dataframe[feature_names],
    labels=vary_two_eval_dataframe[wc_name(9)],
    features_dtype="float32",
)
vary_one_train_dataset = Dataset.from_pandas(
    features=vary_one_train_dataframe[feature_names],
    labels=vary_one_train_dataframe[wc_name(9)],
    features_dtype="float32",
)
vary_one_eval_dataset = Dataset.from_pandas(
    features=vary_one_eval_dataframe[feature_names],
    labels=vary_one_eval_dataframe[wc_name(9)],
    features_dtype="float32",
)

for dset in (
    vary_two_train_dataset, 
    vary_two_eval_dataset, 
    vary_one_train_dataset, 
    vary_one_eval_dataset,
):
    dset.features = std_scale(
        data=dset.features,
        reference=reference_train_features
    )

vary_two_train_dataset.labels = to_bins(
    data=vary_two_train_dataset.labels, 
    bin_edges=bin_edges
)
vary_two_eval_dataset.labels = to_bins(
    data=vary_two_eval_dataset.labels, 
    bin_edges=bin_edges
)
vary_one_train_dataset.labels = to_bins(
    data=vary_one_train_dataset.labels, 
    bin_edges=bin_edges
)
vary_one_eval_dataset.labels = to_bins(
    data=vary_one_eval_dataset.labels, 
    bin_edges=bin_edges
)

vary_two_binned_train_datasets = []
for bin_id in range(num_bins):
    mask = vary_two_train_dataset.labels == bin_id
    dset = Dataset(
        vary_two_train_dataset.features[mask],
        vary_two_train_dataset.labels[mask],
    )
    vary_two_binned_train_datasets.append(dset)
vary_two_binned_eval_datasets = []
for bin_id in range(num_bins):
    mask = vary_two_eval_dataset.labels == bin_id
    dset = Dataset(
        vary_two_eval_dataset.features[mask],
        vary_two_eval_dataset.labels[mask],
    )
    vary_two_binned_eval_datasets.append(dset)
vary_one_binned_train_datasets = []
for bin_id in range(num_bins):
    mask = vary_one_train_dataset.labels == bin_id
    dset = Dataset(
        vary_one_train_dataset.features[mask],
        vary_one_train_dataset.labels[mask],
    )
    vary_one_binned_train_datasets.append(dset)
vary_one_binned_eval_datasets = []
for bin_id in range(num_bins):
    mask = vary_one_eval_dataset.labels == bin_id
    dset = Dataset(
        vary_one_eval_dataset.features[mask],
        vary_one_eval_dataset.labels[mask],
    )
    vary_one_binned_eval_datasets.append(dset)




binary_train_datasets = []
for vary_two, vary_one in zip(
    vary_two_binned_train_datasets, 
    vary_one_binned_train_datasets
):
    feat = concatenate([vary_two.features, vary_one.features])
    lab = concatenate(
        [
            zeros(len(vary_two.features), dtype=int64), 
            ones(len(vary_one.features), dtype=int64),
        ]
    )
    dset = Dataset(feat, lab)
    binary_train_datasets.append(dset)


binary_eval_datasets = []
for vary_two, vary_one in zip(
    vary_two_binned_eval_datasets, 
    vary_one_binned_eval_datasets
):
    feat = concatenate([vary_two.features, vary_one.features])
    lab = concatenate(
        [
            zeros(len(vary_two.features),dtype=int64), 
            ones(len(vary_one.features), dtype=int64),
        ]
    )
    dset = Dataset(feat, lab)
    binary_eval_datasets.append(dset)


binary_reweights = [
    calculate_reweights_uniform(dset.labels, num_bins=2).to(device)
    for dset in binary_train_datasets
]

retrain=True
model_name = lambda bin_index: f"classify_bin_{bin_index}"
data_file_path = lambda split: Path(f"data/vary_c9_{split}.parquet")
lr = 3e-4
train_batch_size = 5_000
eval_batch_size = 5_000
epochs = Interval(0, 200)
lr_scheduler_factor = 0.95
lr_scheduler_patience = 0
lr_scheduler_treshold = 0
lr_scheduler_eps = 0

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

        hyperparams = Hyperparams(
            optimizer=AdamW_Hyperparams(
                lr=lr
            ),
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            shuffle=shuffle,
            epochs=epochs,
            loss_fn=CrossEntropyLoss_Hyperparams(
                weight=binary_reweights[bin_index],
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

        train_dataset = binary_train_datasets[bin_index]
        eval_dataset = binary_eval_datasets[bin_index]

        model_dir = models_dir.joinpath(model_name(bin_index))
        model_file_path = model_dir.joinpath("model.pt")

        model = Binary_MLP()
    
        model_dir.mkdir()
        loss_table = train(
            model=model, 
            hyperparams=hyperparams, 
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset, 
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
        savefig(Path("plots/").joinpath(f"loss_{model_name(bin_index)}"), bbox_inches="tight")
        close()

        







vary_two_eval_sets_features = concatenate(
    [
        unsqueeze(
            torch_tensor_from_pandas(
                trial_df.iloc[:num_events_eval_set]
            ), 
            dim=0
        )
        for _, trial_df in vary_two_eval_dataframe[feature_names]
        .groupby(level="trial_num")
    ]
)
vary_two_eval_sets_labels = torch_tensor_from_pandas(
    vary_two_eval_dataframe[wc_name(9)].groupby(
        level="trial_num"
    ).first()
)
vary_two_eval_sets_dataset = Dataset(
    features=vary_two_eval_sets_features, 
    labels=vary_two_eval_sets_labels
)
vary_two_eval_sets_dataset.features = vary_two_eval_sets_dataset.features.to(
    float32
)
vary_two_eval_sets_dataset.features = std_scale(
    data=vary_two_eval_sets_dataset.features, 
    reference=reference_train_features
)
vary_two_eval_sets_dataset.features = vary_two_eval_sets_dataset.features.to(
    device
)

models = [Binary_MLP() for _ in range(num_bins)]
for bin_index in range(num_bins):
    models[bin_index].load_state_dict(
        load_torch_model_state_dict(
            f"models/classify_bin_{bin_index}/model.pt"
        )
    )


l_ratios = zeros((20, num_bins)).to(device)
for bin_index in range(num_bins):
    predictor = Predictor(
        models[bin_index], 
        vary_two_eval_sets_dataset, 
        device,
    )
    log_probs = predictor.calc_log_probs()
    l_ratios[:, bin_index] = log_probs[:,0] - log_probs[:,1]



fig, ax = plt.subplots()
for r in l_ratios:
    ax.plot(r.cpu())
show()


base_model = Base_MLP()
base_model.load_state_dict(
    load_torch_model_state_dict(
        f"models/pred_c9_vary_c9/model.pt"
    )
)

predictor = Predictor(base_model, vary_two_eval_sets_dataset, device)
base_log_probs = predictor.calc_log_probs()
base_expected_values = predictor.calc_expected_values(base_log_probs, bin_mids)

fig, ax = plt.subplots()
for r in base_log_probs:
    ax.plot(r.cpu())
show()

plot_file_path = Path(f"plots/{model_name(1)}_preds.png")
plot_predictions(
    bin_edges.cpu(), 
    base_log_probs.cpu(), 
    base_expected_values.cpu(), 
    vary_two_eval_sets_dataset.labels.cpu(), 
    9, 
    plot_file_path, 
)

total = base_log_probs + l_ratios

total_p = calc_log_probs(total, dim=1)

fig, ax = plt.subplots()
for r in total_p:
    ax.plot(r.cpu())
show()

expected_values = predictor.calc_expected_values(total_p, bin_mids)

plot_file_path = Path(f"plots/{model_name(0)}_preds.png")
plot_predictions(
    bin_edges.cpu(), 
    total_p.cpu(), 
    expected_values.cpu(), 
    vary_two_eval_sets_dataset.labels.cpu(), 
    9, 
    plot_file_path, 
)