
from pathlib import Path

from pandas import read_parquet
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig, subplots, close
from matplotlib.colors import Normalize
from torch import Tensor, linspace, concatenate, unsqueeze, float32
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
    make_bins,
    to_bins,
    std_scale,
    save_torch_model_state_dict,
    load_torch_model_state_dict,
)
from btokstll_sbi_tools.util.misc import (
    Interval,
    save_plot_and_close, 
    turn_on_dark_plots,
)


turn_on_dark_plots()

run_name = lambda wc: f"pred_c{wc}_vary_c{wc}_big_model"
models_dir = Path("models")
data_file_path = lambda wc, split: Path(f"data/vary_c_{wc}_{split}.parquet")
feature_names = ["q_squared", "cos_theta_mu", "cos_theta_k", "chi"]
wc_name = lambda wc: f"wc_set_d_c_{wc}"
bound_name = lambda wc, left_or_right: f"wc_dist_d_c_{wc}_{left_or_right}"
retrain = True
binned_intervals = {
    7: Interval(-1.0, 1.0),
    9: Interval(-10.0, 0.0),
}
num_bins = 30
lr = 3e-4
train_batch_size = 10_000
eval_batch_size = 10_000
epochs = Interval(0, 200)
lr_scheduler_factor = 0.95
lr_scheduler_patience = 0
lr_scheduler_treshold = 0
lr_scheduler_eps = 0
shuffle = True
num_events_eval_set = 25_000

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

for wc in (7, 9):
    model = MLP()
    device = select_device()

    columns = feature_names + [wc_name(wc)]
    train_dataframe = read_parquet(
        data_file_path(wc, "train")
    )[columns]
    eval_dataframe = read_parquet(
        data_file_path(wc, "val")
    )[columns]

    reference_train_features = torch_tensor_from_pandas(
        train_dataframe[feature_names],
        dtype="float32"
    )
    train_dataset = Dataset.from_pandas(
        features=train_dataframe[feature_names],
        labels=train_dataframe[wc_name(wc)],
        features_dtype="float32",
    )
    eval_dataset = Dataset.from_pandas(
        features=eval_dataframe[feature_names],
        labels=eval_dataframe[wc_name(wc)],
        features_dtype="float32",
    )

    bin_edges, bin_mids = make_bins(
        binned_intervals[wc], 
        num_bins,
    )
    train_dataset.labels = to_bins(
        data=train_dataset.labels, 
        bin_edges=bin_edges
    )
    eval_dataset.labels = to_bins(
        data=eval_dataset.labels, 
        bin_edges=bin_edges
    )

    reweights = calculate_reweights_uniform(
        binned_labels=train_dataset.labels,
        num_bins=num_bins
    ).to(device)

    train_dataset.features = std_scale(
        data=train_dataset.features, 
        reference=reference_train_features
    )
    eval_dataset.features = std_scale(
        data=eval_dataset.features, 
        reference=reference_train_features
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
            weight=reweights,
        ),
        lr_scheduler=ReduceLROnPlateau_Hyperparams(
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            threshold=lr_scheduler_treshold,
            eps=lr_scheduler_eps,
        ),
        num_bins=num_bins,
        binned_interval_left=binned_intervals[wc][0],
        binned_interval_right=binned_intervals[wc][1],
    )

    model_dir = models_dir.joinpath(run_name(wc))
    model_file_path = model_dir.joinpath("model.pt")

    if retrain:
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
        hyperparams_dict = asdict(hyperparams)

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
        savefig(Path("../plots/").joinpath(f"loss_{run_name(wc)}"), bbox_inches="tight")
        close()

    else:
        model.load_state_dict(
            load_torch_model_state_dict(
                model_file_path
            )
        )

    eval_sets_features = concatenate(
        [
            unsqueeze(
                to_torch_tensor(
                    trial_df.iloc[:num_events_eval_set]
                ), 
                dim=0
            )
            for _, trial_df in eval_dataframe[feature_names]
            .groupby(level="trial_num")
        ]
    )
    eval_sets_labels = to_torch_tensor(
        eval_dataframe[wc_name(wc)].groupby(
            level="trial_num"
        ).first()
    )
    eval_sets_dataset = Dataset(
        features=eval_sets_features, 
        labels=eval_sets_labels
    )
    eval_sets_dataset.features = eval_sets_dataset.features.to(
        float32
    )
    eval_sets_dataset.features = std_scale(
        data=eval_sets_dataset.features, 
        reference=reference_train_features
    )
    eval_sets_dataset.features = eval_sets_dataset.features.to(
        device
    )

    predictor = Predictor(
        model, 
        eval_sets_dataset, 
        device,
    )
    log_probs = predictor.calc_log_probs()
    expected_values = predictor.calc_expected_values(log_probs, bin_mids)


    alpha = 0.85
    fig, axs = plt.subplots(1, 2, figsize=(7,3), layout="constrained")
    fig.get_layout_engine().set(wspace=0.06)
    ax_dist = axs[0]
    ax_expected = axs[1]
    norm = Normalize(*binned_intervals[wc])
    cmap = mpl.colormaps["viridis"]
    interval = binned_intervals[wc]
    offset = abs(interval[1] - interval[0])*0.05 
    for log_p, label in zip(log_probs, eval_sets_dataset.labels):
        color = cmap(norm(label))
        plot_discrete_dist(ax_dist, bin_edges, log_p.cpu(), color=color, alpha=alpha)
    ax_expected.plot(
        [interval[0]+offset, interval[1]-offset], 
        [interval[0]+offset, interval[1]-offset], 
        color="grey", 
        zorder=-10, 
        alpha=0.5, 
        linestyle="--",
    )
    ax_expected.scatter(
        eval_sets_dataset.labels, 
        expected_values, 
        color=cmap(norm(eval_sets_dataset.labels)), 
        alpha=alpha
    )
    ax_dist.set_xticks(plot_ticks[wc])
    ax_expected.set_xticks(plot_ticks[wc])
    ax_expected.set_yticks(plot_ticks[wc])
    # cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax_dist)
    # cbar.set_ticks(range(num_bins))
    # cbar.set_label(r"Actual $\delta C_9$ Bin", fontsize=15)
    ax_dist.set_xlabel(r"$\delta C_{" f"{wc}" r"}$", fontsize=13)
    ax_dist.set_ylabel(r"$\log P(\delta C_{" f"{wc}" r"} \, | \, \textrm{dataset})$", fontsize=13)
    ax_expected.set_xlabel(r"Actual $\delta C_{" f"{wc}" r"}$", fontsize=13)
    ax_expected.set_ylabel(r"Predicted $\delta C_{" f"{wc}" r"}$", fontsize=13)
    plt.savefig(
        Path("../plots/").joinpath(f"pred_c{wc}_vary_c{wc}.png"), 
        bbox_inches="tight"
    )
    plt.close()





