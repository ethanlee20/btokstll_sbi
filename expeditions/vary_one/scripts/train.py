
from pathlib import Path
from dataclasses import asdict
from json import dump

from pandas import read_parquet
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm, Colormap, Normalize
from matplotlib.cm import ScalarMappable
from torch import Tensor, linspace, concatenate, unsqueeze, float32
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
from btokstll_sbi_tools.eval import Predictor, plot_discrete_distribution
from btokstll_sbi_tools.util import (
    to_torch_tensor,
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
names = {
    "c_7": "2026-03-04_vary_c_7_long",
    "c_9": "2026-03-04_vary_c_9_long",
    "c_10": "2026-03-04_vary_c_10_long",
}
subscripts = {"c_7": 7, "c_9": 9, "c_10": 10}
main_models_dir = Path("../models")
train_file_paths = {
    "c_7": Path("../data/combined_vary_c_7_train.parquet"),
    "c_9": Path("../data/combined_vary_c_9_train.parquet"),
    "c_10": Path("../data/combined_vary_c_10_train.parquet"),
}
eval_file_paths = {
    "c_7": Path("../data/combined_vary_c_7_val.parquet"),
    "c_9": Path("../data/combined_vary_c_9_val.parquet"),
    "c_10": Path("../data/combined_vary_c_10_val.parquet"),
}
feature_names = ["q_squared", "cos_theta_mu", "cos_theta_k", "chi"]
label_names = {
    "c_7": "delta_c_7", 
    "c_9": "delta_c_9", 
    "c_10": "delta_c_10"
} 
binned_intervals = {
    "c_7": (-0.2, 0.2),
    "c_9": (-2.0, 1.0),
    "c_10": (-1.0, 1.0),
}
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
num_events_eval_set = 50_000

plot_ticks = {
    "c_7": [-0.2, -0.1, 0, 0.1, 0.2],
    "c_9": [-2, -1, 0, 1],
    "c_10": [-1, -0.5, 0, 0.5, 1],
}
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


for wc in ["c_7", "c_9", "c_10"]:

    model = MLP()

    device = select_device()

    train_dataframe = read_parquet(train_file_paths[wc])[feature_names]
    train_dataframe[label_names[wc]] = train_dataframe.index.get_level_values(level=label_names[wc])
    reference_train_features = to_torch_tensor(train_dataframe[feature_names]).to(float32) ###
    train_dataset = Dataset.from_pandas(
        features=train_dataframe[feature_names],
        labels=train_dataframe[label_names[wc]]
    )
    eval_dataframe = read_parquet(eval_file_paths[wc])[feature_names]
    eval_dataframe[label_names[wc]] = eval_dataframe.index.get_level_values(level=label_names[wc])
    eval_dataset = Dataset.from_pandas(
        features=eval_dataframe[feature_names],
        labels=eval_dataframe[label_names[wc]]
    )

    train_dataset.features = train_dataset.features.to(
        float32
    )
    eval_dataset.features = eval_dataset.features.to(
        float32
    )

    bin_edges = linspace(*binned_intervals[wc], num_bins+1)
    bin_mids = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
    train_dataset.labels = bin_(
        data=train_dataset.labels, 
        bin_edges=bin_edges
    )
    eval_dataset.labels = bin_(
        data=eval_dataset.labels, 
        bin_edges=bin_edges
    )

    reweights = calculate_reweights_uniform(
        binned_labels=train_dataset.labels,
        num_bins=num_bins
    )
    reweights = reweights.to(device)

    train_dataset.features = std_scale(
        data=train_dataset.features, 
        reference=reference_train_features
    )
    eval_dataset.features = std_scale(
        data=eval_dataset.features, 
        reference=reference_train_features
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
        binned_interval_left=binned_intervals[wc][0],
        binned_interval_right=binned_intervals[wc][1],
    )

    model_dir = main_models_dir.joinpath(names[wc])
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
            path=model_file_path
        )
        loss_table.save_table_as_json(model_dir.joinpath("loss.json"))
        # hyperparams_dict = asdict(hyperparams)
        # with open(model_dir.joinpath("hyperparams.json"), 'x') as f:
        #     dump(hyperparams_dict, f)
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
        eval_dataframe[label_names[wc]].groupby(
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
    norm = CenteredNorm(vcenter=0, halfrange=abs(max(binned_intervals[wc])))
    cmap = mpl.colormaps["coolwarm"]
    interval = binned_intervals[wc]
    offset = (interval[1] - interval[0])*0.05 
    for log_p, label in zip(log_probs, eval_sets_dataset.labels):
        color = cmap(norm(label))
        plot_discrete_distribution(ax_dist, bin_edges, log_p.cpu(), color=color, alpha=alpha)
        # ax_dist.plot(bin_mids, log_p.cpu(), color=color, alpha=alpha)
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
    ax_dist.set_xlabel(r"$\delta C_{" f"{subscripts[wc]}" r"}$", fontsize=13)
    ax_dist.set_ylabel(r"$\log P(\delta C_{" f"{subscripts[wc]}" r"} \, | \, \textrm{dataset})$", fontsize=13)
    ax_expected.set_xlabel(r"Actual $\delta C_{" f"{subscripts[wc]}" r"}$", fontsize=13)
    ax_expected.set_ylabel(r"Predicted $\delta C_{" f"{subscripts[wc]}" r"}$", fontsize=13)
    plt.savefig(
        model_dir.joinpath(f"preds.png"), 
        bbox_inches="tight"
    )
    plt.close()





