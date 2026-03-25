
from pathlib import Path

from pandas import read_parquet
import matplotlib.pyplot as plt
from matplotlib.pyplot import close, savefig, subplots
from torch import Tensor, linspace, concatenate, unsqueeze, float32
from torch.nn import Module, Sequential, Linear, ReLU, Dropout

from btokstll_sbi_tools.train import (
    train,
    calculate_reweights_uniform,
    Adam_Hyperparams,
    AdamW_Hyperparams,
    CrossEntropyLoss_Hyperparams,
    ReduceLROnPlateau_Hyperparams,
    Hyperparams,
    Loss_Table
)
from btokstll_sbi_tools.eval import Predictor, plot_predictions
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

plt.style.use("dark_background")
plt.rcParams.update({
    "figure.dpi": 400, 
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern",

})


### Config
retrain = True
run_name = lambda wc: f"2026-03-18_vary_c7_c9_pred_c{wc}_cond_v15"
models_dir = Path("../models")
data_file_path = lambda split: Path(f"../data/vary_c7_c9_{split}.parquet")
feature_names = ["q_squared", "cos_theta_mu", "cos_theta_k", "chi"]
label_name = lambda wc: f"wc_set_d_c_{wc}"
bound_label = lambda wc, left_or_right: f"wc_dist_d_c_{wc}_{left_or_right}"
binned_intervals = {
    7: (-1, 1),
    9: (-10, 0),
}
num_bins = 10
lr = 3e-4
train_batch_size = 10_000
eval_batch_size = 10_000
epochs = range(0, 600)
lr_scheduler_factor = 0.9999
lr_scheduler_patience = 0
lr_scheduler_treshold = 0
lr_scheduler_eps = 0
shuffle = True
num_events_eval_set = 25_000
weight_decay = 0.5

plot_ticks = {
    7: [-1, 0, 1],
    9: [-10, -5, 0],
}
###

class MLP(Module):

    def __init__(
        self,
    ):
        super().__init__()  
        self.layers = Sequential(
            Linear(5, 32),
            ReLU(),
            Linear(32, 64),
            ReLU(),
            Linear(64, 64),
            ReLU(),
        )
        self.predict_c9 = Linear(64, num_bins)
        self.predict_c7 = Linear(64, num_bins)

    def forward(
        self, 
        x:Tensor,
    ) -> Tensor:
        logits = self.layers(x)
        c9_prediction = self.predict_c9(logits)
        c7_prediction = self.predict_c7(logits)
        return c7_prediction, c9_prediction


for pred_wc in (7, 9):

    model = MLP()

    device = select_device()

    cond_wc = 7 if pred_wc == 9 else 9 if pred_wc == 7 else None

    cond_label = label_name(cond_wc)
    label = label_name(pred_wc)

    columns = feature_names + [label, cond_label]

    train_dataframe = read_parquet(data_file_path("train"))[columns]
    reference_train_features = to_torch_tensor(train_dataframe[feature_names + [cond_label]]).to(float32) ###
    train_dataset = Dataset.from_pandas(
        features=train_dataframe[feature_names + [cond_label]],
        labels=train_dataframe[[label, cond_label]]
    )
    eval_dataframe = read_parquet(data_file_path("val"))[columns]
    eval_dataset = Dataset.from_pandas(
        features=eval_dataframe[feature_names + [cond_label]],
        labels=eval_dataframe[[label, cond_label]]
    )

    train_dataset.features = train_dataset.features.to(
        float32
    )
    eval_dataset.features = eval_dataset.features.to(
        float32
    )

    bin_edges = linspace(*binned_intervals[pred_wc], num_bins+1)
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
        optimizer=AdamW_Hyperparams(
            lr=lr,
            weight_decay=weight_decay,
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
        binned_interval_left=binned_intervals[pred_wc][0],
        binned_interval_right=binned_intervals[pred_wc][1],
    )

    model_dir = models_dir.joinpath(run_name(pred_wc))
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

        # plot loss
        colors = {"train": "goldenrod", "eval": "skyblue"}
        loss_dict = loss_table.as_lists()
        loss_dict["epochs"] = [int(ep) for ep in loss_dict["epochs"]]
        fig, ax = subplots(figsize=(5,4))
        for split in ("train", "eval"):
            ax.scatter(loss_dict["epochs"], loss_dict[split], label=split, s=1, color=colors[split])
        ax.set_ylabel("Cross Entropy Loss", fontsize=13)
        ax.set_xlabel("Epoch", fontsize=13)
        ax.legend(fontsize=13, markerscale=5)
        savefig(f"../plots/{run_name(pred_wc)}_loss.png", bbox_inches="tight")
        close()

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
            for _, trial_df in eval_dataframe[feature_names + [cond_label]]
            .groupby(level="trial_num")
        ]
    )
    eval_sets_labels = to_torch_tensor(
        eval_dataframe[label_name(pred_wc)].groupby(
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




    plot_file_path = Path(f"../plots/{run_name(pred_wc)}_preds.png")
    plot_predictions(
        bin_edges.cpu(), 
        log_probs.cpu(), 
        expected_values.cpu(), 
        eval_sets_dataset.labels.cpu(), 
        pred_wc, 
        plot_file_path, 
        cond_wc_index=cond_wc,
    )






