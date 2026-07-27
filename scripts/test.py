from torch import linspace
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR

from helpers.data_prep import prep_train_data_ref, prep_eval_data
from helpers.train import run_training_on_datasets
from helpers.model import MLP, save_model_state_dict, load_model_state_dict
from helpers.plot import (
    turn_on_hq_plots,
    turn_on_dark_plots,
    plot_to_file,
    plot_losses,
    plot_predictions_multiple_datasets,
)
from helpers.util import select_device
from helpers.evaluate import evaluate, eval_log_likelihood_ratio_sum_ensemble


def train():

    device = select_device()
    train_dataset = prep_train_data_ref(
        vary_data_path="../data/train_vary/combo.parquet",
        ref_data_path="../data/train_sm/combo.parquet",
    )
    eval_dataset = prep_train_data_ref(
        vary_data_path="../data/val_vary/combo.parquet",
        ref_data_path="../data/val_sm/combo.parquet",
    )

    train_means = train_dataset.features.mean(dim=0)
    train_stds = train_dataset.features.std(dim=0)

    for i in range(3):

        model = MLP()
        model.set_std_scale(train_means=train_means, train_stds=train_stds)
        model = model.to(device)

        loss_fn = BCEWithLogitsLoss()
        optimizer = AdamW(model.parameters(), lr=3e-4,)
        lr_scheduler = ExponentialLR(optimizer, 0.95)

        losses = run_training_on_datasets(
            train_dataset=train_dataset,
            train_batch_size=100_000,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            num_epochs=200,
            eval_dataset=eval_dataset,
            eval_batch_size=100_000,
            lr_scheduler=lr_scheduler,
        )

        save_model_state_dict(model, f"../models/model_{i}.pt")

        plot_to_file(
            f"../plots/loss_{i}.png",
            plot_losses,
            train_losses=losses["train"],
            eval_losses=losses["eval"],
            compute_log=True,
        )


def eval():

    device = select_device()

    for j in range(3):

        model = MLP()
        model.load_state_dict(load_model_state_dict(f"../models/model_{j}.pt"))
        eval_set_dataset = prep_eval_data("../data/val/combo.parquet")

        parameters = linspace(-2, 1, 100)
        list_log_probs = []
        list_true_values = []
        for i in (4, 10, 18):
            label = eval_set_dataset.labels[i]
            features = eval_set_dataset.features[i]
            log_probs = eval_log_likelihood_ratio_sum_ensemble(
                [model], features, parameters
            )
            list_log_probs.append(log_probs.numpy(force=True))
            list_true_values.append(label.numpy(force=True))

        plot_to_file(
            f"../plots/predictions_{j}.png",
            plot_predictions_multiple_datasets,
            parameters=parameters.numpy(force=True),
            log_probabilities=list_log_probs,
            true_values=list_true_values,
            colors=["#377eb8", "#ff7f00", "#4daf4a"],
        )
    
    models = [MLP(), MLP(), MLP()]
    for i, model in enumerate(models):
        model.load_state_dict(
            load_model_state_dict(f"../models/model_{i}.pt")            
        )
    eval_set_dataset = prep_eval_data("../data/val/combo.parquet")

    parameters = linspace(-2, 1, 100)
    list_log_probs = []
    list_true_values = []
    for i in (4, 10, 18):
        label = eval_set_dataset.labels[i]
        features = eval_set_dataset.features[i]
        log_probs = eval_log_likelihood_ratio_sum_ensemble(
            models, features, parameters
        )
        list_log_probs.append(log_probs.numpy(force=True))
        list_true_values.append(label.numpy(force=True))

    plot_to_file(
        f"../plots/predictions_ens.png",
        plot_predictions_multiple_datasets,
        parameters=parameters.numpy(force=True),
        log_probabilities=list_log_probs,
        true_values=list_true_values,
        colors=["#377eb8", "#ff7f00", "#4daf4a"],
    )



def main():

    turn_on_hq_plots()
    turn_on_dark_plots()
    train()
    eval()


if __name__ == "__main__":
    main()
