from torch import linspace
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW

from helpers.data_prep import prep_train_data, prep_eval_data
from helpers.train import run_training_on_datasets
from helpers.model import MLP
from helpers.plot import (
    turn_on_hq_plots,
    turn_on_dark_plots,
    plot_to_file,
    plot_losses,
    plot_predictions_multiple_datasets,
)
from helpers.util import select_device
from helpers.evaluate import evaluate


def main():

    turn_on_hq_plots()
    turn_on_dark_plots()

    device = select_device()

    train_dataset = prep_train_data("data/train/combo.parquet")
    eval_dataset = prep_train_data("data/val/combo.parquet")

    train_means = train_dataset.features.mean(dim=0)
    train_stds = train_dataset.features.std(dim=0)

    model = MLP()
    model.set_std_scale(train_means=train_means, train_stds=train_stds)
    model = model.to(device)

    loss_fn = BCEWithLogitsLoss()

    optimizer = AdamW(model.parameters(), lr=3e-4)

    losses = run_training_on_datasets(
        train_dataset=train_dataset,
        train_batch_size=5_000,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_epochs=30,
        eval_dataset=eval_dataset,
        eval_batch_size=5_000,
    )

    plot_to_file(
        "plots/loss.png",
        plot_losses,
        train_losses=losses["train"],
        eval_losses=losses["eval"],
        compute_log=True,
    )

    eval_set_dataset = prep_eval_data("data/val_small.parquet")

    parameters = linspace(-10, 0, 100)
    list_log_probs = []
    list_true_values = []

    for i in (0, 5, 9):

        label = eval_set_dataset.labels[i]
        features = eval_set_dataset.features[i]

        log_probs = evaluate(model, features, parameters)

        list_log_probs.append(log_probs.numpy(force=True))
        list_true_values.append(label.numpy(force=True))

    plot_to_file(
        "plots/predictions.png",
        plot_predictions_multiple_datasets,
        parameters=parameters.numpy(force=True),
        log_probabilities=list_log_probs,
        true_values=list_true_values,
        colors=["#377eb8", "#ff7f00", "#4daf4a"],
    )


if __name__ == "__main__":
    main()
