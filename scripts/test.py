from sklearn.isotonic import IsotonicRegression 
from pandas import DataFrame, cut
import numpy

from torch import linspace
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR

from helpers.data_prep import prep_train_data_ref, prep_eval_data, prep_cal_data
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
    cal_datasets = prep_cal_data(
        vary_data_path="../data/cal_vary/combo.parquet",
        ref_data_path="../data/cal_sm/combo.parquet",
    )

    train_means = train_dataset.features.mean(dim=0)
    train_stds = train_dataset.features.std(dim=0)

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
        num_epochs=1,
        eval_dataset=eval_dataset,
        eval_batch_size=100_000,
        lr_scheduler=lr_scheduler,
    )

    save_model_state_dict(model, "../models/model.pt")

    plot_to_file(
        "../plots/loss.png",
        plot_losses,
        train_losses=losses["train"],
        eval_losses=losses["eval"],
        compute_log=True,
    )

    def bin_in_var(
        dataframe,
        name_of_binning_variable, 
        start, 
        stop, 
        num_bins,
        return_bin_edges=False,
    ):
        bin_edges = numpy.linspace(start=start, stop=stop, num=num_bins+1)
        bins = cut(dataframe[name_of_binning_variable], bin_edges, include_lowest=True) # the interval each event falls into
        groupby_binned = dataframe.groupby(bins, observed=False)

        if return_bin_edges:
            return groupby_binned, bin_edges
        return groupby_binned

    def calc_bin_middles(start, stop, num_bins):
        bin_edges, step = numpy.linspace(
            start=start,
            stop=stop,
            num=num_bins+1,
            retstep=True,
        )
        bin_middles = bin_edges[:-1] + step/2 
        return bin_middles



    cal_preds = {
        param: model.probability(dset.features.to(device)) for param, dset in cal_datasets.items()
    }
    cal_labels = {
        param: dset.labels for param, dset in cal_datasets.items()
    }
    cal_pred_dicts = {
        param: {"p":cal_preds[param].squeeze().numpy(force=True), "label":cal_labels[param].squeeze().numpy(force=True)} for param in cal_preds
    }
    cal_pred_dfs = {
        param: DataFrame(dict_) for param, dict_ in cal_pred_dicts.items()
    }
    binned = {
        param: bin_in_var(df, "p", 0, 1, 10) for param, df in cal_pred_dfs.items()
    }
    def get_true_ratio(df):
        return df[df["label"]==1].count() / df.count()
    true_ratios = {
        param: binned_.apply(get_true_ratio) for param, binned_ in binned.items()
    }
    bin_mids = calc_bin_middles(0, 1, 10)
    breakpoint()
    iso_fits  = {param: IsotonicRegression().fit(bin_mids, true_ratio) for param, true_ratio in true_ratios.items()}


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
    # eval()


if __name__ == "__main__":
    main()
