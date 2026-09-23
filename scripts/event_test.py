

import numpy
import torch
import matplotlib
import pandas
import sklearn

from helpers.plot import (
    turn_on_hq_plots, 
    turn_on_dark_plots, 
    save_fig_and_close,
    plot_loss, 
    plot_linearity, 
    plot_log_likelihood_ratio_scan
)
from helpers.train import train_val
from helpers.dataloader import SetDataLoader
from helpers.set_model import MLP


turn_on_hq_plots()
turn_on_dark_plots()

model_path = "models/model.pt"

feature_columns = ["q_sq", "cos_theta_lepton", "cos_theta_k", "chi", "dC_9"]

def run_training():

    reference_data_data_frame = pandas.read_parquet("data/train_sm/combo.parquet")[feature_columns]
    parameterized_data_data_frame = pandas.read_parquet("data/train_vary/combo.parquet")[feature_columns]

    grouped_parameterized_data_data_frame = parameterized_data_data_frame.groupby(level="trial_num")
    parameterized_data_as_sets_ndarray = numpy.stack(
        [trial.values for _, trial in grouped_parameterized_data_data_frame]
    )

    reference_data_as_sets_ndarray = reference_data_data_frame.values.reshape(
        parameterized_data_as_sets_ndarray.shape
    )

    reference_data_as_sets_ndarray[:, :, 4] = parameterized_data_as_sets_ndarray[:, :, 4]

    reference_data_as_sets_labels_ndarray = numpy.zeros(len(reference_data_as_sets_ndarray))
    parameterized_data_as_sets_labels_ndarray = numpy.ones(len(parameterized_data_as_sets_ndarray))

    sets_ndarray = numpy.vstack([reference_data_as_sets_ndarray, parameterized_data_as_sets_ndarray])
    set_labels_ndarray = numpy.append(reference_data_as_sets_labels_ndarray, parameterized_data_as_sets_labels_ndarray)

    sets_ndarray = sets_ndarray.astype(numpy.float32)
    set_labels_ndarray = set_labels_ndarray.astype(numpy.float32)

    cross_validation_splitter = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    train_indices, val_indices = next(cross_validation_splitter.split(sets_ndarray, set_labels_ndarray))

    sets_train_ndarray = sets_ndarray[train_indices]
    set_labels_train_ndarray = set_labels_ndarray[train_indices]
    sets_val_ndarray = sets_ndarray[val_indices]
    set_labels_val_ndarray = set_labels_ndarray[val_indices]

    sets_train_tensor = torch.from_numpy(sets_train_ndarray)
    set_labels_train_tensor = torch.from_numpy(set_labels_train_ndarray)
    sets_val_tensor = torch.from_numpy(sets_val_ndarray)
    set_labels_val_tensor = torch.from_numpy(set_labels_val_ndarray)

    train_dataloader = SetDataLoader(sets_train_tensor, set_labels_train_tensor, batch_size=32)
    val_dataloader = SetDataLoader(sets_val_tensor, set_labels_val_tensor, batch_size=32)

    train_means = sets_train_tensor.mean(axis=(0,1))
    train_stds = sets_train_tensor.std(axis=(0,1))

    model = MLP()
    model.set_std_scale(train_means, train_stds)
    model = model.to('cuda')

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.95, 0.95))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    num_epochs = 500

    losses = train_val(
        train_data_loader=train_dataloader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_epochs=num_epochs,
        val_data_loader=val_dataloader,
        lr_scheduler=lr_scheduler
    )

    torch.save(model.state_dict(), model_path)

    fig, ax = matplotlib.pyplot.subplots()
    plot_loss(
        ax, 
        train_losses=losses["train"],
        val_losses=losses["val"],
        compute_log=False
    )
    ax.set_ylim(-1, 3)
    fig.savefig("plots/loss.png", bbox_inches="tight")
    matplotlib.pyplot.close()



def run_evaluation():

    model = MLP()
    model.load_state_dict(torch.load("models/model.pt", weights_only=True))
    model = model.to("cuda")
    model.eval()

    val_2_sets_data_frame = pandas.read_parquet("data/train_sm/combo.parquet")

    true_values = []
    mle_xs = []
    mle_ys = []
    confidence_intervals = []

    for trial_num in range(100):
        val_2_set_data_frame = val_2_sets_data_frame.sample(16_000, replace=False) # val_2_sets_data_frame.xs(trial_num, level="trial_num")
        true_value = val_2_set_data_frame["dC_9"].unique()
        features = torch.from_numpy(val_2_set_data_frame[feature_columns].values.copy()).unsqueeze(0).to(dtype=torch.float32, device="cuda")
        mle_x, mle_y, confidence_interval = model.mle_and_confidence_interval(-5, 2, 100, features)
        true_values.append(true_value)
        mle_xs.append(mle_x)
        mle_ys.append(mle_y)
        confidence_intervals.append(confidence_interval)
        
    true_values = numpy.array(true_values)
    mle_xs = numpy.array(mle_xs)
    mle_ys = numpy.array(mle_ys)
    confidence_intervals = numpy.array(confidence_intervals)


    fig, ax = matplotlib.pyplot.subplots()
    plot_linearity(ax, mle_xs, true_values, confidence_intervals, lim=(-5, 2))
    save_fig_and_close(fig, "plots/lin.png")


# run_training()
run_evaluation()