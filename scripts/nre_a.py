

import numpy
import torch
import matplotlib
import pandas
import sklearn

from helpers.plot import (
    color_blind_palette,
    turn_on_hq_plots, 
    turn_on_dark_plots, 
    save_fig_and_close,
    plot_loss, 
    plot_linearity, 
    plot_log_likelihood_ratio_scan
)
from helpers.train import train_val
from helpers.dataloader import EventDataLoader
from helpers.dataset import Dataset
from helpers.event_model import MLP
from helpers.util import bootstrap_tensor

turn_on_hq_plots()
turn_on_dark_plots()

model_path = "models/model.pt"


observables = ["q_sq", "cos_theta_lepton", "cos_theta_k", "chi"]
parameters = ["dC_9"]


def run_training():

    dataframe = pandas.read_parquet("data/train_vary/combo.parquet")
    dataframe = dataframe.sample(frac=1, replace=False)

    dataset = Dataset.from_pandas(
        observables=dataframe[observables], 
        parameters=dataframe[parameters]
    )
    train_dataset, val_dataset = dataset.train_val_split(val_frac=0.2)

    train_means = train_dataset.means()
    train_stdevs = train_dataset.stdevs()
    
    train_dataloader = EventDataLoader(train_dataset, batch_size=100_000, shuffle=True)
    val_dataloader = EventDataLoader(val_dataset, batch_size=100_000, shuffle=True)

    model = MLP()
    model.set_std_scale(train_means, train_stdevs)
    model = model.to('cuda')

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.95, 0.95))

    loss_fn = torch.nn.BCEWithLogitsLoss()
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    num_epochs = 100

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
    fig.savefig("plots/loss_nre_a.png", bbox_inches="tight")
    matplotlib.pyplot.close()



def run_evaluation():
    
    model = MLP()
    model.load_state_dict(torch.load("models/model.pt", weights_only=True))
    model = model.to("cuda")
    model.eval()

    val_2_sets_data_frame = pandas.read_parquet("data/val_sets/combo.parquet")

    def bootstrap_conf(features, model, num_bootstraps=10):
        
        mle, _, _, _, _ = model.mle_and_confidence_interval(-5, 2, 100, features)

        bootstrapped_mles = []
        for _ in range(num_bootstraps):
            bootstrapped_features = bootstrap_tensor(features, dim=1)
            bootstrapped_mle, _, _, _, _ = model.mle_and_confidence_interval(-5, 2, 100, bootstrapped_features)
            bootstrapped_mles.append(bootstrapped_mle)
        bootstrapped_mles = numpy.array(bootstrapped_mles)

        percentile = 0.68
        alpha = 1 - percentile

        mles_quantile_a = numpy.quantile(bootstrapped_mles, 1 - alpha/2)
        mles_quantile_b = numpy.quantile(bootstrapped_mles, alpha/2)

        a = 2 * mle - mles_quantile_a
        b = 2 * mle - mles_quantile_b

        conf_interval = numpy.array([a,b])

        return mle, conf_interval


    true_values = []
    confidence_intervals = []
    mles = []
    
    for trial_num in range(0, 100, 2):
        val_2_set_data_frame = val_2_sets_data_frame.xs(trial_num, level="trial_num")
        true_value = val_2_set_data_frame["dC_9"].unique()
        features = torch.from_numpy(val_2_set_data_frame[observables + parameters].values.copy()).unsqueeze(0).to(dtype=torch.float32, device="cuda")
        mle, conf = bootstrap_conf(features, model, num_bootstraps=100)
        true_values.append(true_value)
        mles.append(mle)
        confidence_intervals.append(conf)
        
    true_values = numpy.array(true_values)
    mles = numpy.array(mles)
    confidence_intervals = numpy.array(confidence_intervals)

    numpy.savez(
        "results/linearity.npz", 
        true_values=true_values, 
        mles=mles, 
        confidence_intervals=confidence_intervals
    )

    
    linearity_result = numpy.load("results/linearity.npz")
    fig, ax = matplotlib.pyplot.subplots()
    plot_linearity(
        ax, 
        linearity_result["mles"], 
        linearity_result["true_values"], 
        linearity_result["confidence_intervals"], 
        lim=(-5, 2)
    )
    save_fig_and_close(fig, "plots/lin.png")

    #fig, ax = matplotlib.pyplot.subplots()
    #for color, i in zip(color_blind_palette, range(9)):
    #    plot_log_likelihood_ratio_scan(
    #        ax,
    #        parameters[i],
    #        ratios[i],
    #        mle_xs[i],
    #        mle_ys[i],
    #        confidence_intervals[i],
    #        true_values[i],
    #        color
    #    )
    #save_fig_and_close(fig, "plots/ratios.png")


# run_training()
run_evaluation()