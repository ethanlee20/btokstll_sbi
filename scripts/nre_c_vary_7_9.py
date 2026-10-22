

import numpy
import torch
from torch import log, cat, logsumexp, mean, tensor, cartesian_prod, linspace
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
from helpers.dataloader import NRE_A_EventDataLoader, NRE_C_EventDataLoader
from helpers.dataset import Dataset
from helpers.event_model import MLP
from helpers.util import bootstrap_tensor

turn_on_hq_plots()
turn_on_dark_plots()

model_path = "models/model_vary_7_9.pt"


parameter_names = ["dC_7", "dC_9"]
feature_names = ["q_sq", "cos_theta_lepton", "cos_theta_k", "chi"]

def run_training():

    train_dataframe = pandas.read_parquet("data/vary_7_9/train/combo.parquet")
    val_dataframe = pandas.read_parquet("data/vary_7_9/val/combo.parquet")

    train_dataset = Dataset.from_pandas(
        parameters=train_dataframe[parameter_names],
        features=train_dataframe[feature_names], 
    )
    val_dataset = Dataset.from_pandas(
        parameters=val_dataframe[parameter_names],
        features=val_dataframe[feature_names],
    )

    gamma = 1
    K = 10
    
    train_dataloader = NRE_C_EventDataLoader(train_dataset, batch_size=100_000, K=K, shuffle=True)
    val_dataloader = NRE_C_EventDataLoader(val_dataset, batch_size=100_000, K=K, shuffle=True)

    model = MLP(
        parameter_size=len(parameter_names), 
        feature_size=len(features_names)
    )
    model.set_std_scale(
        train_parameters=train_dataset.parameters, 
        train_features=train_dataset.features
    )
    model = model.to('cuda')

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.95, 0.95))

    class NreCLoss(torch.nn.Module):
        def __init__(self, gamma, K):
            super().__init__()
            self.gamma = tensor(gamma)
            self.K = tensor(K)

        def forward(self, logits, is_joint):
            device = logits.device
            self.gamma = self.gamma.to(device)
            self.K = self.K.to(device)
            
            extended_logits = cat(
                [
                    log(self.K).view(1,1,1).expand(logits.shape[0], -1, -1), 
                    log(self.gamma) + logits
                ],
                dim=1
            )
            denominator = logsumexp(extended_logits, dim=1)

            numerator_marginal = log(self.K)
            numerator_joint = log(self.gamma) + logits[:, K-1, :]  

            marginal = numerator_marginal - denominator
            joint = numerator_joint - denominator

            marginal_weight = self.gamma / (1 + self.gamma)
            joint_weight = 1 / (1 + self.gamma)

            loss =  -((1 - is_joint) * marginal_weight * marginal + is_joint * joint_weight * joint) 
            expected_loss = 2 * mean(loss)
            return expected_loss

    loss_fn = NreCLoss(gamma=gamma, K=K)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    num_epochs = 300

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
    fig.savefig("plots/loss_nre_c.png", bbox_inches="tight")
    matplotlib.pyplot.close()



def run_evaluation():
    
    model = MLP(
        parameter_size=len(parameter_names), 
        feature_size=len(feature_names)
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to("cuda")
    model.eval()

    val_sets_dataframe = pandas.read_parquet("data/vary_7_9/val_sets/combo.parquet")

    def bootstrap_conf(features, model, num_bootstraps=10):
        
        mle, _ = model.mle(-5, 2, 100, features)

        bootstrapped_mles = []
        for _ in range(num_bootstraps):
            bootstrapped_features = bootstrap_tensor(features, dim=1)
            bootstrapped_mle, _ = model.mle(-5, 2, 100, bootstrapped_features)
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

    
    # true_values = []
    # confidence_intervals = []
    # mles = []
    # 
    # for trial_num in range(0, 100, 2):
    #     val_2_set_data_frame = val_2_sets_data_frame.xs(trial_num, level="trial_num")
    #     true_value = val_2_set_data_frame["dC_9"].unique()
    #     features = torch.from_numpy(val_2_set_data_frame[features + parameters].values.copy()).unsqueeze(0).to(dtype=torch.float32, device="cuda")
    #     mle, conf = bootstrap_conf(features, model, num_bootstraps=100)
    #     true_values.append(true_value)
    #     mles.append(mle)
    #     confidence_intervals.append(conf)
    #     
    # true_values = numpy.array(true_values)
    # mles = numpy.array(mles)
    # confidence_intervals = numpy.array(confidence_intervals)
# 
    # numpy.savez(
    #     "results/linearity.npz", 
    #     true_values=true_values, 
    #     mles=mles, 
    #     confidence_intervals=confidence_intervals
    # )
# 
    # 
    # linearity_result = numpy.load("results/linearity.npz")
    # fig, ax = matplotlib.pyplot.subplots()
    # plot_linearity(
    #     ax, 
    #     linearity_result["mles"], 
    #     linearity_result["true_values"], 
    #     linearity_result["confidence_intervals"], 
    #     lim=(-5, 2)
    # )
    # save_fig_and_close(fig, "plots/lin.png")

    for trial_num, color in zip((0, 4, 10), color_blind_palette):
        fig, ax = matplotlib.pyplot.subplots()
        set_dataframe = val_sets_dataframe.xs(trial_num, level="trial_num")
        true_parameters = set_dataframe[parameter_names].drop_duplicates().values.copy().squeeze()
        features = torch.from_numpy(set_dataframe[feature_names].values.copy()).to(dtype=torch.float32, device="cuda")
        scan_size = 20
        parameter_scan = cartesian_prod(linspace(-0.1, 0.1, scan_size), linspace(-2, 1, scan_size)).to(dtype=torch.float32, device="cuda")
        ratios = model.scan_over(parameter_scan, features) 
        plottable_parameter_scan = parameter_scan.reshape(scan_size, scan_size, -1).numpy(force=True)
        plottable_ratios = ratios.reshape(scan_size, scan_size).numpy(force=True)
        ax.pcolormesh(plottable_parameter_scan[:,:,0], plottable_parameter_scan[:, :, 1], plottable_ratios)
        ax.scatter(*true_parameters)
        save_fig_and_close(fig, f"plots/ratios_vary_7_9_{trial_num}.png")


# run_training()
run_evaluation()