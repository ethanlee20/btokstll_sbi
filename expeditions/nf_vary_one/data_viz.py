
from pathlib import Path
from itertools import product

from matplotlib.pyplot import subplots, close
from pandas import read_parquet

from btokstll_sbi_tools.plot import (
    turn_on_dark_plots, 
    turn_on_hq_plots,
)


# setup paths

dir_ = Path("plots/hist")
dir_.mkdir(exist_ok=True)


# setup plotting

turn_on_dark_plots()
turn_on_hq_plots()


# load data

df = read_parquet("data/vary_dc9_train.parquet")

feature_names = ["q_squared", "cos_theta_mu", "cos_theta_k", "chi"]
label_name = "delta_wc_values_dc9"

fancy_names = {
    "q_squared": r"$q^2$ [GeV$^2$]",
    "cos_theta_mu": r"$\cos\theta_\mu$", 
    "cos_theta_k": r"$\cos\theta_K$",
    "chi": r"$\chi$", 
    "delta_wc_values_dc9": r"$\delta C_9$",
}


# Plot 2D distributions

vars_ = feature_names + [label_name]
norms = ["linear", "log"]

for var, norm in product(vars_, norms):

    fig, ax = subplots(layout="constrained")    
    ax.hist2d(
        x=df[var], 
        y=df[label_name], 
        density=True,
        bins=100,
        norm=norm,
    )
    ax.set_box_aspect(1)
    ax.set_xlabel(fancy_names[var], fontsize=18)
    ax.set_ylabel(fancy_names[label_name], fontsize=18)
    ax.set_ylim(-10.1, 0.1)
    ax.set_xlim(-1.01, 1.01)
    path = dir_.joinpath(f"{var}_{label_name}_{norm}.png")
    fig.savefig(path, bbox_inches="tight")
    close(fig)