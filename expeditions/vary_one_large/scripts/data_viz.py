
from pathlib import Path
from math import pi

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm, Colormap, Normalize
from matplotlib.cm import ScalarMappable
import pandas

var_names = {"q_squared": r"$q^2$ [GeV$^2$]", "cos_theta_mu": r"$\cos\theta_\mu$", "cos_theta_k":r"$\cos\theta_K$", "chi": r"$\chi$"}
hist_intervals = {"q_squared":(0, 20), "cos_theta_mu":(-1, 1), "cos_theta_k":(-1, 1), "chi":(0, 2*pi)}
n_bins = 10
alpha = 0.85
cmap_name = "viridis"

plt.style.use("dark_background")
plt.rcParams.update({
    "figure.dpi": 400, 
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern",
    "text.latex.preamble": r"\usepackage{array} \usepackage{tabularx}"
})


file_path = lambda wc: Path(f"../data/vary_c_{wc}_val.parquet")
label_name = lambda wc: f"wc_set_d_c_{wc}"
bound_label = lambda wc, left_or_right: f"wc_dist_d_c_{wc}_{left_or_right}"

for wc in (7, 9):

    data = pandas.read_parquet(file_path(wc))
    data = data.xs("gen", level="sim_type")

    fig, axs = plt.subplots(2, 2, layout="constrained")

    bounds = (
        data.iloc[0][bound_label(wc, "left")],
        data.iloc[0][bound_label(wc, "right")],
    )
    norm = Normalize(*bounds)
    cmap = mpl.colormaps[cmap_name]

    for var, ax in zip(var_names.keys(), axs.flat):
        
        for trial_num, df in data.groupby(level="trial_num"):
            label = df[label_name(wc)].drop_duplicates().item()
            color = cmap(norm(label))
            ax.hist(df[var], bins=n_bins, range=hist_intervals[var], color=color, histtype="step", linewidth=1.5, alpha=alpha)

        ax.set_xlabel(var_names[var], fontsize=15)
        if var == "chi": ax.set_xticks([0, pi/2, pi, 3*pi/2,  2*pi], [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axs, alpha=alpha)

    cbar.set_label(r"$\delta C_{"f"{wc}"r"}$", fontsize=15)
    plt.savefig(f"../plots/dist_vary_c{wc}.png", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots()
    ax.hist(data[label_name(wc)].drop_duplicates(), bins=100, range=bounds)
    plt.savefig(f"../plots/labels_vary_c_{wc}.png", bbox_inches="tight")
    plt.close()

