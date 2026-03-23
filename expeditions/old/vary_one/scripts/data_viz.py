

from math import pi

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm, Colormap, Normalize
from matplotlib.cm import ScalarMappable
import pandas


data = pandas.read_parquet("../data/combined_vary_c_10_val.parquet")
data = data.xs("gen", level="sim_type")
data.loc[1:]

var_names = {"q_squared": r"$q^2$ [GeV$^2$]", "cos_theta_mu": r"$\cos\theta_\mu$", "cos_theta_k":r"$\cos\theta_K$", "chi": r"$\chi$"}
hist_intervals = {"q_squared":(0, 20), "cos_theta_mu":(-1, 1), "cos_theta_k":(-1, 1), "chi":(0, 2*pi)}
n_bins = 10
alpha = 0.85
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.dpi": 400, 
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern",
    "text.latex.preamble": r"\usepackage{array} \usepackage{tabularx}"
})
fig, axs = plt.subplots(2, 2, layout="constrained")


norm = CenteredNorm(vcenter=0, halfrange=1)
cmap = mpl.colormaps["coolwarm"]

for var, ax in zip(["q_squared", "cos_theta_mu", "cos_theta_k", "chi"], axs.flat):
    
    # legend_title = r"\begin{tabularx}{4cm}{ >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}X } $\delta C_7$ & $\delta C_9$ & $\delta C_{10}$ \\ \end{tabularx}"
    # ax.hist([], label=legend_title, alpha=0)

    for trial in data.index.get_level_values("trial_num").unique()[:10]:
        trial_data = data.xs(trial, level="trial_num")

        # label = one_row_dataframe_to_series(trial_data[["dc7", "dc9", "dc10"]].drop_duplicates())
        # lab_color = label_to_color(**label)
        # rgb_color = skimage.color.lab2rgb(lab_color)

        # hist_label = r"\begin{tabularx}{4cm}{ >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}X } " +  f"{label["dc7"]:+10.2f} & {label["dc9"]:+10.2f} & {label["dc10"]:+10.2f} \\ " +  r"\end{tabularx}" 
        # zorder = 0 if trial != -1 else 5
        label = trial_data.index.get_level_values("delta_c_10").drop_duplicates().item()
        color = cmap(norm(label))
        ax.hist(trial_data[var], bins=n_bins, range=hist_intervals[var], color=color, histtype="step", linewidth=1.5, alpha=alpha)

    ax.set_xlabel(var_names[var], fontsize=15)
    if var == "chi": ax.set_xticks([0, pi/2, pi, 3*pi/2,  2*pi], [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

# axs.flat[3].legend(ncols=1)
cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axs, alpha=alpha)
# cbar.set_ticks([-2, 0, 1])
cbar.set_label(r"$\delta C_{10}$", fontsize=15)
plt.savefig("../results/dist_vary_delta_c_10.png", bbox_inches="tight")


