
from matplotlib.pyplot import subplots
from pandas import read_parquet

from btokstll_sbi_tools.util.misc import (
    turn_on_dark_plots, 
    save_plot_and_close
)


turn_on_dark_plots(
)
data = read_parquet(
    "data/combined.parquet"
)
fig, ax = subplots(
    layout="constrained", 
    figsize=(3.5,2.5),
)
ax.hist(
    data["q_squared"], 
    range=(0.03, 0.1), 
    bins=50, 
    color="sandybrown",
)
ax.set_xlabel(
    r"$q^2$ [GeV$^2$]", 
    fontsize=13,
)
save_plot_and_close(
    "plots/low_q_sq.png",
)
