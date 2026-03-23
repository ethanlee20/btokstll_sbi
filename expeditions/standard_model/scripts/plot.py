
from matplotlib.pyplot import subplots, savefig, style, rcParams
from pandas import read_parquet


data = read_parquet("../data/combo.parquet")


style.use("dark_background")
rcParams.update({
    "figure.dpi": 400, 
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern",
    "text.latex.preamble": r"\usepackage{array} \usepackage{tabularx}"
})

fig, ax = subplots(layout="constrained", figsize=(3.5,2.5))
ax.hist(data["q_squared"], range=(0.03, 0.1), bins=50, color="sandybrown")
ax.set_xlabel(r"$q^2$ [GeV$^2$]", fontsize=13)
savefig("../plots/low_q_sq.png", bbox_inches="tight")
