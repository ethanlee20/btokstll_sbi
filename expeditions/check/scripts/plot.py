
from pathlib import Path

from pandas import read_parquet
from matplotlib.pyplot import hist, show, subplots, style, rcParams, savefig

data_path = Path("../data/combined.parquet")

print(data_path.absolute())

df = read_parquet(data_path)

print(len(df))

style.use("dark_background")
rcParams.update({
    "figure.dpi": 400, 
    "text.usetex": False,
    # "font.family": "serif",
    # "font.serif": "Computer Modern",
    # "text.latex.preamble": r"\usepackage{array} \usepackage{tabularx}"
})

fig, ax = subplots(layout="constrained", figsize=(5,4))
ax.hist(df["q_squared"], bins=50, range=(0.035,0.1), color="lightblue")
ax.set_xlabel("q^2 [GeV^2]", fontsize=15)
savefig("../plots/low_q_squared.png", bbox_inches="tight")


