
from pathlib import Path
from matplotlib.pyplot import subplots, savefig, close, style, rcParams


from btokstll_sbi_tools.util.misc import load_json
from btokstll_sbi_tools.train import Loss_Table


style.use("dark_background")
rcParams.update({
    "figure.dpi": 400, 
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern",
    "text.latex.preamble": r"\usepackage{array} \usepackage{tabularx}"
})

colors = {"train": "goldenrod", "eval": "skyblue"}

for p in Path("../models/").rglob("loss.json"):

    loss_table = Loss_Table()
    loss_table.load_table_from_json(p)
    loss_dict = loss_table.as_lists()

    loss_dict["epochs"] = [int(ep) for ep in loss_dict["epochs"]]

    fig, ax = subplots(figsize=(5,4))

    for split in ("train", "eval"):
        ax.scatter(loss_dict["epochs"], loss_dict[split], label=split, s=1, color=colors[split])
    
    ax.set_ylabel("Cross Entropy Loss", fontsize=13)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.legend(fontsize=13, markerscale=5)

    savefig(p.parent.joinpath("loss.png"), bbox_inches="tight")
    close()

    