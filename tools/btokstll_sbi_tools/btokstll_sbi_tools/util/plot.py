
from pathlib import Path

from matplotlib.pyplot import (
    style, 
    rcParams, 
    savefig, 
    close,
)


def turn_on_hq_plots(
) -> None:
    rcParams.update(
        {
            "figure.dpi": 400, 
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Computer Modern",
        }
    )


def turn_on_dark_plots(
) -> None:
    style.use("dark_background")


def save_plot_and_close(
    path:Path|str,
) -> None:
    savefig(
        path,
        bbox_inches="tight"
    )
    close()