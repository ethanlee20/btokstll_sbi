
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.pyplot import (
    style,
    rcParams,
    close,
)


color_blind_palette = [
    '#377eb8', '#ff7f00', '#4daf4a',
    '#f781bf', '#a65628', '#984ea3',
    '#999999', '#e41a1c', '#dede00'
]


def turn_on_hq_plots() -> None:
    rcParams.update(
        {
            "figure.dpi": 400,
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Computer Modern",
        }
    )


def turn_on_dark_plots() -> None:
    style.use("dark_background")


def save_fig_and_close(
    fig: Figure,
    path: Path | str,
) -> None:
    fig.savefig(path, bbox_inches="tight")
    close(fig)
