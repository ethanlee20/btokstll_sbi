from pathlib import Path

from numpy.typing import ArrayLike
from numpy import arange, log10
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.pyplot import (
    subplots,
    style,
    rcParams,
    close,
)


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


def plot_losses_on_ax(
    ax: Axes,
    train_losses: ArrayLike,
    eval_losses: ArrayLike | None = None,
    yscale: str = "linear",
    scatter_or_plot: str = "scatter",
    compute_log: bool = False,
    **kwargs,
):

    if scatter_or_plot not in ("scatter", "plot"):
        raise ValueError(
            "scatter_or_plot should be 'scatter' or 'plot'." f"\nGot: {scatter_or_plot}"
        )

    plot_fn = ax.scatter if scatter_or_plot == "scatter" else ax.plot

    epochs = arange(len(train_losses))

    if compute_log:
        train_losses = log10(train_losses)
    if compute_log and eval_losses is not None:
        eval_losses = log10(eval_losses)

    plot_fn(epochs, train_losses, label="Train", **kwargs)

    if eval_losses is not None:
        plot_fn(epochs, eval_losses, label="Eval.", **kwargs)

    ax.set_yscale(yscale)
    ax.legend()
    ax.set_ylabel("Log Loss" if compute_log else "Loss", fontsize=15)
    ax.set_xlabel("Epoch", fontsize=15)


def plot_losses_to_file(
    path: Path | str,
    train_losses: ArrayLike,
    eval_losses: ArrayLike | None = None,
    yscale: str = "linear",
    scatter_or_plot: str = "scatter",
    compute_log: bool = False,
    **kwargs,
):

    fig, ax = subplots()

    plot_losses_on_ax(
        ax,
        train_losses,
        eval_losses=eval_losses,
        yscale=yscale,
        scatter_or_plot=scatter_or_plot,
        compute_log=compute_log,
        **kwargs,
    )

    save_fig_and_close(fig, path)
