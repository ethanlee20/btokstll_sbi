from pathlib import Path

from numpy.typing import ArrayLike
from numpy import arange, log10, max, argmax, nonzero
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.pyplot import (
    subplots,
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


def plot_loss(
    ax: Axes,
    train_losses: ArrayLike,
    val_losses: ArrayLike | None = None,
    yscale: str = "linear",
    scatter_or_plot: str = "scatter",
    compute_log: bool = False,
    **kwargs,
):

    if scatter_or_plot not in ("scatter", "plot"):
        raise ValueError(
            "scatter_or_plot should be 'scatter' or 'plot'." f"\nGot: {scatter_or_plot}"
        )

    if val_losses is not None:
        if len(val_losses) != len(train_losses):
            raise ValueError

    plot_fn = ax.scatter if scatter_or_plot == "scatter" else ax.plot if scatter_or_plot == "plot" else None
    if plot_fn is None:
        raise ValueError

    epochs = arange(len(train_losses))

    if compute_log:
        train_losses = log10(train_losses)
    if compute_log and val_losses is not None:
        val_losses = log10(val_losses)

    plot_fn(epochs, train_losses, label="Train", **kwargs)

    if val_losses is not None:
        plot_fn(epochs, val_losses, label="Eval.", **kwargs)

    ax.set_yscale(yscale)
    ax.legend()
    ax.set_ylabel("Log Loss" if compute_log else "Loss", fontsize=15)
    ax.set_xlabel("Epoch", fontsize=15)


def plot_log_likelihood_ratio_scan(
    ax: Axes,
    parameters: ArrayLike,
    log_probabilities: ArrayLike,
    mle_x, 
    mle_y,
    confidence_interval,
    true_value: float,
    color: str,
):
    ax.plot(parameters, log_probabilities, color=color)
    ax.axvline(true_value, color=color, linestyle="--", zorder=-100)
    ax.scatter(mle_x, mle_y, color=color, s=25, zorder=100)
    ax.hlines(mle_y, confidence_interval[0], confidence_interval[-1], color=color, linestyle="--", zorder=-101)

    ax.set_ylabel(r"$\log p(\textrm{data} \;|\; \delta C_9) + c$", fontsize=15)
    ax.set_xlabel(r"$\delta C_9$", fontsize=15)    


def plot_linearity(ax, predictions, true_values, error_bars, lim,):

    ax.plot(lim, lim, color="#377eb8", label="Ideal", zorder=-100)
    ax.scatter(true_values, predictions, color="#ff7f00")
    ax.vlines(true_values, error_bars[:,0], error_bars[:, 1], color='#4daf4a', linestyle="-", zorder=-101, label=r"Log Likelihood $1\sigma$")
    ax.set_ylabel(r"Predicted $\delta C_9$", fontsize=16)
    ax.set_xlabel(r"True $\delta C_9$", fontsize=16)
    ax.legend()