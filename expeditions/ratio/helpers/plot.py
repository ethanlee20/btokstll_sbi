from pathlib import Path


from numpy.typing import ArrayLike
from numpy import arange, log10, max, argmax
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


def plot_to_file(path, plot_fn, **kwargs):
    fig, ax = subplots()
    plot_fn(ax=ax, **kwargs)
    save_fig_and_close(fig, path)


def plot_losses(
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


def plot_predictions_dataset(
    ax: Axes,
    parameters: ArrayLike,
    log_probabilities: ArrayLike,
    true_value: float,
    color: str,
):
    ax.plot(parameters, log_probabilities, color=color)
    ax.axvline(true_value, color=color, linestyle="--", zorder=-100)
    mle_y = max(log_probabilities)
    mle_x = parameters[argmax(log_probabilities)]
    ax.scatter(mle_x, mle_y, color=color, s=25, zorder=100)

    ax.set_ylabel(r"$\log p(\delta C_9 \;|\; \textrm{data}) + C$", fontsize=15)
    ax.set_xlabel(r"$\delta C_9$", fontsize=15)


def plot_predictions_multiple_datasets(
    ax: Axes,
    parameters: ArrayLike,
    log_probabilities: list[ArrayLike],
    true_values: list[float],
    colors: list[str],
):
    assert len(log_probabilities) == len(true_values) == len(colors)

    for log_probs, true_value, color in zip(log_probabilities, true_values, colors):
        plot_predictions_dataset(
            ax=ax,
            parameters=parameters,
            log_probabilities=log_probs,
            true_value=true_value,
            color=color,
        )
