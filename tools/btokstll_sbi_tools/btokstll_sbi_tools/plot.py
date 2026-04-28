
from pathlib import Path

from numpy import array, repeat
from matplotlib.axes import Axes
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


def save_fig_and_close(
    path:Path|str,
) -> None:
    savefig(
        path,
        bbox_inches="tight"
    )
    close()


def set_ax_bounds(
    ax:Axes, 
    bounds:tuple[float,float]|None=None, 
    xbounds:tuple[float,float]|None=None, 
    ybounds:tuple[float,float]|None=None,
):
    if (
        (bounds is None) 
        and (xbounds is None) 
        and (ybounds is None)
    ):
        raise ValueError
    if (
        (bounds is not None) 
        and (
            (xbounds is not None) 
            or (ybounds is not None)
        )
    ):
        raise ValueError

    xbounds = (
        bounds if bounds is not None 
        else xbounds
    )
    ybounds = (
        bounds if bounds is not None 
        else ybounds
    )
    if xbounds is None: # get rid of red squiggles
        xbounds = (0, 0)
    if ybounds is None:
        ybounds = (0, 0)
    ax.set_xbound(*xbounds)
    ax.set_ybound(*ybounds)


def set_ax_ticks(
    ax:Axes, 
    ticks:list[float]|list[int]|None=None, 
    xticks:list[float]|list[int]|None=None, 
    yticks:list[float]|list[int]|None=None,
):
    if (
        (ticks is None) 
        and (xticks is None) 
        and (yticks is None)
    ):
        raise ValueError
    if (
        (ticks is not None) 
        and (
            (xticks is not None) 
            or (yticks is not None)
        )
    ):
        raise ValueError

    xticks = (
        ticks if ticks is not None 
        else xticks
    )
    yticks = (
        ticks if ticks is not None 
        else yticks
    )
    if xticks is None: # get rid of red squiggles 
        xticks = [] 
    if yticks is None: 
        yticks = []
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)


def set_ax_labels(
    ax:Axes, 
    xlabel:str|None=None, 
    ylabel:str|None=None, 
    **kwargs
):
    if xlabel is not None:
        ax.set_xlabel(xlabel, **kwargs)
    if ylabel is not None:
        ax.set_ylabel(ylabel, **kwargs)


def plot_discrete_dist(
    ax, 
    bin_edges, 
    values, 
    **plot_kwargs,
) -> None:
    bin_edges = array(bin_edges)
    values = array(values)
    x = repeat(bin_edges, 2)[1:-1]
    y = repeat(values, 2)
    ax.plot(x, y, **plot_kwargs)


