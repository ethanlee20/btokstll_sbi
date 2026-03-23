
from pathlib import Path

from numpy import linspace
from matplotlib import colormaps
from matplotlib.pyplot import subplots, savefig, close
from matplotlib.colors import Normalize
from torch import no_grad, sum, log, exp, logsumexp, Tensor
from torch.nn import Module
from torch.nn.functional import log_softmax

from .util import Dataset, plot_discrete_dist


class Predictor:

    def __init__(
        self, 
        model:Module, 
        dataset:Dataset, 
        device:str,
    ):
        self.device = device
        self.model = model.to(device)
        self.dataset = dataset
        self.dataset.features = self.dataset.features.to(device)

    def calc_log_probs(
        self,
    ) -> Tensor:
        with no_grad():
            event_logits = self.model(self.dataset.features)
            event_log_probs = log_softmax(event_logits, dim=2)
            set_logits = sum(event_log_probs, dim=1)
            set_log_probs = log_softmax(set_logits, dim=1)
            return set_log_probs
        
    def calc_expected_values(
        self, 
        set_log_probs:Tensor, 
        bin_mids:Tensor,
    ) -> Tensor:
        
        bin_mids = bin_mids.to(self.device)
        set_log_probs = set_log_probs.to(self.device)
        
        def calc_expectation(log_probs):
            bin_shift = 1 - bin_mids[0]
            log_bin_map = log(bin_mids + bin_shift)
            if log_bin_map.isnan().any():
                breakpoint()
            expectation = exp(logsumexp(log_bin_map + log_probs, dim=0)) - bin_shift
            return expectation
    
        with no_grad():
            expected_values = Tensor(
                [calc_expectation(log_p) for log_p in set_log_probs]
            )
            return expected_values


def plot_discrete_dists(
    ax, 
    bin_edges, 
    dists, 
    colors, 
    xlabel, 
    ylabel, 
    num_xticks=4, 
    axis_label_font_size=13, 
    alpha=0.85,
) -> None:
    
    for dist, color in zip(dists, colors):
        plot_discrete_dist(
            ax, 
            bin_edges, 
            dist, 
            color=color, 
            alpha=alpha,
        )

    ticks = linspace(
        bin_edges[0], 
        bin_edges[-1],
        num=num_xticks,
    )
    ax.set_xticks(ticks)

    ax.set_xlabel(
        xlabel, 
        fontsize=axis_label_font_size,
    )
    ax.set_ylabel(
        ylabel, 
        fontsize=axis_label_font_size,
    )


def plot_linearity(
    ax, 
    labels,
    predictions,
    colors, 
    interval, 
    xlabel, 
    ylabel, 
    num_ticks=4,
    alpha=0.85,
    axis_label_fontsize=13,
) -> None:
    
    # diagonal line
    offset = abs(
        interval[1] - interval[0]
    )*0.05 
    ax.plot(
        [interval[0]+offset, interval[1]-offset], 
        [interval[0]+offset, interval[1]-offset], 
        color="grey", 
        zorder=-10, 
        alpha=0.5, 
        linestyle="--",
    )

    # scatter
    ax.scatter(
        labels, 
        predictions, 
        color=colors, 
        alpha=alpha
    )

    ticks = linspace(
        *interval, 
        num=num_ticks,
    )
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xlabel(
        xlabel,
        fontsize=axis_label_fontsize,
    )
    ax.set_ylabel(
        ylabel,
        fontsize=axis_label_fontsize,
    )


def plot_predictions(
    bin_edges,
    log_probs,
    expected_values,
    unbinned_labels,
    pred_wc_index:int,
    out_path:Path,
    cond_wc_index:int|None=None,
    figsize=(7,3), 
    layout="constrained", 
    alpha=0.85, 
    wspace=0.06,
    num_ticks=4,
    axis_label_fontsize=13,
) -> None:

    fig, axs = subplots(
        1, 
        2, 
        figsize=figsize, 
        layout=layout
    )
    dist_ax = axs[0]
    lin_ax = axs[1]
    
    fig.get_layout_engine().set(wspace=wspace)

    norm = Normalize(
        bin_edges[0], 
        bin_edges[-1],
    )
    cmap = colormaps["viridis"]
    colors = cmap(
        norm(
            unbinned_labels
        )
    )

    # distributions plot    
    ylabel = (
        r"$\log P(\delta C_{" 
        f"{pred_wc_index}" 
        r"} \, | \," 
    )
    if cond_wc_index is not None:
        ylabel += (
            r"\delta C_{" 
            f"{cond_wc_index}" 
            r"},\,"
        )
    ylabel += r"\textrm{dataset})$"
    xlabel = (
        r"$\delta C_{" 
        f"{pred_wc_index}" 
        r"}$"
    )
    plot_discrete_dists(
        dist_ax,
        bin_edges,
        log_probs,
        colors,
        xlabel=xlabel,
        ylabel=ylabel,
        num_xticks=num_ticks,
        axis_label_font_size=axis_label_fontsize,
        alpha=alpha,
    )

    # linearity plot
    interval = (bin_edges[0], bin_edges[-1])
    xlabel = (
        r"Actual $\delta C_{" 
        f"{pred_wc_index}" 
        r"}$"
    )
    ylabel = (
        r"Predicted $\delta C_{" 
        f"{pred_wc_index}" 
        r"}$"
    )
    plot_linearity(
        lin_ax, 
        unbinned_labels, 
        expected_values, 
        colors, 
        interval, 
        xlabel=xlabel, 
        ylabel=ylabel, 
        num_ticks=num_ticks, 
        alpha=alpha, 
        axis_label_fontsize=axis_label_fontsize,
    ) 

    savefig(
        out_path, 
        bbox_inches="tight",
    )
    close()