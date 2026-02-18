
import numpy


def plot_discrete_distribution(ax, bin_edges, values, **plot_kwargs):

    bin_edges = numpy.array(bin_edges)
    values = numpy.array(values)

    x = numpy.repeat(bin_edges, 2)
    y = numpy.concatenate([[0], numpy.repeat(values, 2), [0]])
    ax.plot(x, y, **plot_kwargs)