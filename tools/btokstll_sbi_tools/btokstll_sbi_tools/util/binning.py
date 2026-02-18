
import pandas


def bin(data:pandas.Series, bins):

    """Bin data using given bins."""

    binned_indices = pandas.cut(
        data,
        bins,
        labels=False,
        include_lowest=True
    )
    binned_intervals = pandas.cut(
        data, 
        bins, 
        labels=None, 
        include_lowest=True
    )
    binned_mids = binned_intervals.apply(lambda interval : interval.mid)
    binned = pandas.DataFrame(
        {
            "original": data,
            "bin_index": binned_indices, 
            "bin_interval": binned_intervals, 
            "bin_mid": binned_mids
        }
    )
    return binned