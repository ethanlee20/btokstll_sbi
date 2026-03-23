
import numpy
import pandas
import matplotlib as mpl
import matplotlib.pyplot as plt


def bin_in_var(
    dataframe,
    name_of_binning_variable, 
    start, 
    stop, 
    num_bins,
    return_bin_edges=False,
):
    
    bin_edges = numpy.linspace(start=start, stop=stop, num=num_bins+1)
    bins = pandas.cut(dataframe[name_of_binning_variable], bin_edges, include_lowest=True) # the interval each event falls into
    groupby_binned = dataframe.groupby(bins, observed=False)

    if return_bin_edges:
        return groupby_binned, bin_edges
    return groupby_binned


def calc_bin_middles(start, stop, num_bins):

    bin_edges, step = numpy.linspace(
        start=start,
        stop=stop,
        num=num_bins+1,
        retstep=True,
    )
    bin_middles = bin_edges[:-1] + step/2 
    return bin_middles


def calc_afb_of_q_sq(dataframe, num_bins, start, stop):

    """
    Calcuate Afb as a function of q squared.
    Afb is the forward-backward asymmetry.
    """

    def calc_num_forward(df):

        return df["cos_theta_mu"][(df["cos_theta_mu"] > 0) & (df["cos_theta_mu"] < 1)].count()
    
    def calc_num_backward(df):

        return df["cos_theta_mu"][(df["cos_theta_mu"] > -1) & (df["cos_theta_mu"] < 0)].count()

    def calc_afb(df):

        f = calc_num_forward(df)
        b = calc_num_backward(df)
        return (f - b) / (f + b)

    def calc_afb_err(df):

        f = calc_num_forward(df)
        b = calc_num_backward(df)
        f_stdev = numpy.sqrt(f)
        b_stdev = numpy.sqrt(b)
        return 2*f*b / (f+b)**2 * numpy.sqrt((f_stdev/f)**2 + (b_stdev/b)**2) # this is stdev?

    groupby_binned = bin_in_var(
        dataframe, 
        "q_squared", 
        start,
        stop,
        num_bins
    )
    
    afbs = groupby_binned.apply(calc_afb)
    errs = groupby_binned.apply(calc_afb_err)
    bin_middles = calc_bin_middles(start, stop, num_bins)

    return bin_middles, afbs, errs


def calc_afb_of_q_sq_over_wc(dataframe, wc, num_bins, start, stop):

    dc9_values = []
    bin_mids_over_dc9 = []
    afbs_over_dc9 = []
    afb_errs_over_dc9 = []

    for trial, df in (dataframe.groupby(level="trial_num")):

        wc_ = df.index.get_level_values(wc).unique().item()
        dc9_values.append(wc_)
        bin_mids, afbs, afb_errs = calc_afb_of_q_sq(df, num_bins, start, stop)
        bin_mids_over_dc9.append(bin_mids)
        afbs_over_dc9.append(afbs)
        afb_errs_over_dc9.append(afb_errs)

    return dc9_values, bin_mids_over_dc9, afbs_over_dc9, afb_errs_over_dc9


def calc_s5_of_q_sq(dataframe, num_bins, start, stop):
    
    def calc_num_forward(df):

        cos_theta_k = df["cos_theta_k"]
        chi = df["chi"]
        
        return df[
            (((cos_theta_k > 0) & (cos_theta_k < 1)) & ((chi > 0) & (chi < numpy.pi/2)))
            | (((cos_theta_k > 0) & (cos_theta_k < 1)) & ((chi > 3*numpy.pi/2) & (chi < 2*numpy.pi)))
            | (((cos_theta_k > -1) & (cos_theta_k < 0)) & ((chi > numpy.pi/2) & (chi < 3*numpy.pi/2)))
        ].count().max()

    def calc_num_backward(df):

        cos_theta_k = df["cos_theta_k"]
        chi = df["chi"]
        
        return df[
            (((cos_theta_k > 0) & (cos_theta_k < 1)) & ((chi > numpy.pi/2) & (chi < 3*numpy.pi/2)))
            | (((cos_theta_k > -1) & (cos_theta_k < 0)) & ((chi > 0) & (chi < numpy.pi/2)))
            | (((cos_theta_k > -1) & (cos_theta_k < 0)) & ((chi > 3*numpy.pi/2) & (chi < 2*numpy.pi)))
        ].count().max()

    def calc_s5(df):

        f = calc_num_forward(df)
        b = calc_num_backward(df)

        try: 

            s5 = 4/3 * (f - b) / (f + b)

        except ZeroDivisionError:

            print("S5 calculation: division by 0, returning nan")
            s5 = numpy.nan

        return s5

    def calc_s5_err(df):

        """
        Calculate the error of S_5.

        The error is calculated by assuming the forward and backward
        regions have uncorrelated Poisson errors and propagating
        the errors.
        """

        f = calc_num_forward(df)
        b = calc_num_backward(df)

        f_stdev = numpy.sqrt(f)
        b_stdev = numpy.sqrt(b)

        try: 

            err =  4/3 * 2*f*b / (f+b)**2 * numpy.sqrt((f_stdev/f)**2 + (b_stdev/b)**2) # this is stdev?

        except ZeroDivisionError:

            print("S5 error calculation: division by 0, returning nan")
            err = numpy.nan
        
        return err

    groupby_binned = bin_in_var(
        dataframe, 
        "q_squared", 
        start,
        stop,
        num_bins,     
    )

    s5s = groupby_binned.apply(calc_s5)
    errs = groupby_binned.apply(calc_s5_err)
    bin_middles = calc_bin_middles(start, stop, num_bins)

    return bin_middles, s5s, errs


def calc_s5_of_q_sq_over_wc(dataframe, wc, num_bins, start, stop):

    dc9_values = []
    bin_mids_over_dc9 = []
    s5s_over_dc9 = []
    s5_errs_over_dc9 = []

    for trial, df in dataframe.groupby(level="trial_num"):

        wc_ = df.index.get_level_values(wc).unique().item()
        dc9_values.append(wc_)
        bin_middles, s5s, s5_errs = calc_s5_of_q_sq(df, num_bins, start, stop)
        bin_mids_over_dc9.append(bin_middles)
        s5s_over_dc9.append(s5s)
        s5_errs_over_dc9.append(s5_errs)

    return dc9_values, bin_mids_over_dc9, s5s_over_dc9, s5_errs_over_dc9



def plot_afb(afb_results, ax, cmap, norm, alpha):
    
    for dc9, bin_mids, afbs, afb_errs in zip(*afb_results):
        
        color = cmap(norm(dc9), alpha=alpha)
        
        ax.scatter(
            bin_mids, 
            afbs, 
            color=color, 
            edgecolors='none',
            s=10,
        )

        ax.errorbar(
            bin_mids, 
            afbs, 
            yerr=afb_errs, 
            fmt='none', 
            ecolor=color, 
            elinewidth=0.5, 
            capsize=0, 
        )

    ax.set_ylabel(r"$A_{FB}$")
    ax.set_ylim(-0.25, 0.46)


def plot_s5(s5_results, ax, cmap, norm, alpha):

    for dc9, bin_mids, s5s, s5_errs in zip(*s5_results):
        
        color = cmap(norm(dc9), alpha=alpha)

        ax.scatter(
            bin_mids, 
            s5s, 
            color=color, 
            edgecolors='none',
            s=10,
        )

        ax.errorbar(
            bin_mids, 
            s5s, 
            yerr=s5_errs, 
            fmt='none', 
            ecolor=color, 
            elinewidth=0.5, 
            capsize=0, 
        )
    
    
    ax.set_ylabel(r"$S_{5}$")
    ax.set_ylim(-0.48, 0.38)


def plot_afb_and_s5(fig, axs_2x1, afb_results, s5_results, alpha=0.85):

    def get_colorbar_halfrange(delta_wc_values):        
        return abs(min(delta_wc_values))
    
    assert afb_results[0] == s5_results[0]
   
    afb_ax = axs_2x1.flat[0]
    s5_ax = axs_2x1.flat[1]

    cmap = plt.cm.coolwarm
    norm = mpl.colors.CenteredNorm(
        vcenter=0, 
        halfrange=get_colorbar_halfrange(afb_results[0]),
    )

    plot_afb(afb_results=afb_results, ax=afb_ax, cmap=cmap, norm=norm, alpha=alpha)
    plot_s5(s5_results=s5_results, ax=s5_ax, cmap=cmap, norm=norm, alpha=alpha)

    # s5_ax.get_legend().remove()
    # afb_ax.get_xlabel().remove()

    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
        ax=axs_2x1, 
        orientation='vertical', 
        label=r'$\delta C_9$',
    )

    fig.supxlabel(r"$q^2$ [GeV$^2$]")


for i in (7,9,10):

    data = pandas.read_parquet(f"../data/combined_vary_c_{i}_val.parquet")
    wc_max = data.index.get_level_values(f"delta_c_{i}").max()
    wc_min = data.index.get_level_values(f"delta_c_{i}").min()
    data = data[(data.index.get_level_values(f"delta_c_{i}") == wc_max) | (data.index.get_level_values(f"delta_c_{i}") == wc_min)]

    s5_result = calc_s5_of_q_sq_over_wc(data, wc=f"delta_c_{i}", num_bins=10, start=0, stop=20)
    afb_result = calc_afb_of_q_sq_over_wc(data, wc=f"delta_c_{i}", num_bins=10, start=0, stop=20)


    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.dpi": 400, 
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern",
        "text.latex.preamble": r"\usepackage{array} \usepackage{tabularx}"
    })

    fig, axs = plt.subplots(1,2, layout="constrained", figsize=(6,3))

    afb_ax = axs.flat[0]
    s5_ax = axs.flat[1]


    def get_colorbar_halfrange(delta_wc_values):        
            return abs(min(delta_wc_values))
        
    assert afb_result[0] == s5_result[0]

    # cmap = plt.cm.coolwarm
    # norm = mpl.colors.CenteredNorm(
    #     vcenter=0, 
    #     halfrange=get_colorbar_halfrange(afb_result[0]),
    # )
    alpha = 1
    point_size = 18
    font_size = 14

    colors = ["coral", "cornflowerblue"]

    for dc9, bin_mids, afbs, afb_errs, color in zip(*afb_result, colors):
        
        # color = cmap(norm(dc9), alpha=alpha)
        
        afb_ax.scatter(
            bin_mids, 
            afbs, 
            color=color, 
            edgecolors='none',
            s=point_size,
        )

        afb_ax.errorbar(
            bin_mids, 
            afbs, 
            yerr=afb_errs, 
            fmt='none', 
            ecolor=color, 
            elinewidth=0.5, 
            capsize=0, 
        )

        afb_ax.set_ylabel(r"$A_{FB}$", fontsize=font_size)
        afb_ax.set_ylim(-0.25, 0.46)


    for dc9, bin_mids, s5s, s5_errs, color in zip(*s5_result, colors):
        
        # color = cmap(norm(dc9), alpha=alpha)

        s5_ax.scatter(
            bin_mids, 
            s5s, 
            color=color, 
            edgecolors='none',
            s=point_size,
            label=r"$\delta C_{" + f"{i}" + r"}$:" + f" {dc9:>.2f}"
        )

        s5_ax.errorbar(
            bin_mids, 
            s5s, 
            yerr=s5_errs, 
            fmt='none', 
            ecolor=color, 
            elinewidth=0.5, 
            capsize=0, 
        )


    s5_ax.set_ylabel(r"$S_{5}$", fontsize=font_size)
    s5_ax.set_ylim(-0.48, 0.38)

    s5_ax.legend()


    # fig.colorbar(
    #     mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
    #     ax=axs, 
    #     orientation='vertical', 
    #     label=r'$\delta C_9$',
    # )

    fig.supxlabel(r"$q^2$ [GeV$^2$]", x=0.55)


    plt.savefig(f"../results/asym_vary_c_{i}.png", bbox_inches="tight")


