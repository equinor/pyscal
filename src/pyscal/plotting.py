"""Module for plotting relative permeability and capillary pressure curves.

Potential improvements:
    Plotting low/base/high curves before interpolation

    Plotting curve sets on the same plot, e.g. curves from different SATNUMs or
    low/base/high curves from a SCAL recommendation

    Option to plot GasWater curves with Sg axis (instead of Sw; the Sg column
    is already present in the dataframe)

"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pyscal import GasOil, GasWater, PyscalList, WaterOil, WaterOilGas, getLogger_pyscal

logger = getLogger_pyscal(__name__)

# Data for configuring plot based on pyscal model type
PLOT_CONFIG_OPTIONS = {
    "WaterOil": {
        "axis": "SW",
        "kra_name": "KRW",
        "krb_name": "KROW",
        "kra_colour": "blue",
        "krb_colour": "green",
        "pc_name": "PCOW",
        "xlabel": "Sw",
        "ylabel": "krw, krow",
        "curves": "krw_krow",
    },
    "GasOil": {
        "axis": "SG",
        "kra_name": "KRG",
        "krb_name": "KROG",
        "kra_colour": "red",
        "krb_colour": "green",
        "xlabel": "Sg",
        "ylabel": "krg, krog",
        "pc_name": "PCOG",
        "curves": "krg_krog",
    },
    "GasWater": {
        "axis": "SW",
        "kra_name": "KRW",
        "krb_name": "KRG",
        "kra_colour": "blue",
        "krb_colour": "red",
        "pc_name": "PCGW",
        "xlabel": "Sw",
        "ylabel": "krg, krw",
        "curves": "krg_krw",
    },
}


def format_gaswater_table(model: GasWater) -> pd.DataFrame:
    """

    Format the tables held by the GasWater object (through the GasOil and
    WaterOil objects).

    The GasWater object is a bit trickier to work with. Like WaterOilGas,
    GasWater takes from both WaterOil and GasOil objects, but unlike
    WaterOilGas, it is a two-phase model. It is based on one curve from GasOil,
    krg, and one curve from WaterOil, krw. This is a different format to the
    other models, where the relperm curves to be plotted in the same figure are
    found in the same table and are accessed easily using the "table" instance
    variable, e.g. WaterOil.table. To plot the correct curves for the GasWater
    model, additional formatting is required. That is handled by this function.

    Process: sort both the GasOil and WaterOil tables by increasing SW and then
    merge on index. Can't join on "SW" due to potential floating-point number
    errors.

    Args:
        model (GasWater): GasWater model

    Returns:
        pd.DataFrame: saturation table with Sw, Sg, krg from the GasOil
        instance, and krw and Pc from the WaterOil instance
    """

    gasoil = model.gasoil.table[["SL", "KRG"]].copy()
    gasoil = gasoil.rename(columns={"SL": "SW"})
    gasoil["SG"] = 1 - gasoil["SW"]
    wateroil = model.wateroil.table[["SW", "KRW", "PC"]].copy()

    # Sort by SW to be able to join on index
    gasoil = gasoil.sort_values("SW", ascending=True)
    wateroil = wateroil.sort_values("SW", ascending=True)
    gasoil = gasoil.reset_index(drop=True)
    wateroil = wateroil.reset_index(drop=True)

    # Check if SW in the two models differs using an epsilon of 1E-6 If any
    # absolute differences are greater than this threshold, an assertion error
    # will be raised.
    assert (
        abs(gasoil["SW"] - wateroil["SW"]) < 1e-6
    ).all(), "SW for the GasOil model does not match SW for the WaterOil model"

    # Merge dataframes and format
    gaswater = gasoil.merge(wateroil, left_index=True, right_index=True)
    gaswater = gaswater.rename(columns={"SW_x": "SW"})
    return gaswater.drop("SW_y", axis=1)


def get_satnum_from_tag(string: str) -> int:
    """
    Get SATNUM from the model tag. Used in the naming of figures.

    Args:
        string (str): String from the model .tag instance variable

    Returns:
        int: SATNUM number
    """
    return int(string.split("SATNUM")[1].strip())


def get_plot_config_options(curve_type: str, **kwargs) -> dict:
    """
    Get config data from plot config dictionary based on the curve (model) type.

    Args:
        curve_type (str): Name of the curve type. Allowed types are given in
        the PLOT_CONFIG_OPTIONS dictionary

    Returns:
        dict: Config parameters for the chosen model type
    """

    config = PLOT_CONFIG_OPTIONS[curve_type].copy()

    # If semilog plot, add suffix to the name of the saved relperm figure
    suffix = "_semilog" if kwargs["semilog"] else ""

    config["suffix"] = suffix

    return config


def format_relperm_plot(fig: plt.Figure, **kwargs) -> plt.Figure:
    """
    Formatting options for individual relative permeability plots.

    Args:
        fig (plt.Figure): Relative permeability figure to be formatted

    Returns:
        plt.Figure: Formatted relative permeability figure
    """

    ax = fig.gca()

    # Set axis lables
    ax.set_xlabel(kwargs["xlabel"])
    ax.set_ylabel(kwargs["ylabel"])

    # Set axis limits
    ax.set_xlim((0.0, 1.0))

    # Log y-axis fro semilog plot
    # Some cases may have kr > 1 depending on base/absolute
    # permeability used to calculate kr
    if kwargs["semilog"]:
        ax.set_yscale("log")
        ax.set_ylim((1e-6, 1.0))
    else:
        ax.set_ylim((0.0, 1.0))

    # Add legend
    plt.legend()
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=5,
    )

    return fig


def format_cap_pressure_plot(
    fig: plt.Figure, neg_pc: bool = False, **kwargs
) -> plt.Figure:
    """
    Formatting options for individual capillary pressure plots.

    Args:
        fig (plt.Figure): Capillary pressure figure to be formatted
        neg_pc (bool, optional): Negative Pc flag. True if the Pc curve has a
        negative part. Defaults to False.

    Returns:
        plt.Figure: Formatted capillary pressure figure
    """

    ax = fig.gca()

    # Set axis lables
    ax.set_xlabel(kwargs["xlabel"])
    ax.set_ylabel(kwargs["pc_name"].lower().capitalize())

    # Set axis limits
    ax.set_xlim((0.0, 1.0))

    # Set lower y-axis limit to 0 if Pc >= 0
    if not neg_pc:
        ax.set_ylim(bottom=0.0)

    return fig


def plot_pc(table: pd.DataFrame, satnum: int, **kwargs) -> plt.Figure:
    """
    Plot capillary pressure curves.

    Called if the Pc plot is requested, regardless of if Pc is non-zero.

    Args:
        table (pd.DataFrame): Saturation table with Pc curves to be plotted
        satnum (int): SATNUM number

    Returns:
        plt.Figure: Pc figure with a single Pc curve for a given model and
        SATNUM
    """

    fig = plt.figure(1, figsize=(5, 5), dpi=300)
    plt.title(f"SATNUM {satnum}")
    plt.plot(table[kwargs["axis"]], table["PC"])

    # Flag for negative Pc
    neg_pc = table["PC"].min() < 0
    fig = format_cap_pressure_plot(fig, neg_pc, **kwargs)

    # Log warning if Pc plot is requested but is zero or practically zero.
    # There is no checking of units of Pc in pyscal, but this should work as
    # intended for Pc in bar, Pa, MPa, atm and psi, i.e. the most common units
    # of pressure. In any case, the figures will still be made.
    if not (abs(table["PC"]) > 1e-6).any():
        logger.warning("Pc plots were requested, but Pc is zero.")

    return fig


def plot_relperm(
    table: pd.DataFrame, satnum: int, config: dict, **kwargs
) -> plt.Figure:
    """

    Function for plotting one relperm curve set for each SATNUM.

    Takes kwargs from the plotter() function, which in turn come from the
    pyscal CLI, and are passed on to the plot formatting functions.

    Args:
        table (pd.DataFrame): Saturation table with curves to be plotted
        satnum (int): SATNUM number
        config (dict): Plot config

    Returns:
        plt.Figure: Relative permeability figure with two relative permeability
        curves for a given model and SATNUM
    """

    # Plotting relative permeability curves
    fig = plt.figure(1, figsize=(5, 5), dpi=300)

    plt.title(f"SATNUM {satnum}")

    # Plot first relperm curve
    plt.plot(
        table[config["axis"]],
        table[config["kra_name"]],
        label=config["kra_name"].lower(),
        color=config["kra_colour"],
    )

    # Plot second relperm curve
    plt.plot(
        table[config["axis"]],
        table[config["krb_name"]],
        label=config["krb_name"].lower(),
        color=config["krb_colour"],
    )

    return format_relperm_plot(fig, **kwargs, **config)


def save_figure(
    fig: plt.Figure,
    satnum: int,
    config: dict,
    plot_type: str,
    outdir: str,
) -> None:
    """

    Save the provided figure.

    Args:
        fig (plt.Figure): Figure to be saved
        satnum (int): SATNUM number
        config (dict): Plot config
        plot_type (str): Figure type. Allowed types are 'relperm' and 'pc'
        outdir (str): Directory where the figure will be saved
    """

    # Get curve name
    if plot_type == "relperm":
        curve_names = config["curves"]
        suffix = config["suffix"]
    elif plot_type == "pc":
        curve_names = config["pc_name"].lower()
        suffix = ""
    else:
        raise ValueError(f"'type' given ({plot_type}) must be one of 'relperm' or 'pc'")

    fname = f"{curve_names}_SATNUM_{satnum}{suffix}".replace(" ", "")
    fout = Path(outdir).joinpath(fname)

    fig.savefig(
        fout,
        bbox_inches="tight",
    )

    print(f"Figure saved to {fout}.png")

    # Clear figure so that it is empty for the next SATNUM's plot
    fig.clear()


def wog_plotter(model: WaterOilGas, **kwargs) -> None:
    """

    Plot a WaterOilGas (WaterOil and GasOil) model.


    For a WaterOilGas instance, the WaterOil and GasOil instances can be
    accessed, then the "table" instance variable.

    Args:
        model (WaterOilGas): WaterOilGas instance
    """

    outdir = kwargs["outdir"]

    # the wateroil and gasoil instance variables are optional for the
    # WaterOilGas class. If statements used to check if they are provided
    if model.wateroil:
        config_wo = get_plot_config_options("WaterOil", **kwargs)
        satnum_wo = get_satnum_from_tag(model.wateroil.tag)

        fig_wo = plot_relperm(
            model.wateroil.table,
            satnum_wo,
            config_wo,
            **kwargs,
        )

        save_figure(fig_wo, satnum_wo, config_wo, "relperm", outdir)

        if kwargs["pc"]:
            fig_pcwo = plot_pc(
                model.wateroil.table,
                get_satnum_from_tag(model.wateroil.tag),
                **config_wo,
            )

            save_figure(fig_pcwo, satnum_wo, config_wo, "pc", outdir)

    if model.gasoil:
        config_go = get_plot_config_options("GasOil", **kwargs)
        satnum_go = get_satnum_from_tag(model.gasoil.tag)

        fig_go = plot_relperm(
            model.gasoil.table,
            satnum_go,
            config_go,
            **kwargs,
        )

        save_figure(fig_go, satnum_go, config_go, "relperm", outdir)


def wo_plotter(model: WaterOil, **kwargs) -> None:
    """

    Plot a WaterOil model.

    For a WaterOil instance, the saturation table can be accessed using the
    "table" instance variable.

    Args:
        model (WaterOil): WaterOil instance
    """
    config = get_plot_config_options("WaterOil", **kwargs)
    satnum = get_satnum_from_tag(model.tag)

    outdir = kwargs["outdir"]

    fig = plot_relperm(
        model.table,
        satnum,
        config,
        **kwargs,
    )

    save_figure(fig, satnum, config, "relperm", outdir)

    if kwargs["pc"]:
        fig_pc = plot_pc(model.table, get_satnum_from_tag(model.tag), **config)

        save_figure(fig_pc, satnum, config, "pc", outdir)


def go_plotter(model: GasOil, **kwargs) -> None:
    """

    Plot a GasOil model.

    For a GasOil instance, the saturation table can be accessed using the
    "table" instance variable.

    Args:
        model (GasOil): GasOil instance
    """

    config = get_plot_config_options("GasOil", **kwargs)
    satnum = get_satnum_from_tag(model.tag)

    outdir = kwargs["outdir"]

    fig = plot_relperm(
        model.table,
        satnum,
        config,
        **kwargs,
    )

    save_figure(fig, satnum, config, "relperm", outdir)

    # Note that there are no supporting functions for adding Pc to GasOil
    # instances. This can only be done by modifying the "table" instance
    # variable for a GasOil object
    if kwargs["pc"]:
        fig_pc = plot_pc(model.table, get_satnum_from_tag(model.tag), **config)

        save_figure(fig_pc, satnum, config, "pc", outdir)


def gw_plotter(model: GasWater, **kwargs) -> None:
    """

    For GasWater, the format is different, and an additional formatting step is
    required. Use the formatted table as an argument to the plotter function,
    instead of the "table" instance variable

    Args:
        model (GasWater): GasWater instance
    """

    table = format_gaswater_table(model)
    config = get_plot_config_options("GasWater", **kwargs)
    satnum = get_satnum_from_tag(model.tag)
    outdir = kwargs["outdir"]

    fig = plot_relperm(
        table,
        satnum,
        config,
        **kwargs,
    )

    save_figure(fig, satnum, config, "relperm", outdir)

    if kwargs["pc"]:
        fig_pc = plot_pc(table, get_satnum_from_tag(model.tag), **config)

        save_figure(fig_pc, satnum, config, "pc", outdir)


def plotter(
    models: PyscalList, pc: bool = False, semilog: bool = False, outdir: str = "./"
) -> None:
    """

    Runner function for creating plots.

    Iterate over PyscalList and plot curves based on type of pyscal objects
    encountered.

    PyscalList is a list of WaterOilGas, WaterOil, GasOil or GasWater objects.

    For WaterOil and GasOil, the saturation table can be accessed using the
    "table" instance variable.

    For WaterOilGas, the WaterOil and GasOil instances can be accessed, then
    the "table" instance variable.

    For GasWater, the format is different, and an additional formatting step is
    required.

    Args:
        models (PyscalList): List of models
        pc (bool, optional): Plot Pc flag. Defaults to False.
        semilog (bool, optional): Plot relperm with log y-axis. Defaults to
        False.

    """

    # kwargs to be passed on to other functions
    kwargs = {"pc": pc, "semilog": semilog, "outdir": outdir}

    for model in models.pyscal_list:
        if isinstance(model, WaterOilGas):
            wog_plotter(model, **kwargs)
        elif isinstance(model, WaterOil):
            wo_plotter(model, **kwargs)
        elif isinstance(model, GasOil):
            go_plotter(model, **kwargs)
        elif isinstance(model, GasWater):
            gw_plotter(model, **kwargs)
        else:
            raise TypeError(
                f"Model type received was {type(model)} but\
         must be one of: {WaterOil, WaterOilGas, GasOil, GasWater}"
            )
