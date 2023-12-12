"""Module for plotting relative permeability and capillary pressure curves.

Potential improvements:
    Plotting low/base/high curves before interpolation

    Plotting curve sets on the same plot, e.g. curves from different SATNUMs or
    low/base/high curves from a SCAL recommendation

    Option to plot GasWater curves with Sg axis (instead of Sw; the Sg column
    is already present in the dataframe)

"""

import matplotlib.pyplot as plt
import pandas as pd

from pyscal.pyscallist import PyscalList
from pyscal.wateroil import WaterOil
from pyscal.gaswater import GasWater
from pyscal.wateroilgas import WaterOilGas
from pyscal.gasoil import GasOil


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
    ax.set_xlim([0, 1])

    # Log y-axis fro semilog plot
    # Some cases may have kr > 1 depending on base/absolute
    # permeability used to calculate kr
    if kwargs["semilog"]:
        ax.set_yscale("log")
        ax.set_ylim([1e-6, 1])
    else:
        ax.set_ylim([0, 1])

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
    print(kwargs)

    ax = fig.gca()

    # Set axis lables
    ax.set_xlabel(kwargs["xlabel"])
    ax.set_ylabel(kwargs["pc_name"].lower().capitalize())

    # Set axis limits
    ax.set_xlim([0, 1])

    # Set lower y-axis limit to 0 if Pc >= 0
    if not neg_pc:
        ax.set_ylim(bottom=0)

    return fig


def plot_pc(table: pd.DataFrame, satnum: int, **kwargs) -> None:
    """
    Plot capillary pressure curves.

    Called if the Pc plot is requested, regardless of if Pc is non-zero.

    Args:
        table (pd.DataFrame): Saturation table with Pc curves to be plotted
        satnum (int): SATNUM number
    """

    axis = kwargs["axis"]
    pc_name = kwargs["pc_name"].lower()

    fig = plt.figure(1, figsize=(5, 5), dpi=300)
    plt.title(f"SATNUM {satnum}")

    plt.plot(table[axis], table["PC"])

    # Flag for negative Pc
    neg_pc = table["PC"].min() < 0

    fig = format_cap_pressure_plot(fig, neg_pc, **kwargs)

    fname = f"{pc_name}_SATNUM_{satnum}"

    fig.savefig(
        fname,
        bbox_inches="tight",
    )

    fig.clear()


def plot_individual_curves(
    curve_type: str, table: pd.DataFrame, satnum: int, **kwargs
) -> None:
    """

    Function for plotting one relperm curve set for each SATNUM.

    Takes kwargs from the plotter() function, which in turn come from the
    pyscal CLI, and are passed on to the plot formatting functions.

    Args:
        curve_type (str): Used to pick the correct plot config options
        table (pd.DataFrame): Saturation table with curves to be plotted
        satnum (int): SATNUM number
    """

    # Get data from plot config dictionary based on the curve type
    # Should this dependency be injected?
    # Have chosen to assign local variables here for code readability
    plot_config = PLOT_CONFIG_OPTIONS[curve_type]
    axis = plot_config["axis"]
    kra_curve = plot_config["kra_name"]
    krb_curve = plot_config["krb_name"]
    curve_names = plot_config["curves"]
    kra_colour = plot_config["kra_colour"]
    krb_colour = plot_config["krb_colour"]

    # If semilog plot, add suffix to the name of the saved relperm figure
    if kwargs["semilog"]:
        suffix = "_semilog"
    else:
        suffix = ""

    # Plotting relative permeability curves
    fig = plt.figure(1, figsize=(5, 5), dpi=300)

    plt.title(f"SATNUM {satnum}")

    # Plot first relperm curve
    plt.plot(
        table[axis],
        table[kra_curve],
        label=kra_curve.lower(),
        color=kra_colour,
    )

    # Plot second relperm curve
    plt.plot(
        table[axis],
        table[krb_curve],
        label=krb_curve.lower(),
        color=krb_colour,
    )

    fig = format_relperm_plot(fig, **kwargs, **plot_config)

    fname = f"{curve_names}_SATNUM_{satnum}{suffix}"

    fig.savefig(
        fname,
        bbox_inches="tight",
    )

    # Clear figure so that it is empty for the next SATNUM's plot
    fig.clear()

    # If Pc plot has been requestd, plot Pc
    if kwargs["pc"]:
        plot_pc(table, satnum, **plot_config)


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
    gasoil.rename(columns={"SL": "SW"}, inplace=True)
    gasoil["SG"] = 1 - gasoil["SW"]
    wateroil = model.wateroil.table[["SW", "KRW", "PC"]].copy()

    # Sort by SW to be able to join on index
    gasoil.sort_values("SW", ascending=True, inplace=True)
    wateroil.sort_values("SW", ascending=True, inplace=True)
    gasoil.reset_index(inplace=True, drop=True)
    wateroil.reset_index(inplace=True, drop=True)

    # Check if SW in the two models differs
    assert (
        gasoil["SW"].round(2).tolist() == wateroil["SW"].round(2).tolist()
    ), "SW for the GasOil model does not match SW for the WaterOil model"

    # Merge dataframes and format
    gaswater = gasoil.merge(wateroil, left_index=True, right_index=True)
    gaswater.rename(columns={"SW_x": "SW"}, inplace=True)
    gaswater.drop("SW_y", axis=1, inplace=True)

    return gaswater


def plotter(
    models: PyscalList,
    pc: bool = False,
    semilog: bool = False,
) -> None:
    """

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
    #

    # kwargs to be passed on to other functions
    kwargs = {"pc": pc, "semilog": semilog}

    for model in models.pyscal_list:
        # Get SATNUM number as an integer. Used in the naming of saved figures
        satnum = model.tag.split("SATNUM")[1]

        if isinstance(model, WaterOilGas):
            plot_individual_curves("WaterOil", model.wateroil.table, satnum, **kwargs)
            plot_individual_curves("GasOil", model.gasoil.table, satnum, **kwargs)

        elif isinstance(model, WaterOil):
            plot_individual_curves("WaterOil", model.table, satnum, **kwargs)

        elif isinstance(model, GasOil):
            plot_individual_curves("WaterOil", model.table, satnum, **kwargs)

        elif isinstance(model, GasWater):
            # The GasWater object has a different structure to the others, and
            # requires formatting
            table = format_gaswater_table(model)
            plot_individual_curves("GasWater", table, satnum, **kwargs)

        # else:
        #     raise KeyError(
        #         f"Model type received was {type(model)} but\
        #  must be one of: {WaterOil, WaterOilGas, GasWater}"
        #     )
