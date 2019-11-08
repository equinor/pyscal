"""Utility function for pyscal
"""
import pandas as pd
import six

from pyscal.constants import SWINTEGERS
from pyscal.constants import EPSILON as epsilon


def interpolator(
    tableobject, curve1, curve2, parameter, sat="sw", kr1="krw", kr2="krow", pc="pc"
):
    """Interpolates between two curves.

    The interpolation parameter is 0 through 1,
    irrespective of phases or low-base/base-high/low-high.

    Args:
        tabjeobject (WaterOil or GasOil): A partially setup object where
            relperm and pc columns are to be filled with numbers.
        curve1 (WaterOil or GasOil): "Low" case of interpolation (relates
            to interpolation parameter 0). Must be copies, as they
            will be modified.
        curve2: Ditto, relates to interpolation parameter 1
        parameter (float): Between 0 and 1, what you want to interpolate to.
        sat (str): Name of the saturation column, typically 'sw' or 'sg'
        kr1 (str): Name of the first relperm column ('krw' or 'krg')
        kr2 (str): Name of the second relperm column ('krow' or 'krog')
        pc (str): Name of the capillary pressure column ('pc')

    Returns:
        None, but modifies the first argument.
    """

    curve1.table.rename(columns={kr1: kr1 + "_1"}, inplace=True)
    curve2.table.rename(columns={kr1: kr1 + "_2"}, inplace=True)
    curve1.table.rename(columns={kr2: kr2 + "_1"}, inplace=True)
    curve2.table.rename(columns={kr2: kr2 + "_2"}, inplace=True)
    curve1.table.rename(columns={pc: pc + "_1"}, inplace=True)
    curve2.table.rename(columns={pc: pc + "_2"}, inplace=True)

    # Result data container:
    satresult = pd.DataFrame(data=tableobject.table[sat], columns=[sat])

    # Merge swresult with curve1 and curve2, and interpolate all
    # columns in sw:
    intdf = (
        pd.concat([curve1.table, curve2.table, satresult], sort=True)
        .set_index(sat)
        .sort_index()
        .interpolate(method="slinear")
        .fillna(method="bfill")
        .fillna(method="ffill")
    )

    # Normalized saturations does not make sense for the
    # interpolant, remove:
    for col in ["swn", "son", "swnpc", "H", "J"]:
        if col in intdf.columns:
            del intdf[col]

    intdf[kr1] = intdf[kr1 + "_1"] * (1 - parameter) + intdf[kr1 + "_2"] * parameter
    intdf[kr2] = intdf[kr2 + "_1"] * (1 - parameter) + intdf[kr2 + "_2"] * parameter
    if pc + "_1" in curve1.table.columns and pc + "_2" in curve2.table.columns:
        intdf[pc] = intdf[pc + "_1"] * (1 - parameter) + intdf[pc + "_2"] * parameter
    else:
        intdf[pc] = 0

    # Slice out the resulting sw values and columns. Slicing on
    # floating point indices is not robust so we need to slice on an
    # integer version of the sw column
    tableobject.table["swint"] = list(
        map(int, list(map(round, tableobject.table[sat] * SWINTEGERS)))
    )
    intdf["swint"] = list(map(int, list(map(round, intdf.index.values * SWINTEGERS))))
    intdf = intdf.reset_index()
    intdf.drop_duplicates("swint", inplace=True)
    intdf.set_index("swint", inplace=True)
    intdf = intdf.loc[tableobject.table["swint"].values]
    intdf = intdf[[sat, kr1, kr2, pc]].reset_index()

    # intdf['swint'] = (intdf['sw'] * SWINTEGERS).astype(int)
    # intdf.drop_duplicates('swint', inplace=True)

    # Populate the WaterOil object
    tableobject.table[kr1] = intdf[kr1]
    tableobject.table[kr2] = intdf[kr2]
    tableobject.table[pc] = intdf[pc]
    tableobject.table.fillna(method="ffill", inplace=True)
    return


def estimate_diffjumppoint(table, xcol=None, ycol=None, side="right"):
    """Estimate the point where the y-data jumps from being linear
    in x to being nonlinear, or where it shift from one linear domain
    to another (for a piecewise linear function)

    If xcol is sw, and ycol is krw, and side is 'right', this
    will typically estimate sorw for you. If side is 'left' it will
    give you swcr.

    Args:
        table (pd.DataFrame): A Dataframe with x and y data
        xcol (string): The name of the column in table containing x-data. If\
            None (default) the first column in table will be used.
        ycol (string): The name of the column in table containing y-data.
            If None (default) the second column in the table will be used.
        side (string): Must be 'left' or 'right'. Decided whether to look from
            the right side of the x-interval or from the left side for the
            linear domain.
    Returns:
        float: The x value where the start-linear domain ends.
    """

    if not xcol:
        xcol = table.columns[0]
    if not ycol:
        ycol = table.columns[1]
    assert isinstance(ycol, six.string_types)
    assert isinstance(xcol, six.string_types)
    if not side:
        raise ValueError("side cannot be None, use left or right")
    side = side.lower()
    assert side in ["left", "right"]

    # Compute the derivative:
    table["_deriv"] = table[ycol].diff() / table[xcol].diff()
    # The first becomes NaN, extrapolate from the second row:
    table.loc[0, "_deriv"] = table["_deriv"].iloc[1]

    # Pick the derivative at the first or last segment:
    iloc = {"left": 0, "right": -1}
    lin_a = table["_deriv"].iloc[iloc[side]]

    # Make a linear extrapolation from the last segment, starting at max x
    table["_linear"] = (table[xcol] - table[xcol].iloc[iloc[side]]) * lin_a + table[
        ycol
    ].iloc[iloc[side]]
    assert table["_linear"].values[iloc[side]] == table[ycol].values[iloc[side]]

    # Compute how much krw deviates from the linear krw:
    table["_lindev"] = (table[ycol] - table["_linear"]).abs()

    # Use the cumulative sum to determine the onset of non-zero deviation
    # starting from sw=1:
    table["_lindevcumsum"] = table["_lindev"].cumsum()

    if side == "right":
        maxcumsum = table["_lindevcumsum"].max()
        linearpart = table[(table["_lindevcumsum"] - maxcumsum).abs() < epsilon]
        return linearpart.iloc[1][xcol]
    else:
        linearpart = table[(table["_lindevcumsum"] < epsilon)]
        if len(linearpart) == 1:
            linearpart = table[(table["_lindevcumsum"].shift(1) < epsilon)]
        return linearpart.iloc[-1][xcol]
