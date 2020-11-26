"""Utility functions for computations on relative permeability curves"""

import logging

import numpy as np
import pandas as pd

from ..constants import EPSILON as epsilon


logger = logging.getLogger(__name__)


def crosspoint(dframe, satcol, kr1col, kr2col):
    """Locate the crosspoint where kr1col == kr2col

    Args:
        dframe (pd.DataFrame): Dataframe with at least three columns
        satcol (str): Column name for the saturation column
        kr1col (str): Column name for first relperm column
        kr2col (str): Columnn ame for second column

    Returns:
        float, the saturation value (interpolated) where
            kr1col == kr2col, when krXcol is linearly interpolated
            as a function of the saturation values.
    """
    dframe = pd.DataFrame(dframe[[satcol, kr1col, kr2col]])  # Copy
    dframe.loc[:, "krdiff"] = dframe[kr1col] - dframe[kr2col]

    # Add a zero value for the difference column, and interpolate
    # the saturation column to the zero value
    zerodf = pd.DataFrame(index=[len(dframe)], data={"krdiff": 0.0})
    dframe = pd.concat([dframe, zerodf], sort=True).set_index("krdiff")

    if dframe.index.isnull().any():
        logger.warning("Could not compute crosspoint. Bug?")
        logger.debug(str(dframe))
        return -1

    dframe.interpolate(method="slinear", inplace=True)

    return dframe[np.isclose(dframe.index, 0.0)][satcol].values[0]


def estimate_diffjumppoint(table, xcol=None, ycol=None, side="right"):
    """Estimate the point where the y-data jumps from being linear
    in x to being nonlinear, or where it shift from one linear domain
    to another (for a piecewise linear function)

    If xcol is sw, and ycol is krw, and side is 'right', this
    will typically estimate sorw for you. If side is 'left' it will
    give you swcr.

    Args:
        table (pd.DataFrame): A Dataframe with x and y data
        xcol (string): The name of the column in table containing x-data. If
            None (default) the first column in table will be used.
        ycol (string): The name of the column in table containing y-data.
            If None (default) the second column in the table will be used.
        side (string): Must be 'left' or 'right'. Decides whether to look from
            the right side of the x-interval or from the left side for the
            linear domain.
    Returns:
        float: The x value where the start-linear domain ends.
    """

    if not xcol:
        xcol = table.columns[0]
    if not ycol:
        ycol = table.columns[1]
    assert isinstance(ycol, str)
    assert isinstance(xcol, str)
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
    # else:
    linearpart = table[(table["_lindevcumsum"] < epsilon)]
    if len(linearpart) == 1:
        linearpart = table[(table["_lindevcumsum"].shift(1) < epsilon)]
    return linearpart.iloc[-1][xcol]


def comment_formatter(multiline, prefix="-- "):
    """Prepends comment characters to every line in input

    Args:
        multiline (str): String that can contain newlines
        prefix (str): Comment characters to prepend every line with
            Default is the Eclipse comment syntax '-- '

    Returns:
        string, with newlines preserved, and where each line
            starts with the given prefix. Always ends with a newline.
    """
    if multiline is None or not multiline.strip():
        # Ensure we indicate that there is placeholder for something.
        return "-- \n"
    return (
        "\n".join([prefix + line.strip() for line in multiline.splitlines()]).strip()
        + "\n"
    )
