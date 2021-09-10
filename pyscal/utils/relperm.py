"""Utility functions for computations on relative permeability curves"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..constants import EPSILON as epsilon

logger = logging.getLogger(__name__)


def crosspoint(
    inputframe: pd.DataFrame, satcol: str, kr1col: str, kr2col: str
) -> float:
    """Locate the crosspoint where kr1col == kr2col

    Args:
        inputframe: Dataframe with at least three columns
        satcol: Column name for the saturation column
        kr1col: Column name for first relperm column
        kr2col: Columnn ame for second column

    Returns:
        The saturation value (interpolated) where kr1col == kr2col, when krXcol
        is linearly interpolated as a function of the saturation values. In case
        of errors, the function will return the value -1
    """
    if len(inputframe) < 2:
        return -1

    dframe = pd.DataFrame(inputframe[[satcol, kr1col, kr2col]])

    if dframe.isna().any().any():
        logger.error("nan in input to crosspoint()")
        logger.debug(str(dframe))
        return -1

    dframe.loc[:, "krdiff"] = dframe[kr1col] - dframe[kr2col]

    # Add a zero value for the difference column, and interpolate
    # the saturation column to the zero value
    zerodf = pd.DataFrame(index=[len(dframe)], data={"krdiff": 0.0})
    dframe = pd.concat([dframe, zerodf], sort=True).set_index("krdiff")

    dframe.interpolate(method="slinear", inplace=True)

    if dframe.isna().any().any():
        logger.error("Could not compute crosspoint)")
        logger.debug(str(dframe))
        return -1

    return dframe[np.isclose(dframe.index, 0.0)][satcol].values[0]


def estimate_diffjumppoint(
    table: pd.DataFrame,
    xcol: Optional[str] = None,
    ycol: Optional[str] = None,
    side: str = "right",
) -> float:
    """Estimate the point where the y-data jumps from being linear
    in x to being nonlinear, or where it shift from one linear domain
    to another (for a piecewise linear function)

    If xcol is sw, and ycol is krw, and side is 'right', this
    will typically estimate sorw for you. If side is 'left' it will
    give you swcr.

    Args:
        table: A Dataframe with x and y data
        xcol: The name of the column in table containing x-data. If
            None (default) the first column in table will be used.
        ycol: The name of the column in table containing y-data.
            If None (default) the second column in the table will be used.
        side: Must be 'left' or 'right'. Decides whether to look from
            the right side of the x-interval or from the left side for the
            linear domain.
    Returns:
        The x value where the start-linear domain ends.
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

    linearpart = table[(table["_lindevcumsum"] < epsilon)]
    if len(linearpart) == 1:
        linearpart = table[(table["_lindevcumsum"].shift(1) < epsilon)]
    return linearpart.iloc[-1][xcol]
