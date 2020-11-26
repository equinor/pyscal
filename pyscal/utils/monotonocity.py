"""Monotonocity support functions for pyscal"""

import logging

import numpy as np
import pandas as pd

from pyscal.constants import EPSILON as epsilon


logger = logging.getLogger(__name__)


def modify_dframe_monotonocity(dframe, monotonocity, digits):
    """Modify a dataframe for monotonicity.

    Columns in the dataframe are modified in-place.

    Number intervals to consider when enforcing monotonocity::

      <value>                          <orig>    <fixed>
      <lower limit>                     0.00      0.00
      <values smaller than accuracy>    0.0002    0.00
      <accuracy limit>                  0.01      0.01
      <potential constants>             0.010001  0.02
      <allow ups/downs below accuracy>  0.0100001 0.03
                                        0.01      0.04
      <upper limit minus accuracy>      0.99      0.99
      <values too close to upper limit> 0.999     1.00
      <overshooting values>             1.0001    1.00
      <upper limit>                     1.00      1.00

    Values close to  upper or lower limits (if limits are
    supplied), but which deviate from the limit by less
    than the requested accuracy are allowed, and will be
    shifted to the limits.

    Only strict monotonocity is supported. Non-strict
    monotonicity is only allowed at upper and lower limit, or
    for all-zero vectors if that option is activated.

    For non-strict monotocity, see the function clip_accumulate()

    Args:
        dframe (pd.DataFrame): Data to modify.
        monotonocity (dict): see df2str() for syntax.
        digits (int): Number of digits to ensure monotonocity for.
    """
    validate_monotonocity_arg(monotonocity, dframe.columns)

    # Wateroil.SWOF() (and similar) supply a column view
    # of the internal wateroil.table dataframe. When asked
    # to enforce monotonocity, it must be done on a copy, both
    # for speed and for not compromising the original data.

    # Round to an accuracy one notch finer than end results,
    # to avoid representation errors:
    dframe = dframe.round(digits + 1)

    # Prepare and check columns:
    for col in monotonocity:
        if dframe[col].dtype != np.float64:
            dframe.loc[:, col] = dframe[col].astype(float)

        # Bail on clearly erroneous data:
        check_almost_monotone(dframe[col], digits, monotonocity[col]["sign"])

        check_limits(dframe[col], monotonocity[col])

    # Modify data for monotonocity:
    for col in monotonocity:
        accuracy = 1.0 / 10.0 ** digits - epsilon

        if "allowzero" in monotonocity[col]:
            # Treat zero as an exception for strict monotonocity:
            max_value = dframe[col].abs().max()
            if max_value < accuracy and monotonocity[col]["allowzero"]:
                continue

        constants = rows_to_be_fixed(dframe[col], monotonocity[col], digits)
        iterations = 0
        sign = monotonocity[col]["sign"]
        while constants.any():
            iterations += 1
            if iterations > 2 * len(dframe[col]):
                raise Exception("Too many iterations for monotonocity fix")

            dframe.loc[constants, col] = (
                dframe.loc[constants, col] + sign / 10.0 ** digits - epsilon
            )

            # Ensure nonstrict monotonocity and clips after each modification:
            dframe[col] = clip_accumulate(dframe[col], monotonocity[col])

            # Evaluate what is left to fix:
            constants = rows_to_be_fixed(dframe[col], monotonocity[col], digits)

        # Warn if more iterations than 5% of the rows
        # (number of iterations do not necessarily correspond with
        # number of changed rows)
        if float(iterations) / float(len(dframe[col])) > 0.05:
            logger.warning(
                "Needed %s iterations on column %s of length %s",
                str(iterations),
                col,
                str(len(dframe[col])),
            )

        # Check result for monotonocity:
        # Is this possible when rows_to_be_fixed returns none??
        allowance = 1.0 / 10.0 ** digits
        if sign > 0:
            if (dframe[col].round(digits).diff() < -allowance).any():
                raise ValueError("Not possible to make colum monotonically increasing")
        else:
            if (dframe[col].round(digits).diff() > allowance).any():
                raise ValueError("Not possible to make colum monotonically decreasing")
    return dframe


def clip_accumulate(series, monotonocity):
    """
    Modify a series (vector of numbers) for non-strict monotonocity, and
    optionally clip at lower and upper limits.

    Args:
        series (pd.Series or np.array): Vector of numbers to modify
        monotonocity (dict): Monotonocity options. The keys 'lower' and 'upper'
            can be provided for clipping the vector.

    Returns:
        np.array, copy of original.
    """
    if isinstance(series, (list, np.ndarray)):
        series = pd.Series(series, dtype="float64")
    if monotonocity["sign"] > 0:
        series = np.maximum.accumulate(series)
    else:
        series = np.minimum.accumulate(series)
    if "lower" in monotonocity and "upper" in monotonocity:
        series.clip(
            lower=monotonocity["lower"], upper=monotonocity["upper"], inplace=True
        )
    elif "lower" in monotonocity:
        series.clip(lower=monotonocity["lower"], inplace=True)
    elif "upper" in monotonocity:
        series.clip(upper=monotonocity["upper"], inplace=True)
    return series


def check_limits(series, monotonocity, colname=""):
    """
    Check a series whether it obeys numerical limits.
    Equivalence to limits is allowed.

    Exceptions will be raised in case of error. Nothing is returned
    when everything is ok.

    Args:
        series (pd.Series): Vector of numbers to check
        monotonocity (dict): Keys 'upper' and 'lower' are optional
            and point to numerical limits.
        colname (str): Optional string for a column name that will be
            included in any error message.
    Returns:
        None
    """
    if isinstance(series, (list, np.ndarray)):
        series = pd.Series(series, dtype="float64")
    if series.empty:
        return
    if "upper" in monotonocity and (series > monotonocity["upper"]).any():
        raise ValueError("Values larger than upper limit in column {}".format(colname))
    if "lower" in monotonocity and (series < monotonocity["lower"]).any():
        raise ValueError("Values smaller than lower limit in column {}".format(colname))


def rows_to_be_fixed(series, monotonocity, digits):
    """Compute boolean array of rows that must be modified

    Args:
        series (pd.Series):
        monotonocity (dict): Can contain "upper" or "lower"
            numerical bounds, and "sign", where >0 means positive.
            "sign" is mandatory.
        digits (int): Accuracy required, how many digits
            that are to be printed, and to which we should relate
            constancy to.
    Returns:
        boolean series.
    """
    if isinstance(series, (list, np.ndarray)):
        series = pd.Series(series, dtype="float64")

    # minus epsilon is critical to avoid being greedy
    accuracy = 1.0 / 10.0 ** digits - epsilon
    if monotonocity["sign"] > 0:
        constants = series.round(digits + 1).diff() < accuracy
    else:
        constants = series.round(digits + 1).diff() > -accuracy

    # Allow constants at the lower and upper limits.
    if "upper" in monotonocity:
        constants = constants & (series < (monotonocity["upper"] - accuracy))
    if "lower" in monotonocity:
        constants = constants & (series > (monotonocity["lower"] + accuracy))
    return constants


def check_almost_monotone(series, digits, sign):
    """Raise a ValueError if a series is not sufficiently close
    to constant or monotone in a certain direction.

    Args:
        series (pd.Series): Vector of numbers
        digits (int):
        sign (int): direction. >0 means positive
    """
    if isinstance(series, (list, np.ndarray)):
        series = pd.Series(series, dtype="float64")

    allowance = 1.0 / 10.0 ** (digits - 1)
    if sign > 0:
        if series.diff().min() < -allowance:
            raise ValueError("Series is not almost monotone")
    else:
        if series.diff().max() > allowance:
            raise ValueError("Series is not almost monotone")


def validate_monotonocity_arg(monotonocity, dframe_colnames):
    """
    Validate a dictionary with monotonocity arguments that
    can be given to df2str().

    Will raise ValueError exceptions if anything is wrong.

    Args:
        monotonocity (dict): Keys are 'sign', 'upper', 'lower'
            and  'allowzero'.
        dframe_colnames (list of str): Names of column names
            in dframes. Used in error messages.

    Returns:
        None
    """
    valid_keys = ["sign", "upper", "lower", "allowzero"]
    if monotonocity is None:
        return
    if not isinstance(monotonocity, dict):
        raise ValueError("monotonocity argument must be a dict")
    for col in monotonocity:
        if not isinstance(monotonocity[col], dict):
            raise ValueError("monotonocity argument must be a dict of dicts")
        if not set(monotonocity[col].keys()).issubset(valid_keys):
            raise ValueError(
                "Unknown keys in monotonocity {}".format(monotonocity[col].keys())
            )
        if col not in dframe_colnames:
            raise ValueError("Column {} does not exist in dataframe".format(col))
        if "sign" not in monotonocity[col]:
            raise ValueError("Monotonocity sign not specified for {}".format(col))
        try:
            signvalue = float(monotonocity[col]["sign"])
        except ValueError as err:
            raise ValueError(
                "Monotonocity sign {} not valid".format(monotonocity[col]["sign"])
            ) from err
        if "upper" in monotonocity[col]:
            float(monotonocity[col]["upper"])
        if "lower" in monotonocity[col]:
            float(monotonocity[col]["lower"])
        if abs(signvalue) > 1:
            raise ValueError("Monotonocity sign must be -1 or +1, not larger/smaller")

        if "allowzero" in monotonocity[col]:
            if monotonocity[col]["allowzero"] not in {True, False}:
                raise ValueError(
                    "allowzero in monotonocity argument must be True/False"
                )


def remap_deprecated_monotonocity(monotone_column, monotone_direction):
    """Remove this function around pyscal 0.9"""
    signs = {"1": 1, "+1": 1, "inc": 1, "-1": -1, "dec": -1}
    if monotone_column is not None and monotone_direction is None:
        return {monotone_column: {"sign": -1}}
    if monotone_column is not None and monotone_direction is not None:
        if str(monotone_direction) not in signs:
            raise ValueError("Invalid monotone_direction {}".format(monotone_direction))
        return {monotone_column: {"sign": signs[str(monotone_direction)]}}
    return {}
