"""Monotonocity support functions for pyscal"""

import logging

import numpy as np
import pandas as pd

from pyscal.constants import EPSILON as epsilon


logger = logging.getLogger(__name__)


def modify_dframe_monotonicity(dframe, monotonicity, digits):
    """Modify a dataframe for monotonicity.

    Columns in the dataframe are modified in-place.

    Number intervals to consider when enforcing monotonicity::

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

    Only strict monotonicity is supported. Non-strict
    monotonicity is only allowed at upper and lower limit, or
    for all-zero vectors if that option is activated.

    For non-strict monotocity, see the function clip_accumulate()

    Args:
        dframe (pd.DataFrame): Data to modify.
        monotonicity (dict): see df2str() for syntax.
        digits (int): Number of digits to ensure monotonicity for.
    """
    validate_monotonicity_arg(monotonicity, dframe.columns)

    # Wateroil.SWOF() (and similar) supply a column view
    # of the internal wateroil.table dataframe. When asked
    # to enforce monotonicity, it must be done on a copy, both
    # for speed and for not compromising the original data.

    # Round to an accuracy one notch finer than end results,
    # to avoid representation errors:
    dframe = dframe.round(digits + 1)

    # Prepare and check columns:
    for col in monotonicity:
        if dframe[col].dtype != np.float64:
            dframe.loc[:, col] = dframe[col].astype(float)

        # Bail on clearly erroneous data:
        check_almost_monotone(dframe[col], digits, monotonicity[col]["sign"])

        check_limits(dframe[col], monotonicity[col])

    # Modify data for monotonicity:
    for col in monotonicity:
        accuracy = 1.0 / 10.0 ** digits - epsilon

        if "allowzero" in monotonicity[col]:
            # Treat zero as an exception for strict monotonicity:
            max_value = dframe[col].abs().max()
            if max_value < accuracy and monotonicity[col]["allowzero"]:
                continue

        constants = rows_to_be_fixed(dframe[col], monotonicity[col], digits)
        iterations = 0
        sign = monotonicity[col]["sign"]
        while constants.any():
            iterations += 1
            if iterations > 2 * len(dframe[col]):
                raise Exception("Too many iterations for monotonicity fix")

            dframe.loc[constants, col] = (
                dframe.loc[constants, col] + sign / 10.0 ** digits - epsilon
            )

            # Ensure nonstrict monotonicity and clips after each modification:
            dframe[col] = clip_accumulate(dframe[col], monotonicity[col])

            # Evaluate what is left to fix:
            constants = rows_to_be_fixed(dframe[col], monotonicity[col], digits)

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

        # Check result for monotonicity:
        # Is this possible when rows_to_be_fixed returns none??
        allowance = 1.0 / 10.0 ** digits
        if sign > 0:
            if (dframe[col].round(digits).diff() < -allowance).any():
                raise ValueError("Not possible to make colum monotonically increasing")
        else:
            if (dframe[col].round(digits).diff() > allowance).any():
                raise ValueError("Not possible to make colum monotonically decreasing")
    return dframe


def clip_accumulate(series, monotonicity):
    """
    Modify a series (vector of numbers) for non-strict monotonicity, and
    optionally clip at lower and upper limits.

    Args:
        series (pd.Series or np.array): Vector of numbers to modify
        monotonicity (dict): Monotonocity options. The keys 'lower' and 'upper'
            can be provided for clipping the vector.

    Returns:
        np.array, copy of original.
    """
    if isinstance(series, (list, np.ndarray)):
        series = pd.Series(series, dtype="float64")
    if monotonicity["sign"] > 0:
        series = np.maximum.accumulate(series)
    else:
        series = np.minimum.accumulate(series)
    if "lower" in monotonicity and "upper" in monotonicity:
        series.clip(
            lower=monotonicity["lower"], upper=monotonicity["upper"], inplace=True
        )
    elif "lower" in monotonicity:
        series.clip(lower=monotonicity["lower"], inplace=True)
    elif "upper" in monotonicity:
        series.clip(upper=monotonicity["upper"], inplace=True)
    return series


def check_limits(series, monotonicity, colname=""):
    """
    Check a series whether it obeys numerical limits.
    Equivalence to limits is allowed.

    Exceptions will be raised in case of error. Nothing is returned
    when everything is ok.

    Args:
        series (pd.Series): Vector of numbers to check
        monotonicity (dict): Keys 'upper' and 'lower' are optional
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
    if "upper" in monotonicity and (series > monotonicity["upper"]).any():
        raise ValueError("Values larger than upper limit in column {}".format(colname))
    if "lower" in monotonicity and (series < monotonicity["lower"]).any():
        raise ValueError("Values smaller than lower limit in column {}".format(colname))


def rows_to_be_fixed(series, monotonicity, digits):
    """Compute boolean array of rows that must be modified

    Args:
        series (pd.Series):
        monotonicity (dict): Can contain "upper" or "lower"
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
    if monotonicity["sign"] > 0:
        constants = series.round(digits + 1).diff() < accuracy
    else:
        constants = series.round(digits + 1).diff() > -accuracy

    # Allow constants at the lower and upper limits.
    if "upper" in monotonicity:
        constants = constants & (series < (monotonicity["upper"] - accuracy))
    if "lower" in monotonicity:
        constants = constants & (series > (monotonicity["lower"] + accuracy))
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


def validate_monotonicity_arg(monotonicity, dframe_colnames):
    """
    Validate a dictionary with monotonicity arguments that
    can be given to df2str().

    Will raise ValueError exceptions if anything is wrong.

    Args:
        monotonicity (dict): Keys are 'sign', 'upper', 'lower'
            and  'allowzero'.
        dframe_colnames (list of str): Names of column names
            in dframes. Used in error messages.

    Returns:
        None
    """
    valid_keys = ["sign", "upper", "lower", "allowzero"]
    if monotonicity is None:
        return
    if not isinstance(monotonicity, dict):
        raise ValueError("monotonicity argument must be a dict")
    for col in monotonicity:
        if not isinstance(monotonicity[col], dict):
            raise ValueError("monotonicity argument must be a dict of dicts")
        if not set(monotonicity[col].keys()).issubset(valid_keys):
            raise ValueError(
                "Unknown keys in monotonicity {}".format(monotonicity[col].keys())
            )
        if col not in dframe_colnames:
            raise ValueError("Column {} does not exist in dataframe".format(col))
        if "sign" not in monotonicity[col]:
            raise ValueError("Monotonocity sign not specified for {}".format(col))
        try:
            signvalue = float(monotonicity[col]["sign"])
        except ValueError as err:
            raise ValueError(
                "Monotonocity sign {} not valid".format(monotonicity[col]["sign"])
            ) from err
        if "upper" in monotonicity[col]:
            float(monotonicity[col]["upper"])
        if "lower" in monotonicity[col]:
            float(monotonicity[col]["lower"])
        if abs(signvalue) > 1:
            raise ValueError("Monotonocity sign must be -1 or +1, not larger/smaller")

        if "allowzero" in monotonicity[col]:
            if monotonicity[col]["allowzero"] not in {True, False}:
                raise ValueError(
                    "allowzero in monotonicity argument must be True/False"
                )
