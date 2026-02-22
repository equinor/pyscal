"""Monotonicity support functions for pyscal"""

import logging
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..constants import EPSILON as epsilon

logger = logging.getLogger(__name__)


class MonotonicitySpec(TypedDict, total=False):
    """Specification of monotonicity for a vector of values"""

    sign: int
    """Value of +1 dictates strictly increasing,
    value of -1 dictates scrictly decreasing. Required parameter."""

    upper: float
    """Values will be clipped at upper limit, and non-strict monotonicity is
    allowed at limit. Optional parameter."""

    lower: float
    """Values will be clipped at lower limit, and non-strict monotonicity is
    allowed at limit. Optional parameter."""

    allowzero: bool
    """If True, consecutive zeros will be allowes in an otherwise strictly
    monotonic column. Optional parameter."""


def _quantize_to_fixed_point_int(
    value: float, digits: int, rounding: str | None = None
) -> int:
    """Convert a IEEE754 floating point to an integer representation
    for a specified accuracy.

    Examples: f(0.01, 2) becomes 1, f(1.00, 2) becomes 100.

    Decimal objects are used to guarantee that we are
    snapping the floating points to the same values as we want
    to see in the output, avoiding IEEE754 fallacies.

    When enforcing monotonicity, rounding should be ceiling
    for decreasing sequences, and floor for increasing sequences.
    """
    accuracy = Decimal(1).scaleb(-digits)
    dec = Decimal(str(value)).quantize(accuracy, rounding=rounding)
    return int(dec * 10**digits)


def _format_fixed_point_int(fixed_point_int: int, digits: int) -> str:
    """Convert integers that represent floating point number
    to strings. Numbers will be zero-padded with to the
    specified number of digits.

    Examples:
    f(100, 2) becomes "1.00" and f(1, 2) becomes "0.01"
    """
    scale = 10**digits
    sign = "-" if fixed_point_int < 0 else ""
    abs_quant_int = -fixed_point_int if fixed_point_int < 0 else fixed_point_int
    return f"{sign}{abs_quant_int // scale}.{abs_quant_int % scale:0{digits}d}"


def modify_dframe_monotonicity(
    dframe: pd.DataFrame, monotonicity: dict[str, MonotonicitySpec], digits: int
) -> pd.DataFrame:
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
        dframe: Data to modify.
        monotonicity: Keys are column names
        digits: Number of digits to ensure monotonicity for.

    Returns a dataframe where monotonicity enforced columns have
    a string datatype.
    """
    validate_monotonicity_arg(monotonicity, dframe.columns.to_list())

    # Wateroil.SWOF() (and similar) supply a column view
    # of the internal wateroil.table dataframe. When asked
    # to enforce monotonicity, it must be done on a copy, both
    # for speed and for not compromising the original data.

    # Round to an accuracy one notch finer than end results,
    # to avoid representation errors:
    dframe = dframe.round(digits + 1)

    # Prepare and check columns:
    for col, spec in monotonicity.items():
        if dframe[col].dtype != np.float64:
            dframe[col] = dframe[col].astype(float)

        assert "sign" in spec

        # Bail on clearly erroneous data:
        check_almost_monotone(dframe[col], digits, spec["sign"])

        check_limits(dframe[col], spec)

    # Modify data for monotonicity:
    for col, spec in monotonicity.items():
        if "allowzero" in spec:
            # Treat all-zero values as an exception for strict monotonicity:
            max_value = dframe[col].abs().max()
            if max_value < 1.0 / 10.0**digits - epsilon and spec["allowzero"]:
                continue

        # Default rounding in Python is ROUND_HALF_EVEN, or "Bankers rounding".
        # That rounding scheme is designed for summing numbers, not for
        # strictly monotone sequences where it is safer to either always round
        # up or down consistently
        rounding: str = ROUND_FLOOR if spec["sign"] == 1 else ROUND_CEILING

        lower_fixed_int = upper_fixed_int = None
        for boundary in ("lower", "upper"):
            if spec.get(boundary) is not None:
                boundary_value = spec.get(boundary)
                assert isinstance(boundary_value, int | float)
                if boundary == "lower":
                    lower_fixed_int = _quantize_to_fixed_point_int(
                        boundary_value, digits, rounding
                    )
                if boundary == "upper":
                    upper_fixed_int = _quantize_to_fixed_point_int(
                        boundary_value, digits, rounding
                    )

        monotone_floatstrings: list[str] = []
        last_fixed_int: int | None = None
        for boundary_value in dframe[col].to_numpy():
            fixed_int = _quantize_to_fixed_point_int(boundary_value, digits, rounding)

            if last_fixed_int is not None and fixed_int not in (
                lower_fixed_int,
                upper_fixed_int,
            ):
                if spec["sign"] == 1:
                    fixed_int = max(fixed_int, last_fixed_int + 1)
                    if upper_fixed_int is not None:  # Clamp at upper limit
                        fixed_int = min(fixed_int, upper_fixed_int)
                else:
                    fixed_int = min(fixed_int, last_fixed_int - 1)
                    if lower_fixed_int is not None:  # Clamp at lower limit
                        fixed_int = max(fixed_int, lower_fixed_int)
            last_fixed_int = fixed_int

            monotone_floatstrings.append(_format_fixed_point_int(fixed_int, digits))

        dframe[col] = monotone_floatstrings
    return dframe


def clip_accumulate(
    series: list[float] | pd.Series | npt.NDArray[np.floating],
    monotonicity: MonotonicitySpec,
) -> npt.NDArray[np.floating]:
    """
    Modify a series (vector of numbers) for non-strict monotonicity, and
    optionally clip at lower and upper limits.

    Args:
        series: Vector of numbers to modify
        monotonicity:

    Returns:
        np.array, copy of original.
    """
    series = np.array(series)
    if monotonicity["sign"] > 0:
        series = np.maximum.accumulate(series)
    else:
        series = np.minimum.accumulate(series)
    if "lower" in monotonicity and "upper" in monotonicity:
        series = series.clip(min=monotonicity["lower"], max=monotonicity["upper"])
    elif "lower" in monotonicity:
        series = series.clip(min=monotonicity["lower"])
    elif "upper" in monotonicity:
        series = series.clip(max=monotonicity["upper"])
    return series


def check_limits(
    series: list[float] | pd.Series | npt.NDArray[np.floating],
    monotonicity: MonotonicitySpec,
    colname: str = "",
) -> None:
    """
    Check a series whether it obeys numerical limits.
    Equivalence to limits is allowed.

    Exceptions will be raised in case of error. Nothing is returned
    when everything is ok.

    Args:
        series: Vector of numbers to check
        monotonicity:
        colname: Optional string for a column name that will be
            included in any error message.
    """
    if isinstance(series, (list, np.ndarray)):
        series = pd.Series(series, dtype="float64")
    if series.empty:
        return
    if "upper" in monotonicity and (series > monotonicity["upper"]).any():
        raise ValueError(f"Values larger than upper limit in column {colname}")
    if "lower" in monotonicity and (series < monotonicity["lower"]).any():
        raise ValueError(f"Values smaller than lower limit in column {colname}")


def check_almost_monotone(series: pd.Series, digits: int, sign: int) -> None:
    """Raise a ValueError if a series is not sufficiently close
    to constant or monotone in a certain direction.

    Args:
        series: Vector of numbers
        digits:
        sign: direction. >0 means positive
    """
    if isinstance(series, (list, np.ndarray)):
        series = pd.Series(series, dtype="float64")

    allowance = 1.0 / 10.0 ** (digits - 1)
    if sign > 0:
        if series.diff().min() < -allowance:  # type: ignore[operator]
            raise ValueError("Series is not almost monotone")
    elif series.diff().max() > allowance:  # type: ignore[operator]
        raise ValueError("Series is not almost monotone")


def validate_monotonicity_arg(
    monotonicity: dict[str, MonotonicitySpec], dframe_colnames: list[str]
) -> None:
    """
    Validate a dictionary with monotonicity arguments that
    can be given to df2str().

    Will raise ValueError exceptions if anything is wrong.

    Args:
        monotonicity: Keys are column names.
        dframe_colnames: Names of column names
            in dframes. Used in error messages.
    """
    valid_keys = ["sign", "upper", "lower", "allowzero"]
    if monotonicity is None:
        return
    if not isinstance(monotonicity, dict):
        raise ValueError("monotonicity argument must be a dict")
    for col, spec in monotonicity.items():
        if not isinstance(spec, dict):
            raise ValueError("monotonicity argument must be a dict of dicts")
        if not set(spec.keys()).issubset(valid_keys):
            raise ValueError(f"Unknown keys in monotonicity {spec.keys()}")
        if col not in dframe_colnames:
            raise ValueError(f"Column {col} does not exist in dataframe")
        if "sign" not in spec:
            raise ValueError(f"Monotonocity sign not specified for {col}")
        try:
            signvalue = float(spec["sign"])
        except ValueError as err:
            raise ValueError(f"Monotonocity sign {spec['sign']} not valid") from err
        if "upper" in spec:
            float(spec["upper"])
        if "lower" in spec:
            float(spec["lower"])
        if abs(signvalue) > 1:
            raise ValueError("Monotonocity sign must be -1 or +1, not larger/smaller")

        if "allowzero" in spec and spec["allowzero"] not in {
            True,
            False,
        }:
            raise ValueError("allowzero in monotonicity argument must be True/False")
