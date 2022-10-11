"""Common functions and mock data for usage in pyscal testing"""

import io
from typing import Union

import numpy as np
import pandas as pd

try:
    from hypothesis import settings
except ModuleNotFoundError:
    # This module is only needed for testing, but this testing module
    # should be importable in other scenarios as well.
    pass

from pyscal import GasOil, WaterOil


def slow_hypothesis(slow_function):
    """A decorator for tests that are always slow.

    The default deadline for hypothesis is set through the hypothesis profile
    being used, and is set to None for CI runs (through a command line argument
    and code in conftest.py), meaning no deadline. On tests that are slow on
    local iron as well, we must ensure that the deadline set for these do not
    override the "no-deadline" in CI.
    """
    if settings().deadline is None:
        return settings()(slow_function)
    return settings(deadline=1000)(slow_function)


def series_decreasing(series: pd.Series):
    """Weaker than pd.Series.is_monotonic_decreasing,
    allows constant parts.

    We do not enforce less than zero here, because there will be
    some positive differences due to representation errors which
    is ok in terms of numerical equivalence. It is not ok when dumped
    to Eclipse, so the representation of the dataframe as a SGOF table
    must be properly rounded before printed."""
    return (series.diff().dropna() < 1e-8).all()


def series_increasing(series: pd.Series):
    """Weaker than pd.Series.is_monotonic_increasing"""
    return (series.diff().dropna() > -1e-8).all()


def sat_table_str_ok(sat_table_str: str) -> None:
    """Test that a supplied string from SWOF()/SGOF() etc is
    probably ok for Eclipse.

    Number of floats pr. line must be constant
    All numerical lines must be parseable to a rectangular dataframe
    with only floats. The first column must contain only unique values
    for every SATNUM.
    """
    assert sat_table_str

    for line in sat_table_str.splitlines():
        try:
            if not (
                not line
                or line.startswith("S")
                or line.startswith("--")
                or line.startswith("/")
                or int(line[0]) >= 0
            ):
                assert False

        except ValueError as e_msg:
            # the int(line[0]) will get here on strings.
            print(e_msg)
            assert False

    assert "-- pyscal: " in sat_table_str

    # On non-comment lines, number of ascii floats should be the same:
    number_lines = [
        line
        for line in sat_table_str.splitlines()
        if line.strip() and line.strip()[0] in ["0", "1", "."]
    ]

    floats_pr_line = {len(line.split()) for line in number_lines}
    # This must be a constant:
    assert len(floats_pr_line) == 1
    # And not more than 4:
    if not list(floats_pr_line)[0] <= 4:
        print(sat_table_str)
    assert list(floats_pr_line)[0] <= 4

    float_characters = {len(flt) for flt in " ".join(number_lines).split()}
    digits = 7  # This is the default value in utils.df2str()
    for float_str_length in float_characters:
        assert not 1 < float_str_length < digits + 2
        # float_str_length must be 1 (a pure zero value),
        # or above digits + 1, otherwise it is a sign of some error.

    # And pyscal only emits three or four floats pr. line for all keywords:
    assert list(set(floats_pr_line))[0] in [3, 4]

    # So we should be able to parse this to a dataframe:
    dframe = pd.read_csv(io.StringIO("\n".join(number_lines)), sep=" ", header=None)
    assert len(dframe) == len(number_lines)

    # The first column holds saturations, for pyscal test-data that
    # is always between zero and 1
    assert 0 <= dframe[0].min() <= dframe[0].max() <= 1

    # Saturations should be unique, but only within each SATNUM.
    # Assert this by checking that the two consecutive numbers in the
    # first column are never the same:
    assert (~np.isclose(dframe[0].diff().dropna(), 0)).all()

    # Second column is never capillary pressure, so there we can enforce the same
    assert 0 <= dframe[1].min() <= dframe[1].max() <= 1
    # And then sometimes for the third column:
    if len(dframe.columns) > 3 or "SOF3" in sat_table_str:
        assert 0 <= dframe[2].min() <= dframe[2].max() <= 1


def check_table(dframe: pd.DataFrame) -> None:
    """Check that the numbers in a dataframe for WaterOil or GasOil
    has the properties that Eclipse enforces"""
    assert not dframe.empty
    assert not dframe.isnull().values.any()
    if "SW" in dframe and "SG" not in dframe:
        # (avoiding GasWater tables, where sw is an auxiliary column)
        assert len(dframe["SW"].unique()) == len(dframe)
        assert dframe["SW"].is_monotonic_increasing
        assert (dframe["SW"] >= 0.0).all()
        assert dframe["SWN"].is_monotonic_increasing
        assert dframe["SON"].is_monotonic_decreasing
        assert dframe["SWNPC"].is_monotonic_increasing
    if "SG" in dframe:
        assert len(dframe["SG"].unique()) == len(dframe)
        assert dframe["SG"].is_monotonic_increasing
        assert (dframe["SG"] >= 0.0).all()
        assert dframe["SGN"].is_monotonic_increasing
        assert dframe["SON"].is_monotonic_decreasing
    if "KROW" in dframe:
        assert series_decreasing(dframe["KROW"])
        assert (dframe["KROW"] >= 0).all()
        assert (dframe["KROW"] <= 1.0).all()
    if "KRW" in dframe:
        assert series_increasing(dframe["KRW"])
        assert np.isclose(dframe["KRW"].iloc[0], 0.0)
        assert (dframe["KRW"] >= 0).all()
        assert (dframe["KRW"] <= 1.0).all()
    if "PC" in dframe:
        if "SW" in dframe:
            assert series_decreasing(dframe["PC"])
        if "SG" in dframe:
            assert series_increasing(dframe["PC"])
    if "KROG" in dframe:
        assert series_decreasing(dframe["KROG"])
        assert (dframe["KROG"] >= 0).all()
        assert (dframe["KROG"] <= 1.0).all()
    if "KRG" in dframe:
        assert series_increasing(dframe["KRG"])
        assert (dframe["KRG"] >= 0).all()
        assert (dframe["KRG"] <= 1.0).all()


def check_linear_sections(wo_or_go: Union[WaterOil, GasOil]) -> None:
    """Check that the linear segments of a WaterOil or a GasOil
    object are linear."""
    if isinstance(wo_or_go, WaterOil):
        sat_col = "SW"
        right_start = 1.0 - wo_or_go.sorw
        right_end = 1.0
        right_lin_cols = ["KROW", "KRW"]
        left_lin_cols = ["KRW"]
        left_start = wo_or_go.swl
        left_end = wo_or_go.swcr
    if isinstance(wo_or_go, GasOil):
        sat_col = "SG"
        if wo_or_go.krgendanchor == "sorg":
            right_start = 1.0 - wo_or_go.sorg - wo_or_go.swl
        else:
            # If not krgendanchor=sorg, then there is no linear
            # segment to the right.
            right_start = 1.0 - wo_or_go.swl
        right_end = 1.0 - wo_or_go.swl
        right_lin_cols = ["KROG", "KRG"]
        left_lin_cols = ["KRG"]
        left_start = 0.0
        left_end = wo_or_go.sgcr

    right_lin_seg = wo_or_go.table[
        (wo_or_go.table[sat_col] >= right_start)
        & (wo_or_go.table[sat_col] <= right_end)
    ]
    left_lin_seg = wo_or_go.table[
        (wo_or_go.table[sat_col] >= left_start) & (wo_or_go.table[sat_col] <= left_end)
    ]
    if len(right_lin_seg) > 5:
        for col in right_lin_cols:
            # We avoid the first and two lasts row in right_lin_seg, because
            # this does not always match the constant saturation segment
            # assumption in this linearity test:
            assert right_lin_seg.iloc[1:-2][col].diff().std() < 1e-9
    if len(left_lin_seg) > 4:
        for col in left_lin_cols:
            assert left_lin_seg.iloc[1:-1][col].diff().std() < 1e-9


def float_df_checker(
    dframe: pd.DataFrame, idxcol: str, value: float, compcol: str, answer: float
) -> Union[np.bool_, np.ndarray]:
    """Looks up in a dataframe, selects the row where idxcol=value
    and compares the value in compcol with answer

    Warning: This is slow code, but only the tests are slow

    Floats are notoriously difficult to handle in computers.
    """
    # Find row index where we should do comparison:
    plus_one = 0
    if abs(answer) < 0.2:
        plus_one = 1
    for swtol in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]:
        idxs = (dframe[idxcol] - value).abs() < swtol
        if sum(idxs) < 1:
            continue
        if sum(idxs) < 10:
            break
    rowidx = (dframe[idxs][idxcol] - value).abs().sort_values().index[0]
    return np.isclose(plus_one + dframe.loc[rowidx, compcol], plus_one + answer)
