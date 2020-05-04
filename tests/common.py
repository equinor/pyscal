"""Common functions and mock data for usage in pyscal testing"""

import numpy as np


def series_decreasing(series):
    """Weaker than pd.Series.is_monotonic_decreasing,
    allows constant parts.

    We do not enforce less than zero here, because there will be
    some positive differences due to representation errors which
    is ok in terms of numerical equivalence. It is not ok when dumped
    to Eclipse, so the representation of the dataframe as a SGOF table
    must be properly rounded before printed."""
    return (series.diff().dropna() < 1e-8).all()


def series_increasing(series):
    """Weaker than pd.Series.is_monotonic_increasing"""
    return (series.diff().dropna() > -1e-8).all()


def sat_table_str_ok(sat_table_str):
    """Test that a supplied string from SWOF()/SGOF() etc is
    probably ok for Eclipse.

    This is checking that the first characters on each line is sensible.

    Wrap this function with assert

    Returns:
        True if tests pass.
    """
    if not sat_table_str:
        return False
    for line in sat_table_str.splitlines():
        try:
            if not (
                not line
                or line.startswith("S")
                or line.startswith("--")
                or line.startswith("/")
                or int(line[0]) >= 0
            ):
                return False
        except ValueError as e_msg:
            # the int(line[0]) will get here on strings.
            print(e_msg)
            return False
    if "-- pyscal: " not in sat_table_str:
        # This should hold the version
        return False
    return True


def check_table(dframe):
    """Check that the numbers in a dataframe for WaterOil or GasOil
    has the properties that Eclipse enforces"""
    assert not dframe.empty
    assert not dframe.isnull().values.any()
    if "sw" in dframe:
        assert len(dframe["sw"].unique()) == len(dframe)
        assert dframe["sw"].is_monotonic
        assert (dframe["sw"] >= 0.0).all()
        assert dframe["swn"].is_monotonic
        assert dframe["son"].is_monotonic_decreasing
        assert dframe["swnpc"].is_monotonic
    if "sg" in dframe:
        assert len(dframe["sg"].unique()) == len(dframe)
        assert dframe["sg"].is_monotonic
        assert (dframe["sg"] >= 0.0).all()
        assert dframe["sgn"].is_monotonic
        assert dframe["son"].is_monotonic_decreasing
    if "krow" in dframe:
        assert series_decreasing(dframe["krow"])
        assert (dframe["krow"] >= 0).all()
        assert (dframe["krow"] <= 1.0).all()
    if "krw" in dframe:
        assert series_increasing(dframe["krw"])
        assert np.isclose(dframe["krw"].iloc[0], 0.0)
        assert (dframe["krw"] >= 0).all()
        assert (dframe["krw"] <= 1.0).all()
    if "pc" in dframe:
        assert series_decreasing(dframe["pc"])
    if "krog" in dframe:
        assert series_decreasing(dframe["krog"])
        assert (dframe["krog"] >= 0).all()
        assert (dframe["krog"] <= 1.0).all()
    if "krg" in dframe:
        assert series_increasing(dframe["krg"])
        assert (dframe["krg"] >= 0).all()
        assert (dframe["krg"] <= 1.0).all()


def float_df_checker(dframe, idxcol, value, compcol, answer):
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
