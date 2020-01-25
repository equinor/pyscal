"""Test module for the WaterOil object"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from hypothesis import given, settings
import hypothesis.strategies as st

from pyscal import WaterOil
from pyscal.constants import SWINTEGERS


def series_decreasing(series):
    """Weaker than pd.Series.is_monotonic_decreasing,
    allows constant parts"""
    return (series.diff().dropna() < 1e-8).all()


def series_increasing(series):
    """Weaker than pd.Series.is_monotonic_increasing"""
    return (series.diff().dropna() > -1e-8).all()


def check_table(dframe):
    """Check sanity of important columns"""
    assert not dframe.empty
    assert not dframe.isnull().values.any()
    assert len(dframe["sw"].unique()) == len(dframe)
    assert dframe["sw"].is_monotonic
    assert (dframe["sw"] >= 0.0).all()
    assert dframe["swn"].is_monotonic
    assert dframe["son"].is_monotonic_decreasing
    assert dframe["swnpc"].is_monotonic
    assert series_decreasing(dframe["krow"])
    assert series_increasing(dframe["krw"])
    assert np.isclose(dframe["krw"].iloc[0], 0.0)
    assert (dframe["krw"] >= 0).all()
    assert (dframe["krw"] <= 1.0).all()
    assert (dframe["krow"] >= 0).all()
    assert (dframe["krow"] <= 1.0).all()


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
    # print(rowidx)
    # todo only sort values a little bit close..
    return np.isclose(plus_one + dframe.loc[rowidx, compcol], plus_one + answer)


def check_endpoints(wateroil, krwend, krwmax, kroend, kromax):
    """Ccheck that the code produces correct endpoints for
    parametrizations, on discrete cases"""
    swtol = 1 / SWINTEGERS

    # Check endpoints for oil curve:
    # krow at swcr should be kroend:
    if wateroil.swcr > wateroil.swl + swtol:
        assert float_df_checker(wateroil.table, "son", 1.0, "krow", kroend)
    # krow at sorw should be zero:
    assert float_df_checker(wateroil.table, "son", 0.0, "krow", 0.0)
    if wateroil.swcr > wateroil.swl + swtol:
        # krow at swl should be kromax:
        assert float_df_checker(wateroil.table, "sw", wateroil.swl, "krow", kromax)
    else:
        # kromax not used when swcr is close to swl.
        assert np.isclose(wateroil.table["krow"].max(), kroend)

    # Check endpoints for water curve: (np.isclose is only reliable around 1)
    assert float_df_checker(wateroil.table, "swn", 0.0, "krw", 0.0)
    assert float_df_checker(wateroil.table, "sw", wateroil.swcr, "krw", 0)

    if wateroil.sorw > swtol:
        # (hard to get it right when sorw is less than h and close to zero)
        assert float_df_checker(wateroil.table, "sw", 1 - wateroil.sorw, "krw", krwend)
        assert np.isclose(wateroil.table["krw"].max(), krwmax)
    else:
        assert np.isclose(wateroil.table["krw"].max(), krwend)


@settings(deadline=1000)
@given(st.floats(), st.floats())
def test_wateroil_corey1(nw, now):
    wateroil = WaterOil()
    try:
        wateroil.add_corey_oil(now=now)
        wateroil.add_corey_water(nw=nw)
    except AssertionError:
        # This happens for "invalid" input
        return

    assert "krow" in wateroil.table
    assert "krw" in wateroil.table
    assert isinstance(wateroil.krwcomment, str)
    check_table(wateroil.table)
    swofstr = wateroil.SWOF()
    assert len(swofstr) > 100


@settings(deadline=1000)
@given(st.floats(), st.floats(), st.floats(), st.floats(), st.floats())
def test_wateroil_let1(l, e, t, krwend, krwmax):
    wateroil = WaterOil()
    try:
        wateroil.add_LET_oil(l, e, t, krwend, krwmax)
        wateroil.add_LET_water(l, e, t, krwend, krwmax)
    except AssertionError:
        # This happens for negative values f.ex.
        return
    assert "krow" in wateroil.table
    assert "krw" in wateroil.table
    assert isinstance(wateroil.krwcomment, str)
    check_table(wateroil.table)
    swofstr = wateroil.SWOF()
    assert len(swofstr) > 100


@settings(max_examples=100, deadline=500)
@given(
    st.floats(min_value=0, max_value=0.3),
    st.floats(min_value=0, max_value=0.3),
    st.floats(min_value=0, max_value=0.3),
    st.floats(min_value=0.1, max_value=1),
    st.floats(min_value=0.1, max_value=1),
    st.floats(min_value=0.1, max_value=1),
    st.floats(min_value=0.1, max_value=1),
    st.floats(min_value=0.0001, max_value=1),
    st.booleans(),
)
def test_wateroil_krendmax(swl, swcr, sorw, kroend, kromax, krwend, krwmax, h, fast):
    try:
        wateroil = WaterOil(swl=swl, swcr=swcr, sorw=sorw, h=h, fast=fast)
    except AssertionError:
        return
    kroend = min(kroend, kromax)
    krwend = min(krwend, krwmax)
    wateroil.add_corey_oil(kroend=kroend, kromax=kromax)
    wateroil.add_corey_water(krwend=krwend, krwmax=krwmax)
    check_table(wateroil.table)
    assert wateroil.selfcheck()
    assert 0 < wateroil.crosspoint() < 1

    check_endpoints(wateroil, krwend, krwmax, kroend, kromax)
    ####################################
    # Do it over again, but with LET:
    wateroil.add_LET_oil(t=1.1, kroend=kroend, kromax=kromax)
    wateroil.add_LET_water(t=1.1, krwend=krwend, krwmax=krwmax)
    assert wateroil.selfcheck()
    check_table(wateroil.table)
    # Check endpoints for oil curve:
    check_endpoints(wateroil, krwend, krwmax, kroend, kromax)
    assert 0 < wateroil.crosspoint() < 1


def test_wateroil_linear():
    """Test linear wateroil curves"""
    wateroil = WaterOil(h=1)
    wateroil.add_corey_water()
    wateroil.add_corey_oil()
    swofstr = wateroil.SWOF(header=False)
    check_table(wateroil.table)
    assert isinstance(swofstr, str)
    assert swofstr
    assert len(wateroil.table) == 2
    assert np.isclose(wateroil.crosspoint(), 0.5)

    # What if there is no space for our choice of h?
    # We should be able to initialize nonetheless
    # (a warning could be given)
    wateroil = WaterOil(swl=0.1, h=1)
    wateroil.add_corey_water()
    wateroil.add_corey_oil()
    check_table(wateroil.table)
    assert len(wateroil.table) == 2
    assert np.isclose(wateroil.table["sw"].min(), 0.1)
    assert np.isclose(wateroil.table["sw"].max(), 1.0)
    assert np.isclose(wateroil.crosspoint(), 0.55)
