# -*- coding: utf-8 -*-
"""Test module for relperm"""

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


def check_table(df):
    """Check sanity of important columns"""
    assert not df.empty
    assert not df.isnull().values.any()
    assert len(df["sw"].unique()) == len(df)
    assert df["sw"].is_monotonic
    assert (df["sw"] >= 0.0).all()
    assert df["swn"].is_monotonic
    assert df["son"].is_monotonic_decreasing
    assert df["swnpc"].is_monotonic
    assert series_decreasing(df["krow"])
    assert series_increasing(df["krw"])
    assert np.isclose(df["krw"].iloc[0], 0.0)
    assert (0 <= df["krw"]).all()
    assert (df["krw"] <= 1.0).all()
    assert (0 <= df["krow"]).all()
    assert (df["krow"] <= 1.0).all()


@settings(deadline=1000)
@given(st.floats(), st.floats())
def test_wateroil_corey1(nw, now):
    wo = WaterOil()
    try:
        wo.add_corey_oil(now=now)
        wo.add_corey_water(nw=nw)
    except AssertionError:
        # This happens for "invalid" input
        return

    assert "krow" in wo.table
    assert "krw" in wo.table
    assert isinstance(wo.krwcomment, str)
    check_table(wo.table)
    swofstr = wo.SWOF()
    assert len(swofstr) > 100


@settings(deadline=1000)
@given(st.floats(), st.floats(), st.floats(), st.floats(), st.floats())
def test_wateroil_let1(l, e, t, krwend, krwmax):
    wo = WaterOil()
    try:
        wo.add_LET_oil(l, e, t, krwend, krwmax)
        wo.add_LET_water(l, e, t, krwend, krwmax)
    except AssertionError:
        # This happens for negative values f.ex.
        return
    assert "krow" in wo.table
    assert "krw" in wo.table
    assert isinstance(wo.krwcomment, str)
    check_table(wo.table)
    swofstr = wo.SWOF()
    assert len(swofstr) > 100


@settings(max_examples=1000, deadline=500)
@given(
    st.floats(min_value=0, max_value=0.3),
    st.floats(min_value=0, max_value=0.3),
    st.floats(min_value=0, max_value=0.3),
    st.floats(min_value=0.1, max_value=1),
    st.floats(min_value=0.1, max_value=1),
    st.floats(min_value=0.1, max_value=1),
    st.floats(min_value=0.1, max_value=1),
    st.floats(min_value=0.0001, max_value=1),
)
def test_wateroil_krendmax(swl, swcr, sorw, kroend, kromax, krwend, krwmax, h):
    try:
        wo = WaterOil(swl=swl, swcr=swcr, sorw=sorw, h=h)
    except AssertionError:
        return
    kroend = min(kroend, kromax)
    krwend = min(krwend, krwmax)
    wo.add_corey_oil(kroend=kroend, kromax=kromax)
    wo.add_corey_water(krwend=krwend, krwmax=krwmax)
    check_table(wo.table)
    assert wo.selfcheck()

    check_endpoints(wo, krwend, krwmax, kroend, kromax)
    ####################################
    # Do it over again, but with LET:
    wo.add_LET_oil(kroend=kroend, kromax=kromax)
    wo.add_LET_water(krwend=krwend, krwmax=krwmax)
    assert wo.selfcheck()
    check_table(wo.table)
    # Check endpoints for oil curve:
    check_endpoints(wo, krwend, krwmax, kroend, kromax)


def float_df_checker(df, idxcol, value, compcol, answer):
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
        idxs = (df[idxcol] - value).abs() < swtol
        if sum(idxs) < 1:
            continue
        if sum(idxs) < 10:
            break
    rowidx = (df[idxs][idxcol] - value).abs().sort_values().index[0]
    # print(rowidx)
    # todo only sort values a little bit close..
    return np.isclose(plus_one + df.loc[rowidx, compcol], plus_one + answer)


def check_endpoints(wo, krwend, krwmax, kroend, kromax):
    swtol = 1 / SWINTEGERS

    # Check endpoints for oil curve:
    # krow at swcr should be kroend:
    if wo.swcr > wo.swl + swtol:
        assert float_df_checker(wo.table, "son", 1.0, "krow", kroend)
    # krow at sorw should be zero:
    assert float_df_checker(wo.table, "son", 0.0, "krow", 0.0)
    if wo.swcr > wo.swl + max(wo.h, swtol):
        # krow at swl should be kromax:
        assert float_df_checker(wo.table, "sw", wo.swl, "krow", kromax)
    else:
        # kromax not used when swcr is close to swl.
        assert np.isclose(wo.table["krow"].max(), kroend)

    # Check endpoints for water curve: (np.isclose is only reliable around 1)
    assert float_df_checker(wo.table, "swn", 0.0, "krw", 0.0)
    assert float_df_checker(wo.table, "sw", wo.swcr, "krw", 0)

    if wo.sorw > swtol:
        # (hard to get it right when sorw is less than h and close to zero)
        assert float_df_checker(wo.table, "sw", 1 - wo.sorw, "krw", krwend)
        assert np.isclose(wo.table["krw"].max(), krwmax)
    else:
        assert np.isclose(wo.table["krw"].max(), krwend)


def test_wateroil_linear():
    wo = WaterOil(h=1)
    wo.add_corey_water()
    wo.add_corey_oil()
    swofstr = wo.SWOF(header=False)
    check_table(wo.table)
    assert isinstance(swofstr, str)
    assert swofstr
    assert len(wo.table) == 2
    assert np.isclose(wo.crosspoint(), 0.5)

    # What if there is no space for our choice of h?
    # We should be able to initialize nonetheless
    # (a warning could be given)
    wo = WaterOil(swl=0.1, h=1)
    wo.add_corey_water()
    wo.add_corey_oil()
    check_table(wo.table)
    assert len(wo.table) == 2
    assert np.isclose(wo.table["sw"].min(), 0.1)
    assert np.isclose(wo.table["sw"].max(), 1.0)
    assert np.isclose(wo.crosspoint(), 0.55)
