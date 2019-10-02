# -*- coding: utf-8 -*-
"""Test module for relperm"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from hypothesis import given, settings
import hypothesis.strategies as st

from pyscal import WaterOil
from pyscal.constants import SWINTEGERS

from test_wateroil import float_df_checker


def check_table(df):
    assert not df.empty
    assert not df.isnull().values.any()
    assert len(df["sw"].unique()) == len(df)
    assert df["sw"].is_monotonic
    assert (df["sw"] >= 0.0).all()
    assert df["swn"].is_monotonic
    assert df["son"].is_monotonic_decreasing
    assert df["swnpc"].is_monotonic


@given(
    st.floats(),
    st.floats(),
    st.floats(),
    st.floats(),
    st.floats(min_value=-0.1, max_value=2),
    st.text(),
)
def test_wateroil_random(swirr, swl, swcr, sorw, h, tag):
    """Shoot wildly with arguments, the code should throw ValueError
    or AssertionError when input is invalid, but we don't want other crashes"""
    try:
        WaterOil(swirr=swirr, swl=swl, swcr=swcr, sorw=sorw, h=h, tag=tag)
    except AssertionError:
        pass


@settings(max_examples=500)
@given(
    st.floats(min_value=0, max_value=0.1),
    st.floats(min_value=0, max_value=0.15),
    st.floats(min_value=0, max_value=0.3),
    st.floats(min_value=0, max_value=0.05),
    st.floats(min_value=0.01, max_value=0.2),
    st.text(),
)
def test_wateroil_normalization(swirr, swl, swcr, sorw, h, tag):
    """Shoot with more realistic values and test normalized saturations"""
    wo = WaterOil(swirr=swirr, swl=swl, swcr=swcr, sorw=sorw, h=h, tag=tag)
    assert not wo.table.empty
    assert not wo.table.isnull().values.any()

    # Check that son is 1 at swcr:
    assert float_df_checker(wo.table, "sw", wo.swcr, "son", 1)
    # Check that son is 0 at sorw:
    if wo.sorw > h:
        assert float_df_checker(wo.table, "sw", 1 - wo.sorw, "son", 0)

    # Check that swn is 0 at swcr:
    assert float_df_checker(wo.table, "sw", wo.swcr, "swn", 0)
    # Check that swn is 1 at 1 - sorw
    if wo.sorw > 1 / SWINTEGERS:
        assert float_df_checker(wo.table, "sw", 1 - wo.sorw, "swn", 1)

    # Check that swnpc is 0 at swirr and 1 at 1:
    if wo.swirr >= wo.swl + h:
        assert float_df_checker(wo.table, "sw", wo.swirr, "swnpc", 0)
    else:
        # Let this go, when swirr is too close to swl. We
        # are not guaranteed to have sw=swirr present
        pass

    assert float_df_checker(wo.table, "sw", 1.0, "swnpc", 1)


@given(st.floats(min_value=0, max_value=1))
def test_wateroil_swir(swirr):
    wo = WaterOil(swirr=swirr)
    check_table(wo.table)


@given(st.floats(min_value=0, max_value=1))
def test_wateroil_swl(swl):
    wo = WaterOil(swl=swl)
    check_table(wo.table)


@given(st.floats(min_value=0, max_value=1))
def test_wateroil_swcr(swcr):
    wo = WaterOil(swcr=swcr)
    check_table(wo.table)


@given(st.floats(min_value=0, max_value=1))
def test_wateroil_sorw(sorw):
    wo = WaterOil(sorw=sorw)
    check_table(wo.table)


# Test combination of 2 floats as parameters:
@settings(deadline=1000)
@given(st.floats(min_value=0, max_value=1), st.floats(min_value=0, max_value=1))
def test_wateroil_dual(p1, p2):
    try:
        wo = WaterOil(swl=p1, sorw=p2)
        check_table(wo.table)
        # Will fail when swl > 1 - sorw
    except AssertionError:
        pass

    try:
        wo = WaterOil(swl=p1, swirr=p2)
        check_table(wo.table)
    except AssertionError:
        pass

    try:
        wo = WaterOil(swcr=p1, sorw=p2)
        check_table(wo.table)
    except AssertionError:
        pass

    try:
        wo = WaterOil(swirr=p1, sorw=p2)
        check_table(wo.table)
    except AssertionError:
        pass
