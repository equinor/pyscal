# -*- coding: utf-8 -*-
"""Test module for relperm"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

from hypothesis import given, settings
import hypothesis.strategies as st

from pyscal import WaterOil, constants


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
    assert df["krow"].is_monotonic_decreasing
    assert df["krw"].is_monotonic


def test_dict_init():
    from test_scalrecommendation import base_sample_corey

    wo = WaterOil(**base_sample_corey)
    assert wo.swl == base_sample_corey["swl"]
    assert wo.sorw == base_sample_corey["sorw"]
    assert wo.swirr == base_sample_corey["swirr"]

    # Should not crash:
    WaterOil(bogusarg="foobar", **base_sample_corey)

    # But we can't override:
    with pytest.raises(TypeError):
        WaterOil(swl=0.4, **base_sample_corey)

    wo.add_corey_oil(**base_sample_corey)
    wo.add_corey_water(**base_sample_corey)
    assert wo.selfcheck()


def test_h():
    from test_scalrecommendation import base_sample_corey as base

    wo = WaterOil(h=0.9, **base)
    assert len(wo.table) == 3  # Due to endpoints.

    h_max = 1 - base["swl"] - base["sorw"]
    wo = WaterOil(h=h_max, **base)
    assert len(wo.table) == 3

    # Slightly lower 'h' gives us an intermediate point, very unevenly spaced though
    wo = WaterOil(h=h_max - 1 / constants.SWINTEGERS, **base)
    assert len(wo.table) == 4

    # But due to SWINTEGERS, too small change in h will be ignored
    wo = WaterOil(h=h_max - 0.1 / constants.SWINTEGERS, **base)
    assert len(wo.table) == 3

    # Still 4 rows when h = h_max/2
    wo = WaterOil(h=h_max / 2, **base)
    assert len(wo.table) == 4

    # But 5 rows when slightly lower
    wo = WaterOil(h=h_max / 2 - 1 / constants.SWINTEGERS, **base)
    assert len(wo.table) == 5


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
