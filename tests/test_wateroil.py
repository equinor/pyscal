# -*- coding: utf-8 -*-
"""Test module for relperm"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from hypothesis import given, settings
import hypothesis.strategies as st

from pyscal import WaterOil

# Avoid some erroneous Flaky-test-reports
settings(deadline=1000)


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
