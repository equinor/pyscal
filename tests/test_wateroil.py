# -*- coding: utf-8 -*-
"""Test module for relperm"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from hypothesis import given, settings
import hypothesis.strategies as st

from pyscal import WaterOil


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
    assert np.isclose(wo.table['sw'].min(), 0.1)
    assert np.isclose(wo.table['sw'].max(), 1.0)
    assert np.isclose(wo.crosspoint(), 0.55)
