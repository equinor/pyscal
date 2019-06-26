# -*- coding: utf-8 -*-
"""Test module for relperm"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyscal import WaterOil


def check_table(df):
    assert not df.empty
    assert not df.isnull().values.any()
    assert len(df["sw"].unique()) == len(df)
    assert df["sw"].is_monotonic
    assert (df["sw"] >= 0.0).all()
    assert df["swn"].is_monotonic
    assert df["son"].is_monotonic_decreasing
    assert df["swnpc"].is_monotonic
    assert df["pc"].is_monotonic_decreasing


def test_simple_J():
    wo = WaterOil(swl=0.01)
    wo.add_simple_J()  # swl set to zero will give infinite pc
    check_table(wo.table)
    assert wo.pccomment

    # Zero gravity:
    wo.add_simple_J(g=0)
    assert wo.table.pc.unique() == 0.0

    # This should give back Sw:
    # This ensures that density and gravity scaling is correct
    wo.add_simple_J(a=1, b=1, poro_ref=1, perm_ref=1, drho=1000, g=100)
    assert (wo.table["pc"] - wo.table["sw"]).sum() < 0.00001
    # (check_table() will fail on this, when b > 0)
