# -*- coding: utf-8 -*-
"""Test module for relperm"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from hypothesis import given, settings
import hypothesis.strategies as st

from pyscal import WaterOil
from pyscal.constants import EPSILON, MAX_EXPONENT


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

    # Some values seen in real life:
    wo.add_simple_J(a=100, b=-1.5, poro_ref=0.12, perm_ref=100, drho=200)
    check_table(wo.table)
    assert "Simplified" in wo.pccomment
    assert "a=100" in wo.pccomment
    assert "b=-1.5" in wo.pccomment
    wo.add_corey_oil()
    wo.add_corey_water()
    swof = wo.SWOF()
    assert isinstance(swof, str)
    assert swof


@given(
    st.floats(min_value=0.001, max_value=1000000),
    st.floats(min_value=-0.9 * MAX_EXPONENT, max_value=-0.001),
    st.floats(min_value=0.01, max_value=0.5),
    st.floats(min_value=0.001, max_value=10),
    st.floats(min_value=0.01, max_value=1000000),
    st.floats(min_value=0.001, max_value=10000000),
)
def test_simple_J_random(a, b, poro_ref, perm_ref, drho, g):
    """Test different J-function parameters.

    Parameter ranges tested through hypothesis are limited, as not
    every number is realistic. Way outside the tested intervals, you
    can get AssertionErrors or the capillary pressure may not be
    monotonically decreasing within machine precision.
    """
    wo = WaterOil(swl=0.01)
    wo.add_simple_J(a=a, b=b, poro_ref=poro_ref, perm_ref=perm_ref, drho=drho, g=g)
    check_table(wo.table)
