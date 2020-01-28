"""Test module for GasOil objects"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from hypothesis import given, settings
import hypothesis.strategies as st

from common import float_df_checker, check_table, sat_table_str_ok

from pyscal import GasOil
from pyscal.constants import SWINTEGERS


def test_gasoil_init():
    """Check the __init__ method for GasOil

    are arguments handled correctly?"""
    gasoil = GasOil()
    assert isinstance(gasoil, GasOil)
    assert gasoil.swirr == 0.0
    assert gasoil.swl == 0.0
    assert gasoil.krgendanchor == ""  # Because sorg is zero

    gasoil = GasOil(swl=0.1)
    assert gasoil.swirr == 0.0
    assert gasoil.swl == 0.1

    gasoil = GasOil(swirr=0.1)
    assert gasoil.swirr == 0.1
    assert gasoil.swl == 0.1  # This one is zero by default, but will follow swirr.
    assert gasoil.sorg == 0.0
    assert gasoil.sgcr == 0.0

    gasoil = GasOil(tag="foobar")
    assert gasoil.tag == "foobar"

    # This will print a warning, but will be the same as ""
    gasoil = GasOil(krgendanchor="bogus")
    assert isinstance(gasoil, GasOil)
    assert gasoil.krgendanchor == ""

    # Test with h=1
    gasoil = GasOil(h=1)
    gasoil.add_corey_gas()
    gasoil.add_corey_oil()
    assert np.isclose(gasoil.crosspoint(), 0.5)
    assert len(gasoil.table) == 2

    gasoil = GasOil(swl=0.1, h=1)
    gasoil.add_corey_gas()
    gasoil.add_corey_oil()
    assert len(gasoil.table) == 2
    assert np.isclose(gasoil.crosspoint(), 0.45)
    assert np.isclose(gasoil.table["sg"].min(), 0)
    assert np.isclose(gasoil.table["sg"].max(), 0.9)


@given(
    st.floats(min_value=0, max_value=0.15),  # swl
    st.floats(min_value=0, max_value=0.3),  # sgcr
    st.floats(min_value=0, max_value=0.05),  # sorg
    st.floats(min_value=0.0001, max_value=0.2),  # h
    st.text(),
)
def test_gasoil_normalization(swl, sgcr, sorg, h, tag):
    """Check that normalization (sgn and son) is correct
    for all possible saturation endpoints"""
    gasoil = GasOil(
        swirr=0.0, swl=swl, sgcr=sgcr, sorg=sorg, h=h, krgendanchor="sorg", tag=tag
    )
    assert not gasoil.table.empty
    assert not gasoil.table.isnull().values.any()

    # Check that son is 1 at sgcr
    assert float_df_checker(gasoil.table, "sg", gasoil.sgcr, "son", 1)

    # Check that son is 0 at sorg with this krgendanchor
    assert float_df_checker(gasoil.table, "sg", 1 - gasoil.sorg - gasoil.swl, "son", 0)

    # Check that sgn is 0 at sgcr
    assert float_df_checker(gasoil.table, "sg", gasoil.sgcr, "sgn", 0)

    # Check that sgn is 1 at sorg
    assert float_df_checker(gasoil.table, "sg", 1 - gasoil.sorg - gasoil.swl, "sgn", 1)

    # Redo with different krgendanchor
    gasoil = GasOil(
        swirr=0.0, swl=swl, sgcr=sgcr, sorg=sorg, h=h, krgendanchor="", tag=tag
    )
    assert float_df_checker(gasoil.table, "sg", 1 - gasoil.swl, "sgn", 1)
    assert float_df_checker(gasoil.table, "sg", gasoil.sgcr, "sgn", 0)


@settings(max_examples=100, deadline=500)
@given(
    st.floats(min_value=0, max_value=0.3),  # swl
    st.floats(min_value=0, max_value=0.3),  # sgcr
    st.floats(min_value=0, max_value=0.4),  # sorg
    st.floats(min_value=0.1, max_value=1),  # kroend
    st.floats(min_value=0.1, max_value=1),  # kromax
    st.floats(min_value=0.1, max_value=1),  # krgend
    st.floats(min_value=0.2, max_value=1),  # krgmax
    st.floats(min_value=0.0001, max_value=1),  # h
    st.booleans(),  # fast mode
)
def test_gasoil_krendmax(swl, sgcr, sorg, kroend, kromax, krgend, krgmax, h, fast):
    """Test that krendmax gets correct in all numerical corner cases"""
    try:
        gasoil = GasOil(swl=swl, sgcr=sgcr, sorg=sorg, h=h, tag="", fast=fast)
    except AssertionError:
        return
    kroend = min(kroend, kromax)
    krgend = min(krgend, krgmax)
    gasoil.add_corey_oil(kroend=kroend, kromax=kromax)
    gasoil.add_corey_gas(krgend=krgend, krgmax=krgmax)
    check_table(gasoil.table)
    assert gasoil.selfcheck()
    check_endpoints(gasoil, krgend, krgmax, kroend, kromax)
    assert 0 < gasoil.crosspoint() < 1

    # Redo with krgendanchor not defaulted
    gasoil = GasOil(swl=swl, sgcr=sgcr, sorg=sorg, h=h, krgendanchor="", tag="")
    gasoil.add_corey_oil(kroend=kroend, kromax=kromax)
    gasoil.add_corey_gas(krgend=krgend, krgmax=krgmax)
    check_table(gasoil.table)
    assert gasoil.selfcheck()
    check_endpoints(gasoil, krgend, krgmax, kroend, kromax)
    assert 0 < gasoil.crosspoint() < 1

    # Redo with LET:
    gasoil = GasOil(swl=swl, sgcr=sgcr, sorg=sorg, h=h, tag="")
    gasoil.add_LET_oil(t=1.1, kroend=kroend, kromax=kromax)
    gasoil.add_LET_gas(krgend=krgend, krgmax=krgmax)
    check_table(gasoil.table)
    assert gasoil.selfcheck()
    check_endpoints(gasoil, krgend, krgmax, kroend, kromax)
    assert 0 < gasoil.crosspoint() < 1


def check_endpoints(gasoil, krgend, krgmax, kroend, kromax):
    """Discrete tests that endpoints get numerically correct"""
    swtol = 1 / SWINTEGERS

    # Check endpoints for oil curve:
    # krog at sgcr should be kroend
    if gasoil.sgcr > swtol:
        assert float_df_checker(gasoil.table, "son", 1.0, "krog", kroend)
    # krog at son=0 (1-sorg-swl or 1 - swl) should be zero:
    assert float_df_checker(gasoil.table, "son", 0.0, "krog", 0)

    if gasoil.sgcr > swtol:
        assert float_df_checker(gasoil.table, "sg", 0, "krog", kromax)
        assert float_df_checker(gasoil.table, "sg", gasoil.sgcr, "krog", kroend)
    else:
        if not np.isclose(gasoil.table["krog"].max(), kroend):
            print(gasoil.table.head())
        assert np.isclose(gasoil.table["krog"].max(), kroend)

    assert float_df_checker(gasoil.table, "sgn", 0.0, "krg", 0)
    assert float_df_checker(gasoil.table, "sg", gasoil.sgcr, "krg", 0)

    # If krgendanchor == "sorg" then krgmax is irrelevant.
    if gasoil.sorg > swtol and gasoil.sorg > gasoil.h and gasoil.krgendanchor == "sorg":
        assert float_df_checker(gasoil.table, "sgn", 1.0, "krg", krgend)
        assert np.isclose(gasoil.table["krg"].max(), krgmax)
    if gasoil.krgendanchor != "sorg":
        assert np.isclose(gasoil.table["krg"].max(), krgend)


def test_gasoil_kromax():
    """Manual test of kromax behaviour"""
    gasoil = GasOil(h=0.1, sgcr=0.1)
    gasoil.add_corey_oil(2, 0.5)  # Default for kromax
    assert float_df_checker(gasoil.table, "sg", 0.0, "krog", 1.0)
    assert float_df_checker(gasoil.table, "sg", 0.1, "krog", 0.5)
    gasoil.add_corey_oil(2, 0.5, 0.7)
    assert float_df_checker(gasoil.table, "sg", 0.0, "krog", 0.7)
    assert float_df_checker(gasoil.table, "sg", 0.1, "krog", 0.5)

    gasoil = GasOil(h=0.1, sgcr=0.0)
    gasoil.add_corey_oil(2, 0.5)
    assert float_df_checker(gasoil.table, "sg", 0.0, "krog", 0.5)
    gasoil.add_corey_oil(2, 0.5, 1)  # A warning will be given
    assert float_df_checker(gasoil.table, "sg", 0.0, "krog", 0.5)


def test_gasoil_kromax_fast():
    """Test some code in fast mode"""
    gasoil = GasOil(h=0.1, sgcr=0.1, fast=True)
    gasoil.add_corey_oil(2, 0.5)  # Default for kromax
    assert float_df_checker(gasoil.table, "sg", 0.0, "krog", 1.0)
    assert float_df_checker(gasoil.table, "sg", 0.1, "krog", 0.5)
    gasoil.add_corey_oil(2, 0.5, 0.7)
    assert float_df_checker(gasoil.table, "sg", 0.0, "krog", 0.7)
    assert float_df_checker(gasoil.table, "sg", 0.1, "krog", 0.5)

    gasoil = GasOil(h=0.1, sgcr=0.0)
    gasoil.add_corey_oil(2, 0.5)
    assert float_df_checker(gasoil.table, "sg", 0.0, "krog", 0.5)
    gasoil.add_corey_oil(2, 0.5, 1)  # A warning will be given
    assert float_df_checker(gasoil.table, "sg", 0.0, "krog", 0.5)


def test_gasoil_krgendanchor():
    """Test behaviour of the krgendanchor"""
    gasoil = GasOil(krgendanchor="sorg", sorg=0.2, h=0.1)
    assert gasoil.sorg
    gasoil.add_corey_gas(ng=1)
    gasoil.add_corey_oil(nog=1)

    # kg should be 1.0 at 1 - sorg due to krgendanchor == "sorg":
    assert (
        gasoil.table[np.isclose(gasoil.table["sg"], 1 - gasoil.sorg)]["krg"].values[0]
        == 1.0
    )
    assert gasoil.table[np.isclose(gasoil.table["sg"], 1.0)]["krg"].values[0] == 1.0

    gasoil = GasOil(krgendanchor="", sorg=0.2, h=0.1)
    assert gasoil.sorg
    gasoil.add_corey_gas(ng=1)
    gasoil.add_corey_oil(nog=1)

    # kg should be < 1 at 1 - sorg due to krgendanchor being ""
    assert (
        gasoil.table[np.isclose(gasoil.table["sg"], 1 - gasoil.sorg)]["krg"].values[0]
        < 1.0
    )
    assert gasoil.table[np.isclose(gasoil.table["sg"], 1.0)]["krg"].values[0] == 1.0
    assert gasoil.selfcheck()
    assert gasoil.crosspoint() > 0

    # Test once more for LET curves:
    gasoil = GasOil(krgendanchor="sorg", sorg=0.2, h=0.1)
    assert gasoil.sorg
    gasoil.add_LET_gas(1, 1, 1.1)
    gasoil.add_LET_oil(1, 1, 1.1)
    assert 0 < gasoil.crosspoint() < 1

    # kg should be 1.0 at 1 - sorg due to krgendanchor == "sorg":
    assert (
        gasoil.table[np.isclose(gasoil.table["sg"], 1 - gasoil.sorg)]["krg"].values[0]
        == 1.0
    )
    assert gasoil.table[np.isclose(gasoil.table["sg"], 1.0)]["krg"].values[0] == 1.0

    gasoil = GasOil(krgendanchor="", sorg=0.2, h=0.1)
    assert gasoil.sorg
    gasoil.add_LET_gas(1, 1, 1.1)
    gasoil.add_LET_oil(1, 1, 1.1)
    assert gasoil.selfcheck()

    # kg should be < 1 at 1 - sorg due to krgendanchor being ""
    assert (
        gasoil.table[np.isclose(gasoil.table["sg"], 1 - gasoil.sorg)]["krg"].values[0]
        < 1.0
    )
    assert gasoil.table[np.isclose(gasoil.table["sg"], 1.0)]["krg"].values[0] == 1.0


def test_kromaxend():
    """Manual testing of kromax and kroend behaviour"""
    gasoil = GasOil(swirr=0.01, sgcr=0.01, h=0.01, swl=0.1, sorg=0.05)
    gasoil.add_LET_gas()
    gasoil.add_LET_oil(2, 2, 2.1)
    assert gasoil.table["krog"].max() == 1
    gasoil.add_LET_oil(2, 2, 2.1, kroend=0.5, kromax=0.9)
    assert gasoil.table["krog"].max() == 0.9
    # Second krog-value should be kroend, values in between will be linearly
    # interpolated in Eclipse
    assert gasoil.table.sort_values("krog")[-2:-1]["krog"].values[0] == 0.5
    assert 0 < gasoil.crosspoint() < 1

    gasoil.add_corey_oil(2)
    assert gasoil.table["krog"].max() == 1
    gasoil.add_corey_oil(nog=2, kroend=0.5, kromax=0.9)
    assert gasoil.table["krog"].max() == 0.9
    assert gasoil.table.sort_values("krog")[-2:-1]["krog"].values[0] == 0.5


@settings(deadline=1000)
@given(st.floats(), st.floats())
def test_gasoil_corey1(ng, nog):
    """Test the Corey formulation for gasoil"""
    gasoil = GasOil()
    try:
        gasoil.add_corey_oil(nog=nog)
        gasoil.add_corey_gas(ng=ng)
    except AssertionError:
        # This happens for "invalid" input
        return

    assert "krog" in gasoil.table
    assert "krg" in gasoil.table
    assert isinstance(gasoil.krgcomment, str)
    check_table(gasoil.table)
    sgofstr = gasoil.SGOF()
    assert len(sgofstr) > 100
    assert sat_table_str_ok(sgofstr)

    gasoil.resetsorg()
    check_table(gasoil.table)
    sgofstr = gasoil.SGOF()
    assert len(sgofstr) > 100
    assert sat_table_str_ok(sgofstr)


@settings(deadline=1000)
@given(st.floats(), st.floats(), st.floats(), st.floats(), st.floats())
def test_gasoil_let1(l, e, t, krgend, krgmax):
    """Test the LET formulation, take 1"""
    gasoil = GasOil()
    try:
        gasoil.add_LET_oil(l, e, t, krgend)
        gasoil.add_LET_gas(l, e, t, krgend, krgmax)
    except AssertionError:
        # This happens for negative values f.ex.
        return
    assert "krog" in gasoil.table
    assert "krg" in gasoil.table
    assert isinstance(gasoil.krgcomment, str)
    check_table(gasoil.table)
    sgofstr = gasoil.SGOF()
    assert len(sgofstr) > 100
    assert sat_table_str_ok(sgofstr)
