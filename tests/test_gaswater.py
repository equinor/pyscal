"""Test module for the GasWater object"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from hypothesis import given, settings
import hypothesis.strategies as st

from pyscal import GasWater
from pyscal.constants import SWINTEGERS

from common import (
    check_table,
    float_df_checker,
    sat_table_str_ok,
    check_linear_sections,
)


def check_endpoints(gaswater, krwend, krwmax, krgend):
    """Check that the code produces correct endpoints for
    parametrizations, on discrete cases"""
    swtol = 1 / SWINTEGERS

    # Check endpoints for gas curve:
    # krg at swl should be krgend:
    assert float_df_checker(gaswater.gasoil.table, "sgn", 1.0, "krg", krgend)
    assert float_df_checker(gaswater.gasoil.table, "sl", gaswater.swl, "krg", krgend)
    # krg at sgcr (sgn is zero there) should be zero:
    assert float_df_checker(gaswater.gasoil.table, "sgn", 0.0, "krg", 0.0)
    assert float_df_checker(gaswater.gasoil.table, "sl", 1 - gaswater.sgcr, "krg", 0.0)

    check_linear_sections(gaswater.gasoil)
    check_linear_sections(gaswater.wateroil)

    # Check endpoints for water curve: (np.isclose is only reliable around 1)
    assert float_df_checker(gaswater.wateroil.table, "swn", 0.0, "krw", 0.0)
    assert float_df_checker(gaswater.wateroil.table, "sw", gaswater.swcr, "krw", 0)

    if gaswater.sgrw > swtol:
        # (hard to get it right when sgrw is less than h and close to zero)
        assert float_df_checker(
            gaswater.wateroil.table, "sw", 1 - gaswater.sgrw, "krw", krwend
        )
        assert np.isclose(gaswater.wateroil.table["krw"].max(), krwmax)
    else:
        assert np.isclose(gaswater.wateroil.table["krw"].max(), krwend)


@given(st.text())
def test_gaswater_tag(tag):
    """Test that we are unlikely to crash Eclipse
    by having ugly tag names"""
    gaswater = GasWater(h=0.5, tag=tag)
    gaswater.add_corey_gas()
    gaswater.add_corey_water()
    sat_table_str_ok(gaswater.SWFN())
    sat_table_str_ok(gaswater.SGFN())


@settings(deadline=1000)
@given(st.floats(), st.floats())
def test_gaswater_corey1(nw, ng):
    """Test random corey parameters"""
    gaswater = GasWater()
    try:
        gaswater.add_corey_gas(ng=ng)
        gaswater.add_corey_water(nw=nw)
    except AssertionError:
        # This happens for "invalid" input
        return

    assert "krg" in gaswater.gasoil.table
    assert "krw" in gaswater.wateroil.table
    assert isinstance(gaswater.wateroil.krwcomment, str)
    check_table(gaswater.wateroil.table)
    check_table(gaswater.gasoil.table)
    check_linear_sections(gaswater.wateroil)
    check_linear_sections(gaswater.gasoil)
    swfnstr = gaswater.SWFN()
    assert len(swfnstr) > 100
    sgfnstr = gaswater.SGFN()
    assert len(sgfnstr) > 100


@settings(deadline=1000)
@given(st.floats(), st.floats(), st.floats(), st.floats(), st.floats())
def test_gaswater_let1(l, e, t, krwend, krwmax):
    """Test random LET parameters"""
    gaswater = GasWater()
    try:
        gaswater.add_LET_gas(l, e, t, krwend)
        gaswater.add_LET_water(l, e, t, krwend, krwmax)
    except AssertionError:
        # This happens for negative values f.ex.
        return
    assert "krg" in gaswater.gasoil.table
    assert "krw" in gaswater.wateroil.table
    assert isinstance(gaswater.wateroil.krwcomment, str)
    check_table(gaswater.wateroil.table)
    check_table(gaswater.gasoil.table)
    check_linear_sections(gaswater.wateroil)
    check_linear_sections(gaswater.gasoil)
    swfnstr = gaswater.SWFN()
    assert len(swfnstr) > 100
    sgfnstr = gaswater.SGFN()
    assert len(sgfnstr) > 100


@settings(deadline=500)
@given(
    st.floats(min_value=0, max_value=0.3),
    st.floats(min_value=0, max_value=0.3),
    st.floats(min_value=0, max_value=0.3),
    st.floats(min_value=0.1, max_value=1),
    st.floats(min_value=0.1, max_value=1),
    st.floats(min_value=0.1, max_value=1),
    st.floats(min_value=0.0001, max_value=1),
    st.booleans(),
)
def test_gaswater_krendmax(swl, swcr, sgrw, krgend, krwend, krwmax, h, fast):
    """Test endpoints for gaswater using hypothesis testing"""
    try:
        gaswater = GasWater(swl=swl, swcr=swcr, sgrw=sgrw, h=h, fast=fast)
    except AssertionError:
        return
    krwend = min(krwend, krwmax)
    gaswater.add_corey_gas(krgend=krgend)
    gaswater.add_corey_water(krwend=krwend, krwmax=krwmax)
    check_table(gaswater.wateroil.table)
    check_table(gaswater.gasoil.table)
    assert gaswater.selfcheck()
    # assert 0 < gaswater.crosspoint() < 1

    check_endpoints(gaswater, krwend, krwmax, krgend)
    ####################################
    # Do it over again, but with LET:
    gaswater.add_LET_gas(t=1.1, krgend=krgend)
    gaswater.add_LET_water(t=1.1, krwend=krwend, krwmax=krwmax)
    assert gaswater.selfcheck()
    check_table(gaswater.gasoil.table)
    check_table(gaswater.wateroil.table)
    # Check endpoints for oil curve:
    check_endpoints(gaswater, krwend, krwmax, krgend)
    check_linear_sections(gaswater.wateroil)
    check_linear_sections(gaswater.gasoil)
    # assert 0 < gaswater.crosspoint() < 1


def test_swfn():
    """Test that we can dump SWFN without giving gas relperm"""
    gaswater = GasWater(h=0.1)
    gaswater.add_corey_water()
    swfnstr = gaswater.SWFN()
    assert "SWFN" in swfnstr
    assert len(swfnstr) > 15


def test_linearsegments():
    """Made for testing the linear segments during
    the resolution of issue #163"""
    gaswater = GasWater(h=0.01, swl=0.1, swcr=0.3, sgrw=0.3)
    gaswater.add_corey_gas(ng=10, krgend=0.5)
    gaswater.add_corey_water(nw=10, krwend=0.5)
    check_table(gaswater.gasoil.table)
    check_table(gaswater.wateroil.table)
    check_linear_sections(gaswater.wateroil)
    check_linear_sections(gaswater.gasoil)


def test_gaswater_linear():
    """Test linear gaswater curves (linear as in giving only
    two saturation points to Eclipse)"""
    gaswater = GasWater(h=1)
    gaswater.add_corey_water()
    gaswater.add_corey_gas()
    swfnstr = gaswater.SWFN(header=False)
    check_table(gaswater.wateroil.table)
    check_table(gaswater.gasoil.table)
    check_linear_sections(gaswater.wateroil)
    check_linear_sections(gaswater.gasoil)
    assert isinstance(swfnstr, str)
    assert swfnstr
    assert len(gaswater.wateroil.table) == 2
    # assert np.isclose(gaswater.crosspoint(), 0.5)

    # What if there is no space for our choice of h?
    # We should be able to initialize nonetheless
    # (a warning could be given)
    gaswater = GasWater(swl=0.1, h=1)
    gaswater.add_corey_water()
    gaswater.add_corey_gas()
    check_table(gaswater.wateroil.table)
    check_table(gaswater.gasoil.table)
    check_linear_sections(gaswater.wateroil)
    check_linear_sections(gaswater.gasoil)
    assert len(gaswater.wateroil.table) == 2
    assert len(gaswater.gasoil.table) == 2
    assert np.isclose(gaswater.wateroil.table["sw"].min(), 0.1)
    assert np.isclose(gaswater.wateroil.table["sw"].max(), 1.0)
    # assert np.isclose(gaswater.crosspoint(), 0.55)


def test_crosspoint():
    """Test the crosspoint computation (on edge cases)"""
    gaswater = GasWater(swl=0.0, sgrw=0.0, sgcr=0.0, h=0.1)
    gaswater.add_corey_water(nw=1)
    gaswater.add_corey_gas(ng=1)
    assert np.isclose(gaswater.crosspoint(), 0.5)

    gaswater = GasWater(swl=0.5, sgrw=0.0, sgcr=0.0, h=0.1)
    gaswater.add_corey_water(nw=1)
    gaswater.add_corey_gas(ng=1)
    assert np.isclose(gaswater.crosspoint(), 0.75)

    gaswater = GasWater(swl=0.0, sgrw=0.5, sgcr=0.0, h=0.1)
    gaswater.add_corey_water(nw=1)
    gaswater.add_corey_gas(ng=1)
    assert np.isclose(gaswater.crosspoint(), 0.3333333)

    gaswater = GasWater(swl=0.0, sgrw=0.5, sgcr=0.5, h=0.1)
    gaswater.add_corey_water(nw=1)
    gaswater.add_corey_gas(ng=1)
    assert np.isclose(gaswater.crosspoint(), 0.25)

    gaswater = GasWater(swl=0.0, sgrw=0.5, sgcr=0.5, h=0.1)
    gaswater.add_corey_water(nw=1, krwend=0.5)
    gaswater.add_corey_gas(ng=1, krgend=0.5)
    assert np.isclose(gaswater.crosspoint(), 0.25)


def test_gaswater_pc():
    """Test that capillary pressure can be added to GasWater.

    The GasWater object is calling up the code in WaterOil, which is
    tested more thorougly, in this test function we need to make
    sure the functionality is in place."""
    gaswater = GasWater(swl=0.1, h=0.2)
    gaswater.add_corey_water()
    gaswater.add_corey_gas()
    gaswater.add_simple_J()
    assert gaswater.wateroil.table["pc"].abs().sum() > 0
    swfn = gaswater.SWFN()
    assert "Simplified J-function" in swfn
    assert "0.1000000 0.0000000 0.23266" in swfn  # this is the first row.
    sat_table_str_ok(swfn)

    sgfn = gaswater.SGFN()
    # Capillary pressure in SGFN must always be zero for GasWater.
    assert "Zero capillary pressure" in sgfn
    sat_table_str_ok(sgfn)

    # Overwrite to zero:
    gaswater.add_simple_J(drho=0)
    swfn = gaswater.SWFN()
    assert "0.1000000 0.0000000 0.0000000" in swfn  # first row
    sat_table_str_ok(sgfn)

    # Petrophysical pc:
    gaswater.add_simple_J_petro(a=1, b=-1)
    swfn = gaswater.SWFN()
    assert "petrophysical version" in swfn
    assert "0.1000000 0.0000000 0.014715" in swfn  # first row


def test_comments():
    """Test that the outputters include endpoints in comments"""
    gaswater = GasWater(h=0.3)
    gaswater.add_corey_water()
    gaswater.add_corey_gas()
    swfn = gaswater.SWFN()
    assert "--" in swfn
    assert "pyscal: " in swfn  # part of version string
    assert "swirr=0" in swfn
    assert "swcr=0" in swfn
    assert "swl=0" in swfn
    assert "sgrw=0" in swfn
    assert "sgcr=0" in swfn
    assert "nw=2" in swfn
    assert "krwend=1" in swfn
    assert "Corey" in swfn
    assert "krw = krg @ sw=0.5" in swfn
    assert "Zero capillary pressure" in swfn
    assert "SW" in swfn
    assert "KRW" in swfn
    assert "KRGW" not in swfn
    assert "PC" in swfn

    sgfn = gaswater.SGFN()
    assert "--" in sgfn
    assert "pyscal: " in sgfn  # part of version string
    assert "swirr=0" in sgfn
    assert "sgcr=0" in sgfn
    assert "swl=0" in sgfn
    assert "ng=2" in sgfn
    assert "krgend=1" in sgfn
    assert "Corey" in sgfn
    assert "krw = krg @ sw=0.5" in sgfn
    assert "Zero capillary pressure" in sgfn
    assert "SG" in sgfn
    assert "KRW" not in sgfn
    assert "KRG" in sgfn
    assert "PC" in sgfn
