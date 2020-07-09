"""Test module for the WaterOil object"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from hypothesis import given, settings
import hypothesis.strategies as st

from pyscal import WaterOil
from pyscal.constants import SWINTEGERS

from common import (
    check_table,
    float_df_checker,
    sat_table_str_ok,
    check_linear_sections,
)


def check_endpoints(wateroil, krwend, krwmax, kroend):
    """Check that the code produces correct endpoints for
    parametrizations, on discrete cases"""
    swtol = 1 / SWINTEGERS

    # Check endpoints for oil curve:
    # krow at swcr should be kroend:
    if wateroil.swcr > wateroil.swl + swtol:
        assert float_df_checker(wateroil.table, "son", 1.0, "krow", kroend)
    # krow at sorw should be zero:
    assert float_df_checker(wateroil.table, "son", 0.0, "krow", 0.0)
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


@given(st.text())
def test_wateroil_tag(tag):
    """Test that we are unlikely to crash Eclipse
    by having ugly tag names"""
    wateroil = WaterOil(h=0.5, tag=tag)
    wateroil.add_corey_oil()
    wateroil.add_corey_water()
    sat_table_str_ok(wateroil.SWOF())
    sat_table_str_ok(wateroil.SWFN())


@settings(deadline=1000)
@given(st.floats(), st.floats())
def test_wateroil_corey1(nw, now):
    """Test random corey parameters"""
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
    check_linear_sections(wateroil)
    swofstr = wateroil.SWOF()
    assert len(swofstr) > 100


@settings(deadline=1000)
@given(st.floats(), st.floats(), st.floats(), st.floats(), st.floats())
def test_wateroil_let1(l, e, t, krwend, krwmax):
    """Test random LET parameters"""
    wateroil = WaterOil()
    try:
        wateroil.add_LET_oil(l, e, t, krwend)
        wateroil.add_LET_water(l, e, t, krwend, krwmax)
    except AssertionError:
        # This happens for negative values f.ex.
        return
    assert "krow" in wateroil.table
    assert "krw" in wateroil.table
    assert isinstance(wateroil.krwcomment, str)
    check_table(wateroil.table)
    check_linear_sections(wateroil)
    swofstr = wateroil.SWOF()
    assert len(swofstr) > 100


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
def test_wateroil_krendmax(swl, swcr, sorw, kroend, krwend, krwmax, h, fast):
    """Test endpoints for wateroil using hypothesis testing"""
    try:
        wateroil = WaterOil(swl=swl, swcr=swcr, sorw=sorw, h=h, fast=fast)
    except AssertionError:
        return
    krwend = min(krwend, krwmax)
    wateroil.add_corey_oil(kroend=kroend)
    wateroil.add_corey_water(krwend=krwend, krwmax=krwmax)
    check_table(wateroil.table)
    assert wateroil.selfcheck()
    assert 0 < wateroil.crosspoint() < 1

    check_endpoints(wateroil, krwend, krwmax, kroend)
    ####################################
    # Do it over again, but with LET:
    wateroil.add_LET_oil(t=1.1, kroend=kroend)
    wateroil.add_LET_water(t=1.1, krwend=krwend, krwmax=krwmax)
    assert wateroil.selfcheck()
    check_table(wateroil.table)
    # Check endpoints for oil curve:
    check_endpoints(wateroil, krwend, krwmax, kroend)
    check_linear_sections(wateroil)
    assert 0 < wateroil.crosspoint() < 1


def test_swfn():
    """Test that we can dump SWFN without giving oil relperm"""
    wateroil = WaterOil(h=0.1)
    wateroil.add_corey_water()
    swfnstr = wateroil.SWFN()
    assert "SWFN" in swfnstr
    assert len(swfnstr) > 15


def test_linearsegments():
    """Made for testing the linear segments during
    the resolution of issue #163"""
    wateroil = WaterOil(h=0.01, swl=0.1, swcr=0.3, sorw=0.3)
    wateroil.add_corey_oil(now=10, kroend=0.5)
    wateroil.add_corey_water(nw=10, krwend=0.5)
    check_table(wateroil.table)
    check_linear_sections(wateroil)
    # wateroil.plotkrwkrow(marker="*")


def test_wateroil_linear():
    """Test linear wateroil curves"""
    wateroil = WaterOil(h=1)
    wateroil.add_corey_water()
    wateroil.add_corey_oil()
    swofstr = wateroil.SWOF(header=False)
    check_table(wateroil.table)
    check_linear_sections(wateroil)
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
    check_linear_sections(wateroil)
    assert len(wateroil.table) == 2
    assert np.isclose(wateroil.table["sw"].min(), 0.1)
    assert np.isclose(wateroil.table["sw"].max(), 1.0)
    assert np.isclose(wateroil.crosspoint(), 0.55)


def test_comments():
    """Test that the outputters include endpoints in comments"""
    wateroil = WaterOil(h=0.3)
    wateroil.add_corey_water()
    wateroil.add_corey_oil()
    swfn = wateroil.SWFN()
    assert "--" in swfn
    assert "pyscal: " in swfn  # part of version string
    assert "swirr=0" in swfn
    assert "swcr=0" in swfn
    assert "swl=0" in swfn
    assert "sorw=0" in swfn
    assert "nw=2" in swfn
    assert "krwend=1" in swfn
    assert "Corey" in swfn
    assert "krw = krow @ sw=0.5" in swfn
    assert "Zero capillary pressure" in swfn
    assert "SW" in swfn
    assert "KRW" in swfn
    assert "KROW" not in swfn
    assert "PC" in swfn

    swof = wateroil.SWOF()
    assert "--" in swof
    assert "pyscal: " in swof  # part of version string
    assert "swirr=0" in swof
    assert "swcr=0" in swof
    assert "swl=0" in swof
    assert "sorw=0" in swof
    assert "nw=2" in swof
    assert "now=2" in swof
    assert "krwend=1" in swof
    assert "Corey" in swof
    assert "krw = krow @ sw=0.5" in swof
    assert "Zero capillary pressure" in swof
    assert "SW" in swof
    assert "KRW" in swof
    assert "KROW" in swof
    assert "PC" in swof
