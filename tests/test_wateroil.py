"""Test module for the WaterOil object"""
import io
import sys

import hypothesis.strategies as st
import matplotlib
import matplotlib.pyplot
import numpy as np
import pandas as pd
import pytest
from hypothesis import given

from pyscal import WaterOil
from pyscal.constants import SWINTEGERS
from pyscal.utils.testing import (
    check_linear_sections,
    check_table,
    float_df_checker,
    sat_table_str_ok,
)


def check_endpoints(wateroil, krwend, krwmax, kroend):
    """Check that the code produces correct endpoints for
    parametrizations, on discrete cases"""
    swtol = 1 / SWINTEGERS

    # Check endpoints for oil curve:
    assert float_df_checker(wateroil.table, "SW", wateroil.swl, "KROW", kroend)
    assert float_df_checker(wateroil.table, "SON", 1, "KROW", kroend)
    assert np.isclose(wateroil.table["KROW"].max(), kroend)
    assert float_df_checker(wateroil.table, "SON", 0.0, "KROW", 0.0)
    assert float_df_checker(wateroil.table, "SW", 1 - wateroil.socr, "KROW", 0.0)

    # Check endpoints for water curve: (np.isclose is only reliable around 1)
    assert float_df_checker(wateroil.table, "SWN", 0.0, "KRW", 0.0)
    assert float_df_checker(wateroil.table, "SW", wateroil.swcr, "KRW", 0)

    if wateroil.sorw > swtol:
        # (hard to get it right when sorw is less than h and close to zero)
        assert float_df_checker(wateroil.table, "SW", 1 - wateroil.sorw, "KRW", krwend)
        assert np.isclose(wateroil.table["KRW"].max(), krwmax)
    else:
        assert np.isclose(wateroil.table["KRW"].max(), krwend)


@given(st.text())
def test_wateroil_tag(tag):
    """Test that we are unlikely to crash Eclipse
    by having ugly tag names"""
    wateroil = WaterOil(h=0.5, tag=tag)
    wateroil.add_corey_oil()
    wateroil.add_corey_water()
    sat_table_str_ok(wateroil.SWOF())
    sat_table_str_ok(wateroil.SWFN())


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

    assert "KROW" in wateroil.table
    assert "KRW" in wateroil.table
    assert isinstance(wateroil.krwcomment, str)
    check_table(wateroil.table)
    check_linear_sections(wateroil)
    swofstr = wateroil.SWOF()
    assert len(swofstr) > 100


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
    assert "KROW" in wateroil.table
    assert "KRW" in wateroil.table
    assert isinstance(wateroil.krwcomment, str)
    check_table(wateroil.table)
    check_linear_sections(wateroil)
    swofstr = wateroil.SWOF()
    assert len(swofstr) > 100


@given(
    st.floats(min_value=0, max_value=0.4),
    st.floats(min_value=0, max_value=0.4),
    st.floats(min_value=0, max_value=0.4),
    st.floats(min_value=0, max_value=0.1),
    st.floats(min_value=0.1, max_value=1),
    st.floats(min_value=0.1, max_value=1),
    st.floats(min_value=0.1, max_value=1),
    st.floats(min_value=0.001, max_value=1),
    st.booleans(),
)
def test_wateroil_krendmax(swl, swcr, sorw, socr_add, kroend, krwend, krwmax, h, fast):
    """Test endpoints for wateroil using hypothesis testing"""
    try:
        wateroil = WaterOil(
            swl=swl, swcr=swcr, socr=sorw + socr_add, sorw=sorw, h=h, fast=fast
        )
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


def test_fast():
    """Test the fast option"""
    # First without fast:
    wateroil = WaterOil(h=0.1)
    wateroil.add_corey_water()
    wateroil.add_corey_oil()
    # This crosspoint computation is only present for fast=False:
    assert "-- krw = krow @ sw=0.5" in wateroil.SWOF()

    # Provoke non-strict-monotone krow:
    wateroil.table.loc[0:2, "KROW"] = [1.00, 0.81, 0.81]
    # (this is valid in non-imbibition, but pyscal will correct it for all
    # curves)
    assert "0.1000000 0.0100000 0.8100000 0.0000000" in wateroil.SWOF()
    assert "0.2000000 0.0400000 0.8099999 0.0000000" in wateroil.SWOF()
    #   monotonicity correction:   ^^^^^^

    # Now redo with fast option:
    wateroil = WaterOil(h=0.1, fast=True)
    wateroil.add_corey_water()
    wateroil.add_corey_oil()
    # This crosspoint computation is only present for fast=False:
    assert "-- krw = krow" not in wateroil.SWOF()

    # Provoke non-strict-monotone krow, in fast-mode
    # this slips through:
    wateroil.table.loc[0:2, "KROW"] = [1.00, 0.81, 0.81]
    assert "0.1000000 0.0100000 0.8100000 0.0000000" in wateroil.SWOF()
    assert "0.2000000 0.0400000 0.8100000 0.0000000" in wateroil.SWOF()
    # not corrected:               ^^^^^^

    wateroil.table.loc[0:2, "KRW"] = [0.00, 0.01, 0.01]
    assert "0.1000000 0.0100000" in wateroil.SWFN()
    assert "0.2000000 0.0100000" in wateroil.SWFN()


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


@pytest.mark.skipif(sys.platform == "win32", reason="Plotting not stable on windows")
def test_plotting(mocker):
    """Test that plotting code pass through (nothing displayed)"""
    mocker.patch("matplotlib.pyplot.show", return_value=None)
    wateroil = WaterOil(swl=0.1, h=0.1)
    wateroil.add_corey_water()
    wateroil.add_corey_oil()
    wateroil.plotkrwkrow(mpl_ax=matplotlib.pyplot.subplots()[1])
    wateroil.plotkrwkrow(logyscale=True, mpl_ax=matplotlib.pyplot.subplots()[1])
    wateroil.plotkrwkrow(mpl_ax=None)

    wateroil.add_simple_J()
    wateroil.plotpc(mpl_ax=matplotlib.pyplot.subplots()[1])
    wateroil.plotpc(mpl_ax=None)
    wateroil.plotpc(logyscale=True)


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
    assert np.isclose(wateroil.table["SW"].min(), 0.1)
    assert np.isclose(wateroil.table["SW"].max(), 1.0)
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
    assert "socr=0" not in swfn  # Only included when nonzero
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
    assert "socr=0" not in swof  # Only included when nonzero
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

    paleo_wo = WaterOil(h=0.3, socr=0.1)
    paleo_wo.add_corey_water()
    paleo_wo.add_corey_oil()
    assert "socr=0.1" in paleo_wo.SWFN()
    assert "socr=0.1" in paleo_wo.SWOF()


def test_nexus():
    """Test the Nexus export"""
    wateroil = WaterOil(h=0.01, swl=0.1, swcr=0.3, sorw=0.3)
    wateroil.add_corey_oil(now=10, kroend=0.5)
    wateroil.add_corey_water(nw=10, krwend=0.5)
    nexus_lines = wateroil.WOTABLE().splitlines()
    no_comments = [line for line in nexus_lines if not line.startswith("!") or not line]
    assert no_comments[0] == "WOTABLE"
    assert no_comments[1] == "SW KRW KROW PC"
    df = pd.read_table(
        io.StringIO("\n".join(no_comments[2:])),
        engine="python",
        sep=r"\s+",
        header=None,
    )
    # pylint: disable=no-member  # false positive on Pandas dataframe
    assert (df.values <= 1.0).all()
    assert (df.values >= 0.0).all()


@pytest.mark.parametrize(
    "columnname, errorvalues",
    [
        ("SW", [1, 0]),
        ("SW", [0, 2]),
        ("KRW", [1, 0]),
        ("KRW", [0, 2]),
        ("KROW", [0, 1]),
        ("KROW", [2, 0]),
        ("PC", [np.inf, 0]),
        ("PC", [np.nan, 0]),
        ("PC", [1, 2]),
        ("PC", [0, 1]),
    ],
)
def test_selfcheck(columnname, errorvalues):
    """Test the selfcheck feature of a WaterOil object"""
    wateroil = WaterOil(h=1)
    wateroil.add_corey_water()
    wateroil.add_corey_oil()
    assert wateroil.selfcheck()

    # Punch the internal table directly to trigger error:
    wateroil.table[columnname] = errorvalues
    assert not wateroil.selfcheck()
    assert wateroil.SWOF() == ""
    if not columnname == "KROW":
        assert wateroil.SWFN() == ""
