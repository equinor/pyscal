"""Test module for the WaterOilGas object"""

import os
from pathlib import Path

import pytest

from pyscal import GasOil, WaterOil, WaterOilGas
from pyscal.utils.testing import sat_table_str_ok

try:
    import opm.io

    HAVE_OPM = True
except ImportError:
    HAVE_OPM = False


def test_wateroilgas_constructor():
    """Test that constructor properties are available as aattributes"""
    wog = WaterOilGas(swirr=0.01, swl=0.02, sorg=0.03, sorw=0.04, tag="Foo")
    assert wog.swirr == 0.01
    assert wog.swl == 0.02
    assert wog.sorg == 0.03
    assert wog.tag == "Foo"
    assert wog.sorw == 0.04

    # Manipulate the tag in the underlying GasOil object:
    wog.gasoil.tag = "Bar"
    assert wog.tag == "Foo Bar"  # Different tags are concatenated.


def test_wateroilgas_simple():
    """Test that default curves will give valid include strings"""
    wog = WaterOilGas()

    # Add default curves:
    wog.wateroil.add_corey_water()
    wog.wateroil.add_corey_oil()
    wog.gasoil.add_corey_gas()
    wog.gasoil.add_corey_oil()

    with pytest.raises(AssertionError):
        # Testing test code:
        sat_table_str_ok("")
    sat_table_str_ok(wog.SWOF())
    sat_table_str_ok(wog.SGOF())
    sat_table_str_ok(wog.SLGOF())
    sat_table_str_ok(wog.SOF3())
    sat_table_str_ok(wog.SGFN())
    sat_table_str_ok(wog.SWFN())


def test_threephasecheck():
    """Test three phase consistency checks"""
    wog = WaterOilGas()
    assert not wog.selfcheck()
    wog.wateroil.add_corey_water(nw=2)
    wog.wateroil.add_corey_oil(now=2, kroend=0.9)
    wog.gasoil.add_corey_gas(ng=2)
    wog.gasoil.add_corey_oil(nog=2, kroend=1)
    assert not wog.threephaseconsistency()


def test_empty():
    """Empty object should give empty strings (and logged errors)"""
    wog = WaterOilGas()
    assert wog.SWOF() == ""
    assert wog.SGOF() == ""
    assert wog.SOF3() == ""
    assert wog.SLGOF() == ""
    assert wog.SWFN() == ""
    assert wog.SGFN() == ""


def test_manipulated_attributes_none():
    """It is allowed to manipulate the WaterOilGas by setting the wateroil
    and/or the gasoil attributes to None, this is performed in
    SCALrecommendation.interpolate() for example. Test this behaviour."""
    # pylint: disable=pointless-statement  # The pointless statements trigger errors

    wog_go = WaterOilGas(tag="gasoilonly")
    # Make it into a two-phase object:
    wog_go.wateroil = None
    wog_go.gasoil.add_corey_gas()
    wog_go.gasoil.add_corey_oil()
    assert wog_go.SWFN() == ""
    assert wog_go.SWOF() == ""
    assert wog_go.SOF3() == ""
    with pytest.raises(ValueError, match="wateroil is None in WaterOilGas"):
        wog_go.swirr
    with pytest.raises(ValueError, match="wateroil is None in WaterOilGas"):
        wog_go.swl
    with pytest.raises(ValueError, match="wateroil is None in WaterOilGas"):
        wog_go.sorw
    assert wog_go.tag == "gasoilonly"
    assert wog_go.threephaseconsistency() is True

    wog_wo = WaterOilGas(tag="wateroilonly")
    wog_wo.gasoil = None
    wog_wo.wateroil.add_corey_water()
    wog_wo.wateroil.add_corey_oil()
    assert wog_wo.SLGOF() == ""
    assert wog_wo.SGFN() == ""
    assert wog_wo.SOF3() == ""
    assert wog_wo.SGOF() == ""
    with pytest.raises(ValueError, match="gasoil is None in WaterOilGas"):
        wog_wo.sorg
    assert wog_wo.tag == "wateroilonly"
    assert wog_wo.threephaseconsistency() is True

    wog_nones = WaterOilGas()
    wog_nones.wateroil = None
    wog_nones.gasoil = None
    wog_nones.tag == ""
    assert wog_nones.threephaseconsistency() is True
    assert wog_nones.SWOF() == ""
    assert wog_nones.SGOF() == ""
    assert wog_nones.selfcheck() is False
    with pytest.raises(ValueError, match="wateroil is None"):
        wog_nones.swirr


def test_not_threephase_consistency():
    """Mock a WaterOilGas object that fails threephase consistency"""
    wog = WaterOilGas()
    # To trigger this, we need to hack the WaterOilGas object
    # by overriding the effect of its __init__
    wog.wateroil = WaterOil(swl=0.4)
    wog.gasoil = GasOil(swl=0.2)
    wog.wateroil.add_corey_water(nw=2)
    wog.wateroil.add_corey_oil(now=2, kroend=0.9)
    wog.gasoil.add_corey_gas(ng=2)
    wog.gasoil.add_corey_oil(nog=2, kroend=1)
    assert not wog.threephaseconsistency()


@pytest.mark.skipif(not HAVE_OPM, reason="ecl2df not installed")
def test_parse_with_opm(tmp_path):
    """Test that the SWOF+SGOF output from pyscal can be
    injected into a valid Eclipse deck"""
    wog = WaterOilGas()
    wog.wateroil.add_corey_water(nw=2)
    wog.wateroil.add_corey_oil(now=2, kroend=0.9)
    wog.gasoil.add_corey_gas(ng=2)
    wog.gasoil.add_corey_oil(nog=2, kroend=1)

    ecldeck = (
        """RUNSPEC
DIMENS
  1 1 1 /
OIL
WATER
GAS
START
  1 'JAN' 2100 /
TABDIMS
   2* 10000 /
EQLDIMS
  1 /
GRID
DX
   10 /
DY
   10 /
DZ
   50 /
TOPS
   1000 /
PORO
   0.3 /
PERMX
   100 /
PERMY
   100 /
PERMZ
   100 /

PROPS

"""
        + wog.SWOF()
        + wog.SGOF()
        + """
DENSITY
  800 1000 1.2 /
PVTW
  1 1 0.0001 0.2 0.00001 /
PVDO
   100 1   1
   150 0.9 1 /
PVDG
   100 1 1
   150 0.9 1 /
ROCK
  100 0.0001 /
SOLUTION
EQUIL
   1000    100     1040    0   1010      0 /"""
    )

    os.chdir(tmp_path)
    Path("RELPERMTEST.DATA").write_text(ecldeck, encoding="utf8")
    deck = opm.io.Parser().parse("RELPERMTEST.DATA")
    assert "SWOF" in deck
    assert "SGOF" in deck
