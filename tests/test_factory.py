"""Test the PyscalFactory module"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyscal import (
    GasOil,
    GasWater,
    PyscalFactory,
    SCALrecommendation,
    WaterOil,
    WaterOilGas,
    factory,
)
from pyscal.utils.testing import check_table, sat_table_str_ok


def test_factory_wateroil():
    """Test that we can create curves from dictionaries of parameters"""
    pyscal_factory = PyscalFactory()

    # Factory refuses to create incomplete defaulted objects.
    with pytest.raises(ValueError):
        pyscal_factory.create_water_oil()

    with pytest.raises(TypeError):
        # (it must be a dictionary)
        # pylint: disable=unexpected-keyword-arg
        pyscal_factory.create_water_oil(swirr=0.01)  # noqa

    with pytest.raises(TypeError):
        pyscal_factory.create_water_oil(params="swirr 0.01")

    wateroil = pyscal_factory.create_water_oil(
        dict(
            swirr=0.01,
            swl=0.1,
            bogus="foobar",
            tag="Good sand",
            nw=3,
            now=2,
            krwend=0.2,
            krwmax=0.5,
        )
    )
    assert isinstance(wateroil, WaterOil)
    assert wateroil.swirr == 0.01
    assert wateroil.swl == 0.1
    assert wateroil.tag == "Good sand"
    assert "KRW" in wateroil.table
    assert "Corey" in wateroil.krwcomment
    assert wateroil.table["KRW"].max() == 0.2  # Because sorw==0 by default
    check_table(wateroil.table)
    sat_table_str_ok(wateroil.SWOF())
    sat_table_str_ok(wateroil.SWFN())

    wateroil = pyscal_factory.create_water_oil(
        dict(nw=3, now=2, sorw=0.1, krwend=0.2, krwmax=0.5)
    )
    assert isinstance(wateroil, WaterOil)
    assert "KRW" in wateroil.table
    assert "Corey" in wateroil.krwcomment
    assert wateroil.table["KRW"].max() == 0.5
    check_table(wateroil.table)
    sat_table_str_ok(wateroil.SWOF())
    sat_table_str_ok(wateroil.SWFN())

    # Ambiguous works, but we don't guarantee that this results
    # in LET or Corey.
    wateroil = pyscal_factory.create_water_oil(dict(nw=3, Lw=2, Ew=2, Tw=2, now=3))
    assert "KRW" in wateroil.table
    assert "Corey" in wateroil.krwcomment or "LET" in wateroil.krwcomment
    check_table(wateroil.table)
    sat_table_str_ok(wateroil.SWOF())
    sat_table_str_ok(wateroil.SWFN())

    # Mixing Corey and LET
    wateroil = pyscal_factory.create_water_oil(dict(Lw=2, Ew=2, Tw=2, krwend=1, now=4))
    assert isinstance(wateroil, WaterOil)
    assert "KRW" in wateroil.table
    assert wateroil.table["KRW"].max() == 1.0
    assert "LET" in wateroil.krwcomment
    check_table(wateroil.table)
    sat_table_str_ok(wateroil.SWOF())
    sat_table_str_ok(wateroil.SWFN())

    wateroil = pyscal_factory.create_water_oil(
        dict(Lw=2, Ew=2, Tw=2, Low=3, Eow=3, Tow=3, krwend=0.5)
    )
    assert isinstance(wateroil, WaterOil)
    assert "KRW" in wateroil.table
    assert "KROW" in wateroil.table
    assert wateroil.table["KRW"].max() == 0.5
    assert wateroil.table["KROW"].max() == 1
    assert "LET" in wateroil.krwcomment
    assert "LET" in wateroil.krowcomment
    check_table(wateroil.table)
    sat_table_str_ok(wateroil.SWOF())
    sat_table_str_ok(wateroil.SWFN())

    # Add capillary pressure
    wateroil = pyscal_factory.create_water_oil(
        dict(swl=0.1, nw=1, now=1, a=2, b=-1, poro_ref=0.2, perm_ref=100, drho=200)
    )
    assert "PC" in wateroil.table
    assert wateroil.table["PC"].max() > 0.0
    assert "Simplified J" in wateroil.pccomment
    check_table(wateroil.table)
    sat_table_str_ok(wateroil.SWOF())
    sat_table_str_ok(wateroil.SWFN())

    # Test that the optional gravity g is picked up:
    wateroil = pyscal_factory.create_water_oil(
        dict(swl=0.1, nw=1, now=1, a=2, b=-1, poro_ref=0.2, perm_ref=100, drho=200, g=0)
    )
    assert "PC" in wateroil.table
    assert wateroil.table["PC"].max() == 0.0
    check_table(wateroil.table)
    sat_table_str_ok(wateroil.SWOF())
    sat_table_str_ok(wateroil.SWFN())

    # Test petrophysical simple J:
    wateroil = pyscal_factory.create_water_oil(
        dict(
            swl=0.1,
            nw=1,
            now=1,
            a_petro=2,
            b_petro=-1,
            poro_ref=0.2,
            perm_ref=100,
            drho=200,
        )
    )
    assert "PC" in wateroil.table
    assert wateroil.table["PC"].max() > 0.0
    assert "etrophysic" in wateroil.pccomment
    check_table(wateroil.table)
    sat_table_str_ok(wateroil.SWOF())
    sat_table_str_ok(wateroil.SWFN())

    # One pc param missing:
    wateroil = pyscal_factory.create_water_oil(
        dict(swl=0.1, nw=1, now=1, a=2, b=-1, perm_ref=100, drho=200, g=0)
    )
    assert "PC" not in wateroil.table


def test_fast_mode():
    """Test that the fast-flag is passed on to constructed objects

    Each object's own test code tests the actual effects of the fast flag"""
    wateroil = PyscalFactory.create_water_oil({"nw": 2, "now": 2})
    assert not wateroil.fast
    wateroil = PyscalFactory.create_water_oil({"nw": 2, "now": 2}, fast=True)
    assert wateroil.fast

    gasoil = PyscalFactory.create_gas_oil({"ng": 2, "nog": 2})
    assert not gasoil.fast
    gasoil = PyscalFactory.create_gas_oil({"ng": 2, "nog": 2}, fast=True)
    assert gasoil.fast

    gaswater = PyscalFactory.create_gas_water({"nw": 2, "ng": 2})
    assert not gaswater.gasoil.fast
    assert not gaswater.wateroil.fast
    gaswater = PyscalFactory.create_gas_water({"nw": 2, "ng": 2}, fast=True)
    assert gaswater.gasoil.fast
    assert gaswater.wateroil.fast
    assert gaswater.fast

    wateroilgas = PyscalFactory.create_water_oil_gas(
        {"nw": 2, "now": 2, "ng": 2, "nog": 2}, fast=True
    )
    assert wateroilgas.fast
    assert wateroilgas.wateroil.fast
    assert wateroilgas.gasoil.fast

    scalrec = PyscalFactory.create_scal_recommendation(
        {
            "low": {"nw": 2, "now": 2, "ng": 2, "nog": 2},
            "base": {"nw": 2, "now": 2, "ng": 2, "nog": 2},
            "high": {"nw": 2, "now": 2, "ng": 2, "nog": 2},
        },
        fast=True,
    )
    assert scalrec.low.fast
    assert scalrec.base.fast
    assert scalrec.high.fast

    interpolant = scalrec.interpolate(-0.5)
    assert interpolant.fast


def test_init_with_swlheight():
    """With sufficient parameters, swl will be calculated on the fly
    when initializing the WaterOil object"""
    pyscal_factory = PyscalFactory()
    wateroil = pyscal_factory.create_water_oil(
        dict(
            swlheight=200,
            nw=1,
            now=1,
            swirr=0.01,
            a=1,
            b=-2,
            poro_ref=0.2,
            perm_ref=100,
            drho=200,
        )
    )
    assert np.isclose(wateroil.swl, 0.02480395)
    assert "swl=0.024" in wateroil.SWOF()

    with pytest.raises(
        ValueError,
        match="Can't initialize from SWLHEIGHT without sufficient simple-J parameters",
    ):
        # This should fail because capillary pressure parameters are not provided.
        pyscal_factory.create_water_oil(dict(swlheight=200, nw=1, now=1))

    # swcr must be larger than swl:
    with pytest.raises(ValueError, match="lower than computed swl"):
        pyscal_factory.create_water_oil(
            dict(
                swlheight=200,
                nw=1,
                now=1,
                swirr=0.01,
                swcr=0.0101,
                a=1,
                b=-2,
                poro_ref=0.2,
                perm_ref=100,
                drho=200,
            )
        )

    # swlheight must be positive:
    with pytest.raises(ValueError, match="swlheight must be larger than zero"):
        pyscal_factory.create_water_oil(
            dict(
                swlheight=-200,
                nw=1,
                now=1,
                swirr=0.01,
                a=1,
                b=-2,
                poro_ref=0.2,
                perm_ref=100,
                drho=200,
            )
        )

    # If swcr is large enough, it will pass:
    wateroil = pyscal_factory.create_water_oil(
        dict(
            swlheight=200,
            nw=1,
            now=1,
            swirr=0.01,
            swcr=0.3,
            a=1,
            b=-2,
            poro_ref=0.2,
            perm_ref=100,
            drho=200,
        )
    )
    assert wateroil.swcr > wateroil.swl
    assert wateroil.swcr == 0.3
    assert "swcr=0.3" in wateroil.SWOF()

    # Test that GasWater also can be initialized with swlheight:
    gaswater = pyscal_factory.create_gas_water(
        dict(
            swlheight=200,
            nw=1,
            ng=1,
            swirr=0.01,
            swcr=0.3,
            a=1,
            b=-2,
            poro_ref=0.2,
            perm_ref=100,
            drho=200,
        )
    )
    assert "swl=0.024" in gaswater.SWFN()
    assert gaswater.swcr > gaswater.swl
    assert gaswater.swcr == 0.3
    assert "swcr=0.3" in gaswater.SWFN()

    # Test error message for missing swirr when swlheight is asked for:
    with pytest.raises(
        ValueError, match="Can't initialize from SWLHEIGHT without sufficient simple-J"
    ):
        pyscal_factory.create_water_oil(
            dict(
                swlheight=200,
                nw=1,
                now=1,
                a=1,
                b=-2,
                poro_ref=0.2,
                perm_ref=100,
                drho=200,
            )
        )


def test_relative_swcr():
    """swcr can be initialized relative to swl

    Relevant when swl is initialized from swlheight."""
    pyscal_factory = PyscalFactory()

    with pytest.raises(ValueError, match="swl must be provided"):
        pyscal_factory.create_water_oil(dict(swcr_add=0.1, nw=1, now=1, swirr=0.01))
    with pytest.raises(ValueError, match="swcr and swcr_add at the same time"):
        pyscal_factory.create_water_oil(
            dict(swcr_add=0.1, swcr=0.1, swl=0.1, nw=1, now=1, swirr=0.01)
        )
    wateroil = pyscal_factory.create_water_oil(
        dict(swcr_add=0.1, swl=0.1, nw=1, now=1, swirr=0.01)
    )
    assert wateroil.swcr == 0.2

    # Test when relative to swlheight:
    wateroil = pyscal_factory.create_water_oil(
        dict(
            swlheight=200,
            swcr_add=0.01,
            nw=1,
            now=1,
            swirr=0.01,
            a=1,
            b=-2,
            poro_ref=0.2,
            perm_ref=100,
            drho=200,
        )
    )
    assert np.isclose(wateroil.swl, 0.02480395)
    assert np.isclose(wateroil.swcr, 0.02480395 + 0.01)

    gaswater = pyscal_factory.create_gas_water(
        dict(
            swlheight=200,
            nw=1,
            ng=1,
            swirr=0.01,
            swcr_add=0.1,
            a=1,
            b=-2,
            poro_ref=0.2,
            perm_ref=100,
            drho=200,
        )
    )
    assert np.isclose(gaswater.swl, 0.02480395)
    assert np.isclose(gaswater.swcr, 0.02480395 + 0.1)


def test_ambiguity():
    """Test how the factory handles ambiguity between Corey and LET
    parameters"""
    pyscal_factory = PyscalFactory()
    wateroil = pyscal_factory.create_water_oil(
        dict(swl=0.1, nw=10, Lw=1, Ew=1, Tw=1, now=2, h=0.1, no=2)
    )
    # Corey is picked here.
    assert "Corey" in wateroil.krwcomment
    assert "KRW" in wateroil.table


def test_factory_gasoil():
    """Test that we can create curves from dictionaries of parameters"""
    pyscal_factory = PyscalFactory()

    # Factory refuses to create incomplete defaulted objects.
    with pytest.raises(ValueError):
        pyscal_factory.create_gas_oil()

    with pytest.raises(TypeError):
        # (this must be a dictionary)
        # pylint: disable=unexpected-keyword-arg
        pyscal_factory.create_gas_oil(swirr=0.01)  # noqa

    with pytest.raises(TypeError):
        pyscal_factory.create_gas_oil(params="swirr 0.01")

    gasoil = pyscal_factory.create_gas_oil(
        dict(swirr=0.01, swl=0.1, sgcr=0.05, tag="Good sand", ng=1, nog=2)
    )
    assert isinstance(gasoil, GasOil)
    assert gasoil.sgcr == 0.05
    assert gasoil.sgro == 0.0
    assert gasoil.swl == 0.1
    assert gasoil.swirr == 0.01
    assert gasoil.tag == "Good sand"
    sgof = gasoil.SGOF()
    sat_table_str_ok(sgof)
    check_table(gasoil.table)
    assert "Corey krg" in sgof
    assert "Corey krog" in sgof
    assert "Zero capillary pressure" in sgof

    gasoil = pyscal_factory.create_gas_oil(
        dict(ng=1.2, nog=2, krgend=0.8, krgmax=0.9, kroend=0.6)
    )
    sgof = gasoil.SGOF()
    sat_table_str_ok(sgof)
    assert "kroend=0.6" in sgof
    assert "krgend=0.8" in sgof
    check_table(gasoil.table)

    gasoil = pyscal_factory.create_gas_oil(dict(ng=1.3, Log=2, Eog=2, Tog=2))
    sgof = gasoil.SGOF()
    check_table(gasoil.table)
    sat_table_str_ok(sgof)
    assert "Corey krg" in sgof
    assert "LET krog" in sgof

    gasoil = pyscal_factory.create_gas_oil(dict(Lg=1, Eg=1, Tg=1, Log=2, Eog=2, Tog=2))
    sgof = gasoil.SGOF()
    sat_table_str_ok(sgof)
    check_table(gasoil.table)
    assert "LET krg" in sgof
    assert "LET krog" in sgof


def test_factory_wog_gascondensate():
    """Test modelling of gas condensate, which in pyscal terms
    is the same as wateroilgas, except that we allow for aliasing
    in sgrw=sorw for the underlying WaterOil object, and also there
    are additional parameters sgro and kromax for GasOil."""
    wcg = PyscalFactory.create_water_oil_gas(
        dict(
            nw=2,
            now=3,
            ng=1,
            nog=2,
            sgrw=0.1,
            swl=0.1,
            sgcr=0.1,
            sgro=0.1,
            kroend=0.5,
            kromax=0.9,
        )
    )
    assert wcg.gasoil.sgro == 0.1
    assert wcg.wateroil.sorw == 0.1

    swof = wcg.SWOF()
    sgof = wcg.SGOF()

    # sgrw has been aliased to sorw, but the WaterOil object does not know that:
    assert "sgrw" not in swof
    assert "sorw=0.1" in swof
    assert "sgro=0.1" in sgof
    assert "kroend=0.5" in sgof
    assert "kromax=0.9" in sgof

    sat_table_str_ok(swof)
    sat_table_str_ok(sgof)

    # Different sorw and sgrw is a hard error:
    with pytest.raises(ValueError, match="must equal"):
        PyscalFactory.create_water_oil_gas(
            dict(nw=2, now=3, ng=1, nog=2, sorw=0.2, sgrw=0.1, swl=0.1)
        )

    # But it will pass if they both are supplied but are equal:
    wcg_2 = PyscalFactory.create_water_oil_gas(
        dict(nw=2, now=3, ng=1, nog=2, sorw=0.2, sgrw=0.2, swl=0.1)
    )
    assert "sorw=0.2" in wcg_2.SWOF()

    # kroend higher than kromax is an error:
    with pytest.raises(AssertionError):
        PyscalFactory.create_water_oil_gas(
            dict(
                nw=2,
                now=3,
                ng=1,
                nog=2,
                sgcr=0.1,
                sgro=0.1,
                kromax=0.5,
                kroend=0.8,
                swl=0.1,
            )
        )


def test_factory_go_gascondensate():
    """In gas condensate problems, the sgro and kromax parameters are relevant"""
    pyscal_factory = PyscalFactory()
    gasoil = pyscal_factory.create_gas_oil(
        dict(sgro=0.1, sgcr=0.1, tag="Good sand", ng=1, nog=2, kroend=0.5, kromax=0.9)
    )
    assert isinstance(gasoil, GasOil)
    assert gasoil.sgro == 0.1
    assert gasoil.tag == "Good sand"
    sgof = gasoil.SGOF()
    sat_table_str_ok(sgof)
    check_table(gasoil.table)
    assert "Corey krog" in sgof
    assert "kroend=0.5" in sgof
    assert "kromax=0.9" in sgof
    assert "sgro=0.1" in sgof


def test_factory_gaswater():
    """Test that we can create gas-water curves from dictionaries of parameters"""
    pyscal_factory = PyscalFactory()

    # Factory refuses to create incomplete defaulted objects.
    with pytest.raises(ValueError):
        pyscal_factory.create_gas_water()

    with pytest.raises(TypeError):
        # pylint: disable=unexpected-keyword-arg
        pyscal_factory.create_gas_water(swirr=0.01)  # noqa

    with pytest.raises(TypeError):
        # (it must be a dictionary)
        # pylint: disable=unexpected-keyword-arg
        pyscal_factory.create_gas_water(params="swirr 0.01")

    gaswater = pyscal_factory.create_gas_water(
        dict(swirr=0.01, swl=0.03, sgrw=0.1, sgcr=0.15, tag="gassy sand", ng=2, nw=2)
    )

    assert isinstance(gaswater, GasWater)

    assert gaswater.swirr == 0.01
    assert gaswater.swl == 0.03
    assert gaswater.sgrw == 0.1
    assert gaswater.sgcr == 0.15
    assert gaswater.tag == "gassy sand"

    sgfn = gaswater.SGFN()
    swfn = gaswater.SWFN()
    sat_table_str_ok(sgfn)
    sat_table_str_ok(swfn)
    check_table(gaswater.wateroil.table)
    check_table(gaswater.gasoil.table)

    assert "sgrw=0.1" in swfn
    assert "swirr=0.01" in sgfn
    assert "swirr=0.01" in swfn
    assert "sgrw=0.1" in swfn
    assert "sgcr=0.15" in sgfn
    assert "nw=2" in swfn
    assert "ng=2" in sgfn
    assert "gassy sand" in sgfn

    gaswater = pyscal_factory.create_gas_water(dict(lg=1, eg=1, tg=1, nw=3))

    sgfn = gaswater.SGFN()
    swfn = gaswater.SWFN()
    sat_table_str_ok(sgfn)
    sat_table_str_ok(swfn)
    check_table(gaswater.wateroil.table)
    check_table(gaswater.gasoil.table)


def test_factory_wateroilgas():
    """Test creating discrete cases of WaterOilGas from factory"""
    pyscal_factory = PyscalFactory()

    # Factory refuses to create incomplete defaulted objects.
    with pytest.raises(ValueError):
        pyscal_factory.create_water_oil_gas()

    with pytest.raises(TypeError):
        # (this must be a dictionary)
        # pylint: disable=unexpected-keyword-arg
        pyscal_factory.create_water_oil_gas(swirr=0.01)  # noqa

    with pytest.raises(TypeError):
        pyscal_factory.create_water_oil_gas(params="swirr 0.01")

    wog = pyscal_factory.create_water_oil_gas(dict(nw=2, now=3, ng=1, nog=2.5))
    swof = wog.SWOF()
    sgof = wog.SGOF()
    sat_table_str_ok(swof)  # sgof code works for swof also currently
    sat_table_str_ok(sgof)
    assert "Corey krg" in sgof
    assert "Corey krog" in sgof
    assert "Corey krw" in swof
    assert "Corey krow" in swof
    check_table(wog.gasoil.table)
    check_table(wog.wateroil.table)

    # Some users will mess up lower vs upper case:
    wog = pyscal_factory.create_water_oil_gas(dict(NW=2, NOW=3, NG=1, nog=2.5))
    swof = wog.SWOF()
    sgof = wog.SGOF()
    sat_table_str_ok(swof)  # sgof code works for swof also currently
    sat_table_str_ok(sgof)
    assert "Corey krg" in sgof
    assert "Corey krog" in sgof
    assert "Corey krw" in swof
    assert "Corey krow" in swof

    # Mangling data
    wateroil = pyscal_factory.create_water_oil_gas(dict(nw=2, now=3, ng=1))
    assert wateroil.gasoil is None


def test_factory_wateroilgas_deprecated_krowgend():
    """Using long-time deprecated krowend and krogend will fail"""
    with pytest.raises(ValueError):
        PyscalFactory.create_water_oil_gas(
            dict(nw=2, now=3, ng=1, nog=2.5, krowend=0.6, krogend=0.7)
        )


def test_factory_wateroilgas_wo():
    """Test making only wateroil through the wateroilgas factory"""
    pyscal_factory = PyscalFactory()
    wog = pyscal_factory.create_water_oil_gas(
        dict(nw=2, now=3, kroend=0.5, sorw=0.04, swcr=0.1)
    )
    swof = wog.SWOF()
    assert "Corey krw" in swof
    assert "KRW" in wog.wateroil.table
    sat_table_str_ok(swof)
    check_table(wog.wateroil.table)
    assert wog.gasoil is None

    wog.SGOF()


def test_factory_wateroil_paleooil(caplog):
    """Test making a WaterOil object with socr different from sorw."""
    pyscal_factory = PyscalFactory()
    sorw = 0.09
    wateroil = pyscal_factory.create_water_oil(
        dict(nw=2, now=3, kroend=0.5, sorw=sorw, socr=sorw + 0.01, swcr=0.1)
    )
    swof = wateroil.SWOF()
    assert "Corey krw" in swof
    assert "socr=0.1" in swof
    sat_table_str_ok(swof)
    check_table(wateroil.table)

    # If socr is close to sorw, socr is reset to sorw.
    for socr in [sorw - 1e-9, sorw, sorw + 1e-9]:
        wo_socrignored = pyscal_factory.create_water_oil(
            dict(nw=2, now=3, kroend=0.5, sorw=0.09, socr=socr, swcr=0.1)
        )
        swof = wo_socrignored.SWOF()
        assert "socr" not in swof  # socr is effectively ignored when = sorw.
        sat_table_str_ok(swof)
        if socr != sorw:
            # This warning should only occur when it seems like the user
            # has tried to explicitly set socr
            assert "socr was close to sorw, reset to sorw" in caplog.text

    with pytest.raises(ValueError, match="socr must be equal to or larger than sorw"):
        pyscal_factory.create_water_oil(
            dict(nw=2, now=3, kroend=0.5, sorw=0.09, socr=0.001, swcr=0.1, h=0.1)
        )


def test_load_relperm_df(tmp_path, caplog):
    """Test loading of dataframes with validation from excel or from csv"""
    testdir = Path(__file__).absolute().parent

    scalfile_xls = testdir / "data/scal-pc-input-example.xlsx"

    scaldata = PyscalFactory.load_relperm_df(scalfile_xls)
    with pytest.raises(IOError):
        PyscalFactory.load_relperm_df("not-existing-file")

    with pytest.raises(ValueError, match="Non-existing sheet-name"):
        PyscalFactory.load_relperm_df(scalfile_xls, sheet_name="foo")

    assert "SATNUM" in scaldata
    assert "CASE" in scaldata
    assert not scaldata.empty

    os.chdir(tmp_path)
    scaldata.to_csv("scal-input.csv")
    scaldata_fromcsv = PyscalFactory.load_relperm_df("scal-input.csv")
    assert "CASE" in scaldata_fromcsv
    assert not scaldata_fromcsv.empty
    scaldata_fromdf = PyscalFactory.load_relperm_df(scaldata_fromcsv)
    assert "CASE" in scaldata_fromdf
    assert "SATNUM" in scaldata_fromdf
    assert len(scaldata_fromdf) == len(scaldata_fromcsv) == len(scaldata)

    scaldata_fromcsv = PyscalFactory.load_relperm_df("scal-input.csv", sheet_name="foo")
    assert "Sheet name only relevant for XLSX files, ignoring foo" in caplog.text

    with pytest.raises(ValueError, match="Unsupported argument"):
        PyscalFactory.load_relperm_df(dict(foo=1))

    # Perturb the dataframe, this should trigger errors
    with pytest.raises(ValueError):
        PyscalFactory.load_relperm_df(scaldata.drop("SATNUM", axis="columns"))
    wrongsatnums = scaldata.copy()
    wrongsatnums["SATNUM"] = wrongsatnums["SATNUM"] * 2
    with pytest.raises(ValueError):
        PyscalFactory.load_relperm_df(wrongsatnums)
    wrongsatnums = scaldata.copy()
    wrongsatnums["SATNUM"] = wrongsatnums["SATNUM"].astype(int)
    wrongsatnums = wrongsatnums[wrongsatnums["SATNUM"] > 2]
    with pytest.raises(ValueError):
        PyscalFactory.load_relperm_df(wrongsatnums)
    wrongcases = scaldata.copy()
    wrongcases["CASE"] = wrongcases["CASE"] + "ffooo"
    with pytest.raises(ValueError):
        PyscalFactory.load_relperm_df(wrongcases)

    with pytest.raises(ValueError):
        PyscalFactory.load_relperm_df(scaldata.drop(["Lw", "Lg"], axis="columns"))

    # Insert a NaN, this replicates what happens if cells are merged
    mergedcase = scaldata.copy()
    mergedcase.loc[3, "SATNUM"] = np.nan
    with pytest.raises(ValueError):
        PyscalFactory.load_relperm_df(mergedcase)

    relpermfile_xls = testdir / "data/relperm-input-example.xlsx"
    relpermdata = PyscalFactory.load_relperm_df(relpermfile_xls)
    assert "TAG" in relpermdata
    assert "SATNUM" in relpermdata
    assert "satnum" not in relpermdata  # always converted to upper-case
    assert len(relpermdata) == 3
    swof_str = PyscalFactory.create_pyscal_list(relpermdata, h=0.2).SWOF()
    assert "Åre 1.8" in swof_str
    assert "SATNUM 2" in swof_str  # Autogenerated in SWOF, generated by factory
    assert "SATNUM 3" in swof_str
    assert "foobar" in swof_str  # Random string injected in xlsx.

    # Make a dummy text file
    Path("dummy.txt").write_text("foo\nbar, com", encoding="utf8")
    with pytest.raises(ValueError):
        PyscalFactory.load_relperm_df("dummy.txt")

    # Make an empty csv file
    Path("empty.csv").write_text("", encoding="utf8")
    with pytest.raises(ValueError, match="Impossible to infer file format"):
        PyscalFactory.load_relperm_df("empty.csv")

    with pytest.raises(ValueError, match="SATNUM must be present"):
        PyscalFactory.load_relperm_df(pd.DataFrame())

    # Merge tags and comments if both are supplied
    Path("tagandcomment.csv").write_text(
        "SATNUM,nw,now,tag,comment\n1,1,1,a-tag,a-comment", encoding="utf8"
    )
    tagandcomment_df = PyscalFactory.load_relperm_df("tagandcomment.csv")
    assert (
        tagandcomment_df["TAG"].values[0] == "SATNUM 1 tag: a-tag; comment: a-comment"
    )

    # Missing SATNUMs:
    Path("wrongsatnum.csv").write_text("SATNUM,nw,now\n1,1,1\n3,1,1", encoding="utf8")
    with pytest.raises(ValueError, match="Missing SATNUMs?"):
        PyscalFactory.load_relperm_df("wrongsatnum.csv")

    # Missing SATNUMs, like merged cells:
    Path("mergedcells.csv").write_text(
        "CASE,SATNUM,nw,now\nlow,,1,1\nlow,1,2,2\nlow,,3,32", encoding="utf8"
    )
    with pytest.raises(ValueError, match="Found not-a-number"):
        PyscalFactory.load_relperm_df("mergedcells.csv")

    # Missing SATNUMs, like merged cells:
    Path("mergedcellscase.csv").write_text(
        "CASE,SATNUM,nw,now\n,1,1,1\nlow,1,2,2\n,1,3,32", encoding="utf8"
    )
    with pytest.raises(ValueError, match="Found not-a-number"):
        PyscalFactory.load_relperm_df("mergedcellscase.csv")


def test_many_nans():
    """Excel or oocalc sometimes saves a xlsx file that gives all NaN rows and
    all-NaN columns, maybe some column setting that triggers Pandas to load
    them as actual columns/rows.

    Ensure we handle extra Nans in both directions"""
    nanframe = pd.DataFrame(
        [
            {"SATNUM": 1, "nw": 2, "now": 2, "Unnamed: 15": np.nan},
            {"SATNUM": np.nan, "nw": np.nan, "now": np.nan, "Unnamed: 15": np.nan},
        ]
    )
    wateroil_list = PyscalFactory.create_pyscal_list(
        PyscalFactory.load_relperm_df(nanframe)
    )
    assert len(wateroil_list) == 1
    sat_table_str_ok(wateroil_list.SWOF())


def test_xls_factory():
    """Test/demonstrate how to go from data in an excel row to pyscal objects

    This test function predates the load_relperm_df() function, but can
    still be in here.
    """
    testdir = Path(__file__).absolute().parent

    xlsxfile = testdir / "data/scal-pc-input-example.xlsx"

    scalinput = pd.read_excel(xlsxfile, engine="openpyxl").set_index(["SATNUM", "CASE"])

    for ((satnum, _), params) in scalinput.iterrows():
        assert satnum
        wog = PyscalFactory.create_water_oil_gas(params.to_dict())
        swof = wog.SWOF()
        assert "LET krw" in swof
        assert "LET krow" in swof
        assert "Simplified J" in swof
        sgof = wog.SGOF()
        sat_table_str_ok(sgof)
        assert "LET krg" in sgof
        assert "LET krog" in sgof


def test_create_scal_recommendation_list():
    """Test the factory methods for making scalrecommendation lists"""
    testdir = Path(__file__).absolute().parent
    scalfile_xls = testdir / "data/scal-pc-input-example.xlsx"
    scaldata = PyscalFactory.load_relperm_df(scalfile_xls)

    scalrec_list = PyscalFactory.create_scal_recommendation_list(scaldata)
    assert len(scalrec_list) == 3
    assert scalrec_list.pyscaltype == SCALrecommendation

    # Erroneous input:
    with pytest.raises(ValueError, match="Too many cases supplied for SATNUM 2"):
        PyscalFactory.create_scal_recommendation_list(
            pd.DataFrame(
                columns=["SATNUM", "CASE", "NW", "NOW"],
                data=[
                    [1, "low", 1, 1],
                    [1, "base", 2, 2],
                    [1, "high", 3, 3],
                    [2, "low", 1, 1],
                    [2, "nearlylow", 1.4, 1.2],
                    [2, "base", 2, 2],
                    [2, "high", 3, 3],
                ],
            )
        )
    with pytest.raises(ValueError, match="Too few cases supplied for SATNUM 2"):
        PyscalFactory.create_scal_recommendation_list(
            pd.DataFrame(
                columns=["SATNUM", "CASE", "NW", "NOW"],
                data=[
                    [1, "low", 1, 1],
                    [1, "base", 2, 2],
                    [1, "high", 3, 3],
                    [2, "low", 1, 1],
                    [2, "high", 3, 3],
                ],
            )
        )


def test_create_pyscal_list():
    """Test the factory methods for making pyscal lists"""
    testdir = Path(__file__).absolute().parent
    scalfile_xls = testdir / "data/scal-pc-input-example.xlsx"
    scaldata = PyscalFactory.load_relperm_df(scalfile_xls)
    basecasedata = scaldata[scaldata["CASE"] == "base"].reset_index()
    relpermlist = PyscalFactory.create_pyscal_list(basecasedata)
    assert len(relpermlist) == 3
    assert relpermlist.pyscaltype == WaterOilGas

    wo_list = PyscalFactory.create_pyscal_list(
        basecasedata.drop(["Lg", "Eg", "Tg", "Log", "Eog", "Tog"], axis="columns")
    )

    assert len(wo_list) == 3
    assert wo_list.pyscaltype == WaterOil

    go_list = PyscalFactory.create_pyscal_list(
        basecasedata.drop(["Lw", "Ew", "Tw", "Low", "Eow", "Tow"], axis="columns")
    )

    assert len(go_list) == 3
    assert go_list.pyscaltype == GasOil

    gw_list = PyscalFactory.create_pyscal_list(
        basecasedata.drop(["Low", "Eow", "Tow", "Log", "Eog", "Tog"], axis="columns")
    )

    assert len(gw_list) == 3
    assert gw_list.pyscaltype == GasWater

    with pytest.raises(
        ValueError, match="Could not determine two or three phase from parameters"
    ):
        PyscalFactory.create_pyscal_list(
            basecasedata.drop(["Ew", "Eg"], axis="columns")
        )


def test_scalrecommendation():
    """Testing making SCAL rec from dict of dict."""
    pyscal_factory = PyscalFactory()

    scal_input = {
        "low": {"nw": 2, "now": 4, "ng": 1, "nog": 2},
        "BASE": {"nw": 3, "NOW": 3, "ng": 1, "nog": 2},
        "high": {"nw": 4, "now": 2, "ng": 1, "nog": 3},
    }
    scal = pyscal_factory.create_scal_recommendation(scal_input)

    with pytest.raises(ValueError, match="Input must be a dict"):
        pyscal_factory.create_scal_recommendation("low")

    # (not supported yet to make WaterOil only..)
    interp = scal.interpolate(-0.5)
    sat_table_str_ok(interp.SWOF())
    sat_table_str_ok(interp.SGOF())
    sat_table_str_ok(interp.SLGOF())
    sat_table_str_ok(interp.SOF3())
    check_table(interp.wateroil.table)
    check_table(interp.gasoil.table)

    # Check that we error if any of the parameters above is missing:
    for case in ["low", "BASE", "high"]:
        copy1 = scal_input.copy()
        del copy1[case]
        with pytest.raises(ValueError):
            pyscal_factory.create_scal_recommendation(copy1)

    go_only = scal_input.copy()
    del go_only["low"]["now"]
    del go_only["low"]["nw"]
    gasoil = pyscal_factory.create_scal_recommendation(go_only)
    assert gasoil.low.wateroil is None
    assert gasoil.base.wateroil is not None
    assert gasoil.high.wateroil is not None
    # SCALrecommendation of gasoil only works as long as you
    # don't try to ask for water data:
    assert "SGFN" in gasoil.interpolate(-0.4).SGFN()
    assert "SWOF" not in gasoil.interpolate(-0.2).SWOF()

    basehigh = scal_input.copy()
    del basehigh["low"]
    with pytest.raises(ValueError, match='"low" case not supplied'):
        pyscal_factory.create_scal_recommendation(basehigh)

    baselow = scal_input.copy()
    del baselow["high"]
    with pytest.raises(ValueError, match='"high" case not supplied'):
        pyscal_factory.create_scal_recommendation(baselow)

    with pytest.raises(
        ValueError, match="All values in parameter dict must be dictionaries"
    ):

        pyscal_factory.create_scal_recommendation(
            {"low": [1, 2], "base": {"swl": 0.1}, "high": {"swl": 0.1}}
        )


def test_scalrecommendation_gaswater():
    """Testing making SCAL rec from dict of dict for gaswater input"""
    pyscal_factory = PyscalFactory()

    scal_input = {
        "low": {"nw": 2, "ng": 1},
        "BASE": {"nw": 3, "ng": 1},
        "high": {"nw": 4, "ng": 1},
    }
    scal = pyscal_factory.create_scal_recommendation(scal_input, h=0.2)
    interp = scal.interpolate(-0.5, h=0.2)
    sat_table_str_ok(interp.SWFN())
    sat_table_str_ok(interp.SGFN())
    check_table(interp.wateroil.table)
    check_table(interp.gasoil.table)


def test_xls_scalrecommendation():
    """Test making SCAL recommendations from xls data"""
    testdir = Path(__file__).absolute().parent

    xlsxfile = testdir / "data/scal-pc-input-example.xlsx"
    scalinput = pd.read_excel(xlsxfile, engine="openpyxl").set_index(["SATNUM", "CASE"])
    for satnum in scalinput.index.levels[0].values:
        dictofdict = scalinput.loc[satnum, :].to_dict(orient="index")
        scalrec = PyscalFactory.create_scal_recommendation(dictofdict)
        scalrec.interpolate(+0.5)


def test_no_gasoil():
    """The command client does not support two-phase gas-oil, because
    that is most likely an sign of a user input error.
    (misspelled other columns f.ex).

    Make sure we fail in that case."""
    dframe = pd.DataFrame(columns=["SATNUM", "NOW", "NG"], data=[[1, 2, 2]])
    with pytest.raises(ValueError):
        PyscalFactory.load_relperm_df(dframe)


def test_check_deprecated_krowgend():
    """Up until pyscal 0.5.x, krogend and krowend were parameters
    to the oil curve parametrization for WaterOil and GasOil. From
    pyscal 0.6.0, krogend and krowend are merged to kroend.
    After pyscal 0.8 presence of krogend and krowend is a ValueError
    """
    with pytest.raises(ValueError):
        PyscalFactory.create_water_oil(dict(swl=0.1, nw=2, now=2, krowend=0.4))

    with pytest.raises(ValueError):
        PyscalFactory.create_gas_oil(dict(swl=0.1, ng=2, nog=2, krogend=0.4))

    # If krogend and kroend are both present, krogend is to be silently ignored
    # (random columns are in general accepted and ignored by pyscal)

    gasoil = PyscalFactory.create_gas_oil(
        dict(swl=0.1, ng=2, nog=2, krogend=0.4, kroend=0.3)
    )
    assert gasoil.table["KROG"].max() == 0.3

    wateroil = PyscalFactory.create_water_oil(
        dict(swl=0.1, nw=2, now=2, krowend=0.4, kroend=0.3)
    )
    assert wateroil.table["KROW"].max() == 0.3


def parse_gensatfuncline(conf_line):
    """Utility function that emulates how gensatfunc could parse
    its configuration lines in a pyscalfactory compatible fashion

    Args:
        conf_line (str): gensatfunc config line
    Returns:
        dict
    """

    # This is how the config line should be interpreted in terms of
    # pyscal parameters. Note that we are case insensitive in the
    # factory class
    line_syntax = [
        "CMD",
        "Lw",
        "Ew",
        "Tw",
        "Lo",
        "Eo",
        "To",
        "Sorw",
        "Swl",
        "krwend",
        "steps",
        "perm",
        "poro",
        "a",
        "b",
        "sigma_costau",
    ]

    if len(conf_line.split()) > len(line_syntax):
        raise ValueError("Too many items on gensatfunc confline")

    params = {}
    for (idx, value) in enumerate(conf_line.split()):
        if idx > 0:  # Avoid the CMD
            params[line_syntax[idx]] = float(value)

    # The 'steps' is not supported in pyscal, convert it:
    if "steps" in params:
        params["h"] = 1.0 / params["steps"]

    if "krwend" not in params:  # Last mandatory item
        raise ValueError("Too few items on gensatfunc confline")

    return params


def test_gensatfunc():
    """Test how the external tool gen_satfunc could use
    the factory functionality"""

    pyscal_factory = PyscalFactory()

    # Example config line for gen_satfunc:
    conf_line_pc = "RELPERM 4 2 1 3 2 1 0.15 0.10 0.5 20 100 0.2 0.22 -0.5 30"

    wateroil = pyscal_factory.create_water_oil(parse_gensatfuncline(conf_line_pc))
    swof = wateroil.SWOF()
    assert "0.17580" in swof  # krw at sw=0.65
    assert "0.0127" in swof  # krow at sw=0.65
    assert "Capillary pressure from normalized J-function" in swof
    assert "2.0669" in swof  # pc at swl

    conf_line_min = "RELPERM 1 2 3 1 2 3 0.1 0.15 0.5 20"
    wateroil = pyscal_factory.create_water_oil(parse_gensatfuncline(conf_line_min))
    swof = wateroil.SWOF()
    assert "Zero capillary pressure" in swof

    conf_line_few = "RELPERM 1 2 3 1 2 3"
    with pytest.raises(ValueError):
        parse_gensatfuncline(conf_line_few)

    # sigma_costau is missing here:
    conf_line_almost_pc = "RELPERM 4 2 1 3 2 1 0.15 0.10 0.5 20 100 0.2 0.22 -0.5"
    wateroil = pyscal_factory.create_water_oil(
        parse_gensatfuncline(conf_line_almost_pc)
    )
    swof = wateroil.SWOF()
    # The factory will not recognize the normalized J-function
    # when costau is missing. Any error message would be the responsibility
    # of the parser
    assert "Zero capillary pressure" in swof


def test_sufficient_params():
    """Test the utility functions to determine whether
    WaterOil and GasOil object have sufficient parameters"""

    assert factory.sufficient_gas_oil_params({"ng": 0, "nog": 0})
    # If it looks like the user meant to create GasOil, but only provided
    # data for krg, then might error hard. If the user did not provide
    # any data for GasOil, then the code returns False
    with pytest.raises(ValueError):
        factory.sufficient_gas_oil_params({"ng": 0}, failhard=True)
    assert not factory.sufficient_gas_oil_params({"ng": 0}, failhard=False)
    assert not factory.sufficient_gas_oil_params({})
    with pytest.raises(ValueError):
        factory.sufficient_gas_oil_params({"lg": 0}, failhard=True)
    assert not factory.sufficient_gas_oil_params({"lg": 0}, failhard=False)
    assert factory.sufficient_gas_oil_params(
        {"lg": 0, "eg": 0, "Tg": 0, "log": 0, "eog": 0, "tog": 0}
    )

    assert factory.sufficient_water_oil_params({"nw": 0, "now": 0})
    with pytest.raises(ValueError):
        factory.sufficient_water_oil_params({"nw": 0}, failhard=True)
    assert not factory.sufficient_water_oil_params({})
    with pytest.raises(ValueError):
        factory.sufficient_water_oil_params({"lw": 0}, failhard=True)
    assert factory.sufficient_water_oil_params(
        {"lw": 0, "ew": 0, "Tw": 0, "low": 0, "eow": 0, "tow": 0}
    )


def test_sufficient_params_gaswater():
    """Test that we can detect sufficient parameters
    for gas-water only"""
    assert factory.sufficient_gas_water_params({"nw": 0, "ng": 0})
    assert not factory.sufficient_gas_water_params({"nw": 0, "nog": 0})
    assert factory.sufficient_gas_water_params(dict(lw=0, ew=0, tw=0, lg=0, eg=0, tg=0))
    assert not factory.sufficient_gas_water_params(dict(lw=0))
    assert not factory.sufficient_gas_water_params(dict(lw=0, lg=0))
    assert not factory.sufficient_gas_water_params(dict(lw=0, lg=0))

    with pytest.raises(ValueError):
        factory.sufficient_gas_water_params(dict(lw=0), failhard=True)
    with pytest.raises(ValueError):
        factory.sufficient_gas_water_params({"nw": 3}, failhard=True)

    assert factory.sufficient_gas_water_params(dict(lw=0, ew=0, tw=0, ng=0))
    assert factory.sufficient_gas_water_params(dict(lg=0, eg=0, tg=0, nw=0))
    assert not factory.sufficient_gas_water_params(dict(lg=0, eg=0, tg=0, ng=0))


def test_case_aliasing():
    """Test that we can use aliases for the CASE column
    in SCAL recommendations"""
    dframe = pd.DataFrame(
        columns=["SATNUM", "CASE", "Nw", "Now", "Ng", "Nog"],
        data=[
            [1, "pess", 2, 2, 1, 1],
            [1, "base", 3, 1, 1, 1],
            [1, "opt", 3, 1, 1, 1],
        ],
    )
    relperm_data = PyscalFactory.load_relperm_df(dframe)
    PyscalFactory.create_scal_recommendation_list(relperm_data, h=0.2).interpolate(-0.4)
    dframe = pd.DataFrame(
        columns=["SATNUM", "CASE", "Nw", "Now", "Ng", "Nog"],
        data=[
            [1, "pessimistic", 2, 2, 1, 1],
            [1, "base", 3, 1, 1, 1],
            [1, "optiMISTIc", 3, 1, 1, 1],
        ],
    )
    relperm_data = PyscalFactory.load_relperm_df(dframe)
    PyscalFactory.create_scal_recommendation_list(relperm_data, h=0.2).interpolate(-0.4)

    with pytest.raises(ValueError):
        PyscalFactory.load_relperm_df(
            pd.DataFrame(
                columns=["SATNUM", "CASE", "Nw", "Now", "Ng", "Nog"],
                data=[
                    [1, "FOOBAR", 2, 2, 1, 1],
                    [1, "base", 3, 1, 1, 1],
                    [1, "optIMIstiC", 3, 1, 1, 1],
                ],
            )
        )

    # Ambigous data:
    with pytest.raises(ValueError):
        amb = PyscalFactory.load_relperm_df(
            pd.DataFrame(
                columns=["SATNUM", "CASE", "Nw", "Now", "Ng", "Nog"],
                data=[
                    [1, "low", 2, 2, 1, 1],
                    [1, "pess", 5, 5, 5, 5],
                    [1, "base", 3, 1, 1, 1],
                    [1, "optIMIstiC", 3, 1, 1, 1],
                ],
            )
        )
        PyscalFactory.create_scal_recommendation_list(amb)

    # Missing a case
    with pytest.raises(ValueError):
        PyscalFactory.load_relperm_df(
            pd.DataFrame(
                columns=["SATNUM", "CASE", "Nw", "Now", "Ng", "Nog"],
                data=[[1, "base", 3, 1, 1, 1], [1, "optIMIstiC", 3, 1, 1, 1]],
            )
        )
    # Missing a case
    with pytest.raises(ValueError):
        PyscalFactory.load_relperm_df(
            pd.DataFrame(
                columns=["SATNUM", "CASE", "Nw", "Now", "Ng", "Nog"],
                data=[[1, "base", 3, 1, 1, 1]],
            )
        )


def test_socr_via_dframe():
    """Test that the "socr" parameter is picked up from a dataframe/xlsx input"""
    p_list = PyscalFactory.create_pyscal_list(
        PyscalFactory.load_relperm_df(
            pd.DataFrame(
                columns=["SATNUM", "Nw", "Now", "socr"],
                data=[[1, 2, 2, 0.5]],
            )
        )
    )
    assert "socr=0.5" in p_list.SWOF()


def test_swirr_partially_missing(tmp_path):
    """Test that swirr can be present for only a subset of the rows,
    and interpreted as zero when not there."""
    dframe = pd.DataFrame(
        columns=[
            "SATNUM",
            "Nw",
            "Now",
            "swl",
            "swirr",
            "a",
            "b",
            "poro_ref",
            "perm_ref",
            "drho",
        ],
        data=[
            [1, 2, 2, 0.2, 0.1, 2, -2, 0.2, 100, 300],
            [2, 3, 3, 0.1, np.nan, np.nan, np.nan, np.nan, np.nan],
        ],
    )
    relperm_data = PyscalFactory.load_relperm_df(dframe)
    p_list = PyscalFactory.create_pyscal_list(relperm_data, h=0.2)
    assert "a=2, b=-2" in p_list[1].pccomment
    assert p_list[2].pccomment == ""

    os.chdir(tmp_path)
    dframe.to_excel("partial_pc.xlsx")
    relperm_data_via_xlsx = PyscalFactory.load_relperm_df("partial_pc.xlsx")
    p_list = PyscalFactory.create_pyscal_list(relperm_data_via_xlsx, h=0.2)
    assert "a=2, b=-2" in p_list[1].pccomment
    assert p_list[2].pccomment == ""


def test_corey_let_mix():
    """Test that we can supply a dataframe where some SATNUMs
    have Corey and others have LET"""
    dframe = pd.DataFrame(
        columns=["SATNUM", "Nw", "Now", "Lw", "Ew", "Tw", "Ng", "Nog"],
        data=[[1, 2, 2, np.nan, np.nan, np.nan, 1, 1], [2, np.nan, 3, 1, 1, 1, 2, 2]],
    )
    relperm_data = PyscalFactory.load_relperm_df(dframe)
    p_list = PyscalFactory.create_pyscal_list(relperm_data, h=0.2)
    swof1 = p_list.pyscal_list[0].SWOF()
    swof2 = p_list.pyscal_list[1].SWOF()
    assert "Corey krw" in swof1
    assert "Corey krow" in swof1
    assert "LET krw" in swof2
    assert "Corey krow" in swof2


def test_infer_tabular_file_format(tmp_path, caplog):
    """Test code that infers the fileformat of files with tabular data"""
    testdir = Path(__file__).absolute().parent
    assert (
        factory.infer_tabular_file_format(testdir / "data/scal-pc-input-example.xlsx")
        == "xlsx"
    )
    assert (
        factory.infer_tabular_file_format(
            str(testdir / "data/scal-pc-input-example.xlsx")
        )
        == "xlsx"
    )
    assert (
        factory.infer_tabular_file_format(testdir / "data/scal-pc-input-example.xls")
        == "xls"
    )
    os.chdir(tmp_path)
    pd.DataFrame([{"SATNUM": 1, "NW": 2}]).to_csv("some.csv", index=False)
    assert factory.infer_tabular_file_format("some.csv") == "csv"

    Path("empty.csv").write_text("", encoding="utf8")
    assert factory.infer_tabular_file_format("empty.csv") == ""
    # Ensure Pandas's error message got through:
    assert "No columns to parse from file" in caplog.text

    # We don't want ISO-8859 files, ensure we fail
    norw_chars = "Dette,er,en,CSV,fil\nmed,iso-8859:,æ,ø,å"
    Path("iso8859.csv").write_bytes(norw_chars.encode("iso-8859-1"))
    assert factory.infer_tabular_file_format("iso8859.csv") == ""
    # Providing an error that this error was due to ISO-8859 and
    # nothing else is deemed too hard.
    Path("utf8.csv").write_bytes(norw_chars.encode("utf-8"))
    assert factory.infer_tabular_file_format("utf8.csv") == "csv"

    # Write some random bytes to a file, this should with very
    # little probability give a valid xlsx/xls/csv file.
    Path("wrong.csv").write_bytes(os.urandom(100))
    assert factory.infer_tabular_file_format("wrong.csv") == ""


@pytest.mark.parametrize(
    "orig_dict, keylist, expected_dict",
    [
        ({}, [], {}),
        ({"foo": 1}, [], {}),
        ({"foo": 1}, ["fo"], {}),
        ({"foo": 1}, ["foo"], {"foo": 1}),
        ({}, ["foo"], {}),
    ],
)
def test_slicedict(orig_dict, keylist, expected_dict):
    """Test that dictionaries can be sliced for subsets"""
    assert factory.slicedict(orig_dict, keylist) == expected_dict
