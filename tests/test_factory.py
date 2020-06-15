"""Test the PyscalFactory module"""
# -*- encoding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import six

import numpy as np
import pandas as pd

import pytest

from pyscal import (
    WaterOil,
    GasOil,
    GasWater,
    WaterOilGas,
    PyscalFactory,
    factory,
    SCALrecommendation,
)

from common import sat_table_str_ok, check_table


def test_factory_wateroil():
    """Test that we can create curves from dictionaries of parameters"""
    factory = PyscalFactory()

    # Factory refuses to create incomplete defaulted objects.
    with pytest.raises(ValueError):
        factory.create_water_oil()

    with pytest.raises(TypeError):
        # (it must be a dictionary)
        factory.create_water_oil(swirr=0.01)  # noqa

    wateroil = factory.create_water_oil(
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
    assert "krw" in wateroil.table
    assert "Corey" in wateroil.krwcomment
    assert wateroil.table["krw"].max() == 0.2  # Because sorw==0 by default
    check_table(wateroil.table)
    sat_table_str_ok(wateroil.SWOF())
    sat_table_str_ok(wateroil.SWFN())

    wateroil = factory.create_water_oil(
        dict(nw=3, now=2, sorw=0.1, krwend=0.2, krwmax=0.5)
    )
    assert isinstance(wateroil, WaterOil)
    assert "krw" in wateroil.table
    assert "Corey" in wateroil.krwcomment
    assert wateroil.table["krw"].max() == 0.5
    check_table(wateroil.table)
    sat_table_str_ok(wateroil.SWOF())
    sat_table_str_ok(wateroil.SWFN())

    # Ambiguous works, but we don't guarantee that this results
    # in LET or Corey.
    wateroil = factory.create_water_oil(dict(nw=3, Lw=2, Ew=2, Tw=2, now=3))
    assert "krw" in wateroil.table
    assert "Corey" in wateroil.krwcomment or "LET" in wateroil.krwcomment
    check_table(wateroil.table)
    sat_table_str_ok(wateroil.SWOF())
    sat_table_str_ok(wateroil.SWFN())

    # Mixing Corey and LET
    wateroil = factory.create_water_oil(dict(Lw=2, Ew=2, Tw=2, krwend=1, now=4))
    assert isinstance(wateroil, WaterOil)
    assert "krw" in wateroil.table
    assert wateroil.table["krw"].max() == 1.0
    assert "LET" in wateroil.krwcomment
    check_table(wateroil.table)
    sat_table_str_ok(wateroil.SWOF())
    sat_table_str_ok(wateroil.SWFN())

    wateroil = factory.create_water_oil(
        dict(Lw=2, Ew=2, Tw=2, Low=3, Eow=3, Tow=3, krwend=0.5)
    )
    assert isinstance(wateroil, WaterOil)
    assert "krw" in wateroil.table
    assert "krow" in wateroil.table
    assert wateroil.table["krw"].max() == 0.5
    assert wateroil.table["krow"].max() == 1
    assert "LET" in wateroil.krwcomment
    assert "LET" in wateroil.krowcomment
    check_table(wateroil.table)
    sat_table_str_ok(wateroil.SWOF())
    sat_table_str_ok(wateroil.SWFN())

    # Add capillary pressure
    wateroil = factory.create_water_oil(
        dict(swl=0.1, nw=1, now=1, a=2, b=-1, poro_ref=0.2, perm_ref=100, drho=200)
    )
    assert "pc" in wateroil.table
    assert wateroil.table["pc"].max() > 0.0
    assert "Simplified J" in wateroil.pccomment
    check_table(wateroil.table)
    sat_table_str_ok(wateroil.SWOF())
    sat_table_str_ok(wateroil.SWFN())

    # Test that the optional gravity g is picked up:
    wateroil = factory.create_water_oil(
        dict(swl=0.1, nw=1, now=1, a=2, b=-1, poro_ref=0.2, perm_ref=100, drho=200, g=0)
    )
    assert "pc" in wateroil.table
    assert wateroil.table["pc"].max() == 0.0
    check_table(wateroil.table)
    sat_table_str_ok(wateroil.SWOF())
    sat_table_str_ok(wateroil.SWFN())

    # Test petrophysical simple J:
    wateroil = factory.create_water_oil(
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
    assert "pc" in wateroil.table
    assert wateroil.table["pc"].max() > 0.0
    assert "etrophysic" in wateroil.pccomment
    check_table(wateroil.table)
    sat_table_str_ok(wateroil.SWOF())
    sat_table_str_ok(wateroil.SWFN())

    # One pc param missing:
    wateroil = factory.create_water_oil(
        dict(swl=0.1, nw=1, now=1, a=2, b=-1, perm_ref=100, drho=200, g=0)
    )
    assert "pc" not in wateroil.table


def test_ambiguity():
    """Test how the factory handles ambiguity between Corey and LET
    parameters"""
    factory = PyscalFactory()
    wateroil = factory.create_water_oil(
        dict(swl=0.1, nw=10, Lw=1, Ew=1, Tw=1, now=2, h=0.1, no=2)
    )
    # Corey is picked here.
    assert "Corey" in wateroil.krwcomment
    assert "krw" in wateroil.table


def test_factory_gasoil():
    """Test that we can create curves from dictionaries of parameters"""
    factory = PyscalFactory()

    with pytest.raises(TypeError):
        # (this must be a dictionary)
        factory.create_gas_oil(swirr=0.01)  # noqa

    gasoil = factory.create_gas_oil(
        dict(swirr=0.01, swl=0.1, sgcr=0.05, tag="Good sand", ng=1, nog=2)
    )
    assert isinstance(gasoil, GasOil)
    assert gasoil.sgcr == 0.05
    assert gasoil.swl == 0.1
    assert gasoil.swirr == 0.01
    assert gasoil.tag == "Good sand"
    sgof = gasoil.SGOF()
    sat_table_str_ok(sgof)
    check_table(gasoil.table)
    assert "Corey krg" in sgof
    assert "Corey krog" in sgof
    assert "Zero capillary pressure" in sgof

    gasoil = factory.create_gas_oil(
        dict(ng=1.2, nog=2, krgend=0.8, krgmax=0.9, krogend=0.6)
    )
    sgof = gasoil.SGOF()
    sat_table_str_ok(sgof)
    assert "kroend=0.6" in sgof
    assert "krgend=0.8" in sgof
    check_table(gasoil.table)

    gasoil = factory.create_gas_oil(dict(ng=1.3, Log=2, Eog=2, Tog=2))
    sgof = gasoil.SGOF()
    check_table(gasoil.table)
    sat_table_str_ok(sgof)
    assert "Corey krg" in sgof
    assert "LET krog" in sgof

    gasoil = factory.create_gas_oil(dict(Lg=1, Eg=1, Tg=1, Log=2, Eog=2, Tog=2))
    sgof = gasoil.SGOF()
    sat_table_str_ok(sgof)
    check_table(gasoil.table)
    assert "LET krg" in sgof
    assert "LET krog" in sgof


def test_factory_gaswater():
    """Test that we can create gas-water curves from dictionaries of parameters"""
    factory = PyscalFactory()

    with pytest.raises(TypeError):
        factory.create_gas_water(swirr=0.01)  # noqa

    gaswater = factory.create_gas_water(
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

    gaswater = factory.create_gas_water(dict(lg=1, eg=1, tg=1, nw=3))

    sgfn = gaswater.SGFN()
    swfn = gaswater.SWFN()
    sat_table_str_ok(sgfn)
    sat_table_str_ok(swfn)
    check_table(gaswater.wateroil.table)
    check_table(gaswater.gasoil.table)


def test_factory_wateroilgas():
    """Test creating discrete cases of WaterOilGas from factory"""
    factory = PyscalFactory()

    wog = factory.create_water_oil_gas(dict(nw=2, now=3, ng=1, nog=2.5))
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
    wog = factory.create_water_oil_gas(dict(NW=2, NOW=3, NG=1, nog=2.5))
    swof = wog.SWOF()
    sgof = wog.SGOF()
    sat_table_str_ok(swof)  # sgof code works for swof also currently
    sat_table_str_ok(sgof)
    assert "Corey krg" in sgof
    assert "Corey krog" in sgof
    assert "Corey krw" in swof
    assert "Corey krow" in swof

    # Mangling data
    wateroil = factory.create_water_oil_gas(dict(nw=2, now=3, ng=1))
    assert wateroil.gasoil is None


def test_factory_wateroilgas_deprecated_krowgend(caplog):
    """Some users will use deprecated  krowend krogend,
    these values should be translated to kroend"""
    wog = PyscalFactory.create_water_oil_gas(
        dict(nw=2, now=3, ng=1, nog=2.5, krowend=0.6, krogend=0.7)
    )
    assert "deprecated" in caplog.text
    swof = wog.SWOF()
    assert "kroend=0.6" in swof
    sgof = wog.SGOF()
    assert "kroend=0.7" in sgof
    assert not wog.threephaseconsistency()
    sat_table_str_ok(swof)  # sgof code works for swof also currently
    sat_table_str_ok(sgof)
    assert "Corey krg" in sgof
    assert "Corey krog" in sgof
    assert "Corey krw" in swof
    assert "Corey krow" in swof


def test_factory_wateroilgas_wo():
    """Test making only wateroil through the wateroilgas factory"""
    factory = PyscalFactory()
    wog = factory.create_water_oil_gas(
        dict(nw=2, now=3, kroend=0.5, sorw=0.04, swcr=0.1)
    )
    swof = wog.SWOF()
    assert "Corey krw" in swof
    assert "krw" in wog.wateroil.table
    sat_table_str_ok(swof)
    check_table(wog.wateroil.table)
    assert wog.gasoil is None

    wog.SGOF()


def test_load_relperm_df(tmpdir):
    """Test loading of dataframes with validation from excel or from csv"""
    if "__file__" in globals():
        # Easen up copying test code into interactive sessions
        testdir = os.path.dirname(os.path.abspath(__file__))
    else:
        testdir = os.path.abspath(".")

    scalfile_xls = testdir + "/data/scal-pc-input-example.xlsx"

    scaldata = PyscalFactory.load_relperm_df(scalfile_xls)
    with pytest.raises(IOError):
        PyscalFactory.load_relperm_df("not-existing-file")

    assert "SATNUM" in scaldata
    assert "CASE" in scaldata
    assert not scaldata.empty

    tmpdir.chdir()
    scaldata.to_csv("scal-input.csv")
    scaldata_fromcsv = PyscalFactory.load_relperm_df("scal-input.csv")
    assert "CASE" in scaldata_fromcsv
    assert not scaldata_fromcsv.empty
    scaldata_fromdf = PyscalFactory.load_relperm_df(scaldata_fromcsv)
    assert "CASE" in scaldata_fromdf
    assert "SATNUM" in scaldata_fromdf
    assert len(scaldata_fromdf) == len(scaldata_fromcsv) == len(scaldata)

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

    relpermfile_xls = testdir + "/data/relperm-input-example.xlsx"
    relpermdata = PyscalFactory.load_relperm_df(relpermfile_xls)
    assert "TAG" in relpermdata
    assert "SATNUM" in relpermdata
    assert "satnum" not in relpermdata  # always converted to upper-case
    assert len(relpermdata) == 3
    swof_str = PyscalFactory.create_pyscal_list(relpermdata, h=0.2).SWOF()
    assert six.ensure_str("Åre 1.8") in six.ensure_str(swof_str)
    assert "SATNUM 2" in swof_str  # Autogenerated in SWOF, generated by factory
    assert "SATNUM 3" in swof_str
    assert "foobar" in swof_str  # Random string injected in xlsx.

    # Make a dummy text file
    with open("dummy.txt", "w") as fhandle:
        fhandle.write("foo\nbar, com")
    with pytest.raises(ValueError):
        PyscalFactory.load_relperm_df("dummy.txt")


def test_xls_factory():
    """Test/demonstrate how to go from data in an excel row to pyscal objects

    This test function predates the load_relperm_df() function, but can
    still be in here.
    """

    if "__file__" in globals():
        # Easen up copying test code into interactive sessions
        testdir = os.path.dirname(os.path.abspath(__file__))
    else:
        testdir = os.path.abspath(".")

    xlsxfile = testdir + "/data/scal-pc-input-example.xlsx"

    scalinput = pd.read_excel(xlsxfile).set_index(["SATNUM", "CASE"])

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


def test_create_lists():
    """Test the factory methods for making pyscal lists"""
    if "__file__" in globals():
        # Easen up copying test code into interactive sessions
        testdir = os.path.dirname(os.path.abspath(__file__))
    else:
        testdir = os.path.abspath(".")

    scalfile_xls = testdir + "/data/scal-pc-input-example.xlsx"
    scaldata = PyscalFactory.load_relperm_df(scalfile_xls)
    scalrec_list = PyscalFactory.create_scal_recommendation_list(scaldata)
    assert len(scalrec_list) == 3
    assert scalrec_list.pyscaltype == SCALrecommendation

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


def test_scalrecommendation():
    """Testing making SCAL rec from dict of dict."""

    factory = PyscalFactory()

    scal_input = {
        "low": {"nw": 2, "now": 4, "ng": 1, "nog": 2},
        "BASE": {"nw": 3, "NOW": 3, "ng": 1, "nog": 2},
        "high": {"nw": 4, "now": 2, "ng": 1, "nog": 3},
    }
    scal = factory.create_scal_recommendation(scal_input)
    # (not supported yet to make WaterOil only..)
    interp = scal.interpolate(-0.5)
    sat_table_str_ok(interp.SWOF())
    sat_table_str_ok(interp.SGOF())
    sat_table_str_ok(interp.SLGOF())
    sat_table_str_ok(interp.SOF3())
    check_table(interp.wateroil.table)
    check_table(interp.gasoil.table)

    incomplete1 = scal_input.copy()
    del incomplete1["BASE"]
    with pytest.raises(ValueError):
        factory.create_scal_recommendation(incomplete1)

    go_only = scal_input.copy()
    del go_only["low"]["now"]
    del go_only["low"]["nw"]
    gasoil = factory.create_scal_recommendation(go_only)
    assert gasoil.low.wateroil is None
    assert gasoil.base.wateroil is not None
    assert gasoil.high.wateroil is not None
    # SCALrecommendation of gasoil only works as long as you
    # don't try to ask for water data:
    assert "SGFN" in gasoil.interpolate(-0.4).SGFN()
    assert "SWOF" not in gasoil.interpolate(-0.2).SWOF()


def test_scalrecommendation_gaswater():
    """Testing making SCAL rec from dict of dict for gaswater input"""

    factory = PyscalFactory()

    scal_input = {
        "low": {"nw": 2, "ng": 1},
        "BASE": {"nw": 3, "ng": 1},
        "high": {"nw": 4, "ng": 1},
    }
    scal = factory.create_scal_recommendation(scal_input, h=0.2)
    interp = scal.interpolate(-0.5, h=0.2)
    sat_table_str_ok(interp.SWFN())
    sat_table_str_ok(interp.SGFN())
    check_table(interp.wateroil.table)
    check_table(interp.gasoil.table)


def test_xls_scalrecommendation():
    """Test making SCAL recommendations from xls data"""

    if "__file__" in globals():
        # Easen up copying test code into interactive sessions
        testdir = os.path.dirname(os.path.abspath(__file__))
    else:
        testdir = os.path.abspath(".")

    xlsxfile = testdir + "/data/scal-pc-input-example.xlsx"
    scalinput = pd.read_excel(xlsxfile).set_index(["SATNUM", "CASE"])
    print(scalinput)
    for satnum in scalinput.index.levels[0].values:
        dictofdict = scalinput.loc[satnum, :].to_dict(orient="index")
        print(dictofdict)
        scalrec = PyscalFactory.create_scal_recommendation(dictofdict)
        print(scalrec.interpolate(-0.5).SWOF())
        scalrec.interpolate(+0.5)


def test_no_gasoil():
    """The command client does not support two-phase gas-oil, because
    that is most likely an sign of a user input error.
    (misspelled other columns f.ex).

    Make sure we fail in that case."""
    dframe = pd.DataFrame(columns=["SATNUM", "NOW", "NG"], data=[[1, 2, 2]])
    with pytest.raises(ValueError):
        PyscalFactory.load_relperm_df(dframe)


def test_check_deprecated_krowgend(caplog):
    """Up until pyscal 0.5.x, krogend and krowend were parameters
    to the oil curve parametrization for WaterOil and GasOil. From
    pyscal 0.6.0, krogend and krowend are merged to kroend.
    """
    wateroil = PyscalFactory.create_water_oil(dict(swl=0.1, nw=2, now=2, krowend=0.4))
    assert "krowend" in caplog.text
    assert "deprecated" in caplog.text
    assert wateroil.table["krow"].max() == 0.4

    gasoil = PyscalFactory.create_gas_oil(dict(swl=0.1, ng=2, nog=2, krogend=0.4))
    assert "krogend" in caplog.text
    assert "deprecated" in caplog.text
    assert gasoil.table["krog"].max() == 0.4


def test_check_deprecated_kromax(caplog):
    """Up until pyscal 0.5.x, kromax was a parameter to the oil
    curve parametrization for WaterOil and GasOil, and kro was
    linear between swl and swcr. From pyscal 0.6, that linear
    segment is gone, and kromax is not needed as a parameter,
    only krowend/krogend is used.

    If kromax is encountered in input data, we should warn
    it is ignored.
    """
    PyscalFactory.create_water_oil(dict(swl=0.1, nw=2, now=2, kroend=0.4, kromax=0.5))
    assert "kromax" in caplog.text
    assert "deprecated" in caplog.text


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

    factory = PyscalFactory()

    # Example config line for gen_satfunc:
    conf_line_pc = "RELPERM 4 2 1 3 2 1 0.15 0.10 0.5 20 100 0.2 0.22 -0.5 30"

    wateroil = factory.create_water_oil(parse_gensatfuncline(conf_line_pc))
    swof = wateroil.SWOF()
    assert "0.17580" in swof  # krw at sw=0.65
    assert "0.0127" in swof  # krow at sw=0.65
    assert "Capillary pressure from normalized J-function" in swof
    assert "2.0669" in swof  # pc at swl

    conf_line_min = "RELPERM 1 2 3 1 2 3 0.1 0.15 0.5 20"
    wateroil = factory.create_water_oil(parse_gensatfuncline(conf_line_min))
    swof = wateroil.SWOF()
    assert "Zero capillary pressure" in swof

    conf_line_few = "RELPERM 1 2 3 1 2 3"
    with pytest.raises(ValueError):
        parse_gensatfuncline(conf_line_few)

    # sigma_costau is missing here:
    conf_line_almost_pc = "RELPERM 4 2 1 3 2 1 0.15 0.10 0.5 20 100 0.2 0.22 -0.5"
    wateroil = factory.create_water_oil(parse_gensatfuncline(conf_line_almost_pc))
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
    assert not factory.sufficient_gas_oil_params(dict())
    with pytest.raises(ValueError):
        factory.sufficient_gas_oil_params({"lg": 0}, failhard=True)
    assert not factory.sufficient_gas_oil_params({"lg": 0}, failhard=False)
    assert factory.sufficient_gas_oil_params(
        {"lg": 0, "eg": 0, "Tg": 0, "log": 0, "eog": 0, "tog": 0}
    )

    assert factory.sufficient_water_oil_params({"nw": 0, "now": 0})
    with pytest.raises(ValueError):
        factory.sufficient_water_oil_params({"nw": 0}, failhard=True)
    assert not factory.sufficient_water_oil_params(dict())
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
