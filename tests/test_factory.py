"""Test the PyscalFactory module"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import pandas as pd

import pytest

from pyscal import WaterOil, GasOil, PyscalFactory, factory

from common import sat_table_str_ok, check_table


def test_factory_wateroil():
    """Test that we can create curves from dictionaries of parameters"""

    logging.getLogger().setLevel("INFO")

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
    assert sat_table_str_ok(wateroil.SWOF())
    assert sat_table_str_ok(wateroil.SWFN())

    wateroil = factory.create_water_oil(
        dict(nw=3, now=2, sorw=0.1, krwend=0.2, krwmax=0.5)
    )
    assert isinstance(wateroil, WaterOil)
    assert "krw" in wateroil.table
    assert "Corey" in wateroil.krwcomment
    assert wateroil.table["krw"].max() == 0.5
    check_table(wateroil.table)
    assert sat_table_str_ok(wateroil.SWOF())
    assert sat_table_str_ok(wateroil.SWFN())

    # Ambiguous works, but we don't guarantee that this results
    # in LET or Corey.
    wateroil = factory.create_water_oil(dict(nw=3, Lw=2, Ew=2, Tw=2, now=3))
    assert "krw" in wateroil.table
    assert "Corey" in wateroil.krwcomment or "LET" in wateroil.krwcomment
    check_table(wateroil.table)
    assert sat_table_str_ok(wateroil.SWOF())
    assert sat_table_str_ok(wateroil.SWFN())

    wateroil = factory.create_water_oil(dict(Lw=2, Ew=2, Tw=2, krwend=1, now=4))
    assert isinstance(wateroil, WaterOil)
    assert "krw" in wateroil.table
    assert wateroil.table["krw"].max() == 1.0
    assert "LET" in wateroil.krwcomment
    check_table(wateroil.table)
    assert sat_table_str_ok(wateroil.SWOF())
    assert sat_table_str_ok(wateroil.SWFN())

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
    assert sat_table_str_ok(wateroil.SWOF())
    assert sat_table_str_ok(wateroil.SWFN())

    # Add capillary pressure
    wateroil = factory.create_water_oil(
        dict(swl=0.1, nw=1, now=1, a=2, b=-1, poro_ref=0.2, perm_ref=100, drho=200)
    )
    assert "pc" in wateroil.table
    assert wateroil.table["pc"].max() > 0.0
    assert "Simplified J" in wateroil.pccomment
    check_table(wateroil.table)
    assert sat_table_str_ok(wateroil.SWOF())
    assert sat_table_str_ok(wateroil.SWFN())

    # Test that the optional gravity g is picked up:
    wateroil = factory.create_water_oil(
        dict(swl=0.1, nw=1, now=1, a=2, b=-1, poro_ref=0.2, perm_ref=100, drho=200, g=0)
    )
    assert "pc" in wateroil.table
    assert wateroil.table["pc"].max() == 0.0
    check_table(wateroil.table)
    assert sat_table_str_ok(wateroil.SWOF())
    assert sat_table_str_ok(wateroil.SWFN())

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

    logging.getLogger().setLevel("INFO")

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
    assert sat_table_str_ok(sgof)
    check_table(gasoil.table)
    assert "Corey krg" in sgof
    assert "Corey krog" in sgof
    assert "Zero capillary pressure" in sgof

    gasoil = factory.create_gas_oil(
        dict(ng=1.2, nog=2, krgend=0.8, krgmax=0.9, krogend=0.6)
    )
    sgof = gasoil.SGOF()
    assert sat_table_str_ok(sgof)
    assert "kroend=0.6" in sgof
    assert "krgend=0.8" in sgof
    check_table(gasoil.table)

    gasoil = factory.create_gas_oil(dict(ng=1.3, Log=2, Eog=2, Tog=2))
    sgof = gasoil.SGOF()
    check_table(gasoil.table)
    assert sat_table_str_ok(sgof)
    assert "Corey krg" in sgof
    assert "LET krog" in sgof

    gasoil = factory.create_gas_oil(dict(Lg=1, Eg=1, Tg=1, Log=2, Eog=2, Tog=2))
    sgof = gasoil.SGOF()
    assert sat_table_str_ok(sgof)
    check_table(gasoil.table)
    assert "LET krg" in sgof
    assert "LET krog" in sgof


def test_factory_wateroilgas():
    """Test creating discrete cases of WaterOilGas from factory"""
    logging.getLogger().setLevel("INFO")

    factory = PyscalFactory()

    wog = factory.create_water_oil_gas(dict(nw=2, now=3, ng=1, nog=2.5))
    swof = wog.SWOF()
    sgof = wog.SGOF()
    assert sat_table_str_ok(swof)  # sgof code works for swof also currently
    assert sat_table_str_ok(sgof)
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
    assert sat_table_str_ok(swof)  # sgof code works for swof also currently
    assert sat_table_str_ok(sgof)
    assert "Corey krg" in sgof
    assert "Corey krog" in sgof
    assert "Corey krw" in swof
    assert "Corey krow" in swof

    # Mangling data
    with pytest.raises(ValueError):
        factory.create_water_oil_gas(dict(nw=2, now=3, ng=1))


def test_factory_wateroilgas_wo():
    """Test making only wateroil through the wateroilgas factory"""
    factory = PyscalFactory()
    wog = factory.create_water_oil_gas(
        dict(nw=2, now=3, krowend=0.5, sorw=0.04, swcr=0.1)
    )
    swof = wog.SWOF()
    assert "Corey krw" in swof
    assert "krw" in wog.wateroil.table
    assert sat_table_str_ok(swof)
    check_table(wog.wateroil.table)
    assert wog.gasoil is None

    wog.SGOF()


def test_xls_factory():
    """Test/demonstrate how to go from data in an excel row to pyscal objects"""

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
        assert sat_table_str_ok(sgof)
        assert "LET krg" in sgof
        assert "LET krog" in sgof


def test_scalrecommendation():
    """Testing making SCAL rec from dict of dict"""

    factory = PyscalFactory()

    scal_input = {
        "low": {"nw": 2, "now": 4, "ng": 1, "nog": 2},
        "BASE": {"nw": 3, "NOW": 3, "ng": 1, "nog": 2},
        "high": {"nw": 4, "now": 2, "ng": 1, "nog": 3},
    }
    scal = factory.create_scal_recommendation(scal_input)
    # (not supported yet to make WaterOil only..)
    scal.interpolate(-0.5).SWOF()

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
    interp = scal.interpolate(-0.5)
    assert sat_table_str_ok(interp.SWOF())
    assert sat_table_str_ok(interp.SGOF())
    assert sat_table_str_ok(interp.SLGOF())
    assert sat_table_str_ok(interp.SOF3())
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
    # data for krg, then we error hard. If the user did not provide
    # any data for GasOil, then the code returns False
    with pytest.raises(ValueError):
        factory.sufficient_gas_oil_params({"ng": 0})
    assert not factory.sufficient_gas_oil_params(dict())
    with pytest.raises(ValueError):
        factory.sufficient_gas_oil_params({"lg": 0})
    assert factory.sufficient_gas_oil_params(
        {"lg": 0, "eg": 0, "Tg": 0, "log": 0, "eog": 0, "tog": 0}
    )

    assert factory.sufficient_water_oil_params({"nw": 0, "now": 0})
    with pytest.raises(ValueError):
        factory.sufficient_water_oil_params({"nw": 0})
    assert not factory.sufficient_water_oil_params(dict())
    with pytest.raises(ValueError):
        factory.sufficient_water_oil_params({"lw": 0})
    assert factory.sufficient_water_oil_params(
        {"lw": 0, "ew": 0, "Tw": 0, "low": 0, "eow": 0, "tow": 0}
    )
