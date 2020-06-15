"""Test the PyscalList module"""
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import pandas as pd

import pytest

from pyscal import (
    WaterOil,
    GasOil,
    GasWater,
    WaterOilGas,
    PyscalFactory,
    PyscalList,
    SCALrecommendation,
)

from common import sat_table_str_ok

try:
    import ecl2df

    HAVE_ECL2DF = True
except ImportError:
    HAVE_ECL2DF = False


def test_pyscallist_basic():
    """Test that the class acts like a list"""
    p_list = PyscalList()
    assert isinstance(p_list, PyscalList)
    assert not p_list

    p_list.append(None)
    assert not p_list
    p_list.append([])
    assert not p_list

    with pytest.raises(ValueError):
        p_list.append(1)
    with pytest.raises(ValueError):
        p_list.append("foo")

    p_list.append(WaterOil())
    assert len(p_list) == 1
    assert isinstance(p_list[1], WaterOil)
    with pytest.raises(IndexError):
        # pylint: disable=W0104
        p_list[0]
    with pytest.raises(IndexError):
        # pylint: disable=W0104
        p_list[2]
    with pytest.raises(ValueError):
        p_list.append(GasOil())
    assert len(p_list) == 1

    p_list.append(WaterOil())
    assert len(p_list) == 2


def test_load_scalrec():
    """Load a SATNUM range from xlsx"""
    if "__file__" in globals():
        # Easen up copying test code into interactive sessions
        testdir = os.path.dirname(os.path.abspath(__file__))
    else:
        testdir = os.path.abspath(".")

    scalrec_data = PyscalFactory.load_relperm_df(
        testdir + "/data/scal-pc-input-example.xlsx"
    )
    scalrec_list = PyscalFactory.create_scal_recommendation_list(scalrec_data)
    wog_list = scalrec_list.interpolate(-0.3)

    with pytest.raises((AssertionError, ValueError)):
        scalrec_list.interpolate(-1.1)
    assert wog_list.pyscaltype == WaterOilGas

    with pytest.raises(TypeError):
        scalrec_list.dump_family_1()
    with pytest.raises(TypeError):
        scalrec_list.dump_family_2()
    with pytest.raises(TypeError):
        scalrec_list.SWOF()

    scalrec_list.interpolate([-1, 1, 0])
    with pytest.raises((AssertionError, ValueError)):
        scalrec_list.interpolate([-1, 0, 1.1])

    # Not enough interpolation parameters
    with pytest.raises(ValueError):
        scalrec_list.interpolate([-1, 1])
    with pytest.raises(ValueError):
        scalrec_list.interpolate([-1, 1, 0, 0])

    scalrec_list.interpolate(-1, 1)
    scalrec_list.interpolate(-1, [0, -1, 1])

    with pytest.raises(ValueError):
        scalrec_list.interpolate(1, [-1, 1, 0, 0])


def test_df():
    """Test dataframe dumps"""
    if "__file__" in globals():
        # Easen up copying test code into interactive sessions
        testdir = os.path.dirname(os.path.abspath(__file__))
    else:
        testdir = os.path.abspath(".")

    scalrec_data = PyscalFactory.load_relperm_df(
        testdir + "/data/scal-pc-input-example.xlsx"
    )
    scalrec_list = PyscalFactory.create_scal_recommendation_list(scalrec_data)
    wog_list = scalrec_list.interpolate(-0.3)

    # Test dataframe dumps:
    dframe = scalrec_list.df()
    assert "SG" in dframe
    assert "KRG" in dframe
    assert "KROG" in dframe
    assert "PCOW" in dframe
    if "PCOG" in dframe:
        # Allow PCOG to be included later.
        assert dframe["PCOG"].sum() == 0
    assert "KRW" in dframe
    assert "KROW" in dframe
    assert "SATNUM" in dframe
    assert dframe["SATNUM"].min() == 1
    assert len(dframe["SATNUM"].unique()) == len(scalrec_list)
    assert set(dframe["CASE"]) == set(["pess", "base", "opt"])
    assert dframe["SATNUM"].max() == len(scalrec_list)
    if HAVE_ECL2DF:
        # Test using ecl2df to do the include file printing. First we need to
        # massage the dataframe into what ecl2df can handle:
        base_df_swof = (
            dframe.set_index("CASE")
            .loc["base"][["SW", "KRW", "KROW", "PCOW", "SATNUM"]]
            .assign(KEYWORD="SWOF")
            .dropna()
            .reset_index(drop=True)
        )
        ecl_inc = ecl2df.satfunc.df2ecl(base_df_swof)
        dframe_from_inc = ecl2df.satfunc.df(ecl_inc)
        pd.testing.assert_frame_equal(base_df_swof, dframe_from_inc)

        # Test also SGOF
        base_df_sgof = (
            dframe.set_index("CASE")
            .loc["base"][["SG", "KRG", "KROG", "SATNUM"]]
            .assign(KEYWORD="SGOF", PCOG=0.0)
            .dropna()
            .reset_index(drop=True)
        )
        ecl_inc = ecl2df.satfunc.df2ecl(base_df_sgof)
        dframe_from_inc = ecl2df.satfunc.df(ecl_inc)
        pd.testing.assert_frame_equal(base_df_sgof, dframe_from_inc, check_like=True)

    # WaterOilGasList:
    dframe = wog_list.df()
    assert "SG" in dframe
    assert "KRG" in dframe
    assert "KROG" in dframe
    assert "PCOW" in dframe
    assert "PCOG" in dframe  # This gets included through interpolation
    assert "KRW" in dframe
    assert "KROW" in dframe
    assert "SATNUM" in dframe
    assert dframe["SATNUM"].min() == 1
    assert len(dframe["SATNUM"].unique()) == len(wog_list)
    assert "CASE" not in dframe
    assert dframe["SATNUM"].max() == len(wog_list)

    # WaterOil list
    input_dframe = pd.DataFrame(columns=["SATNUM", "Nw", "Now"], data=[[1, 2, 2]])
    relperm_data = PyscalFactory.load_relperm_df(input_dframe)
    p_list = PyscalFactory.create_pyscal_list(relperm_data, h=0.2)
    dframe = p_list.df()
    assert "SW" in dframe
    assert "KRW" in dframe
    assert "KROW" in dframe
    assert "PCOW" not in dframe  # to be interpreted as zero
    assert "SATNUM" in dframe
    assert len(dframe.columns) == 4
    assert not dframe.empty

    # GasOil list
    input_dframe = pd.DataFrame(columns=["SATNUM", "Ng", "Nog"], data=[[1, 2, 2]])
    relperm_data = PyscalFactory.load_relperm_df(input_dframe)
    p_list = PyscalFactory.create_pyscal_list(relperm_data, h=0.2)
    dframe = p_list.df()
    assert "SG" in dframe
    assert "KRG" in dframe
    assert "KROG" in dframe
    assert "PCOG" not in dframe  # to be interpreted as zero
    assert "SATNUM" in dframe
    assert len(dframe.columns) == 4
    assert not dframe.empty


def test_load_scalrec_tags():
    """Test tag handling for a SCAL recommendation with SATNUM range"""
    if "__file__" in globals():
        # Easen up copying test code into interactive sessions
        testdir = os.path.dirname(os.path.abspath(__file__))
    else:
        testdir = os.path.abspath(".")

    scalrec_data = PyscalFactory.load_relperm_df(
        testdir + "/data/scal-pc-input-example.xlsx"
    )
    scalrec_list = PyscalFactory.create_scal_recommendation_list(scalrec_data)

    wog_list = scalrec_list.interpolate(-1)

    swof = wog_list.SWOF()
    assert swof.count("SCAL recommendation interpolation to -1") == 3
    assert swof.count("SATNUM 1") == 1
    assert swof.count("SATNUM") == 3

    sof3 = wog_list.SOF3()
    assert sof3.count("SCAL recommendation interpolation to -1") == 3
    assert sof3.count("SATNUM 1") == 1
    assert sof3.count("SATNUM") == 3

    assert (
        scalrec_list.interpolate(1)
        .SWOF()
        .count("SCAL recommendation interpolation to 1\n")
        == 3
    )
    assert (
        scalrec_list.interpolate(0)
        .SWOF()
        .count("SCAL recommendation interpolation to 0\n")
        == 3
    )
    assert (
        scalrec_list.interpolate(-0.444)
        .SWOF()
        .count("SCAL recommendation interpolation to -0.444\n")
        == 3
    )

    # Give each row in the SCAL recommendation its own tag (low, base, high
    # explicitly in each tag, someone will do that)
    scalrec_data["TAG"] = [
        "SAT1 low",
        "SAT1 base",
        "SAT1 high",
        "SAT2 pess",
        "SAT2 base",
        "SAT2 opt",
        "SAT3 pessimistic",
        "SAT3 base case",
        "SAT3 optimistic",
    ]
    scalrec_list2 = PyscalFactory.create_scal_recommendation_list(scalrec_data)
    swof = scalrec_list2.interpolate(-0.5, h=0.2).SWOF()
    assert swof.count("SCAL recommendation") == 3
    for tag in scalrec_data["TAG"]:
        assert swof.count(tag) == 1


def test_dump():
    """Test dumping Eclipse include data to file"""
    if "__file__" in globals():
        # Easen up copying test code into interactive sessions
        testdir = os.path.dirname(os.path.abspath(__file__))
    else:
        testdir = os.path.abspath(".")

    relperm_data = PyscalFactory.load_relperm_df(
        testdir + "/data/relperm-input-example.xlsx"
    )
    pyscal_list = PyscalFactory.create_pyscal_list(relperm_data)

    fam1 = pyscal_list.dump_family_1()
    sat_table_str_ok(fam1)

    fam2 = pyscal_list.dump_family_2()
    sat_table_str_ok(fam2)


def test_capillary_pressure():
    """Test that we recognize capillary pressure parametrizations"""
    dframe = pd.DataFrame(
        columns=[
            "SATNUM",
            "nw",
            "now",
            "swl",
            "a",
            "b",
            "PORO_reF",
            "PERM_ref",
            "drho",
        ],
        data=[[1, 1, 1, 0.05, 3.6, -3.5, 0.25, 15, 150]],
    )
    pyscal_list = PyscalFactory.create_pyscal_list(
        PyscalFactory.load_relperm_df(dframe)
    )
    swof = pyscal_list.dump_family_1()
    assert "Simplified J-function" in swof
    assert "petrophysical" not in swof

    dframe = pd.DataFrame(
        columns=[
            "SATNUM",
            "nw",
            "now",
            "swl",
            "a_petro",
            "b_petro",
            "PORO_reF",
            "PERM_ref",
            "drho",
        ],
        data=[[1, 1, 1, 0.05, 3.6, -3.5, 0.25, 15, 150]],
    )
    pyscal_list = PyscalFactory.create_pyscal_list(
        PyscalFactory.load_relperm_df(dframe)
    )
    swof = pyscal_list.dump_family_1()
    assert "Simplified J-function" in swof
    assert "petrophysical" in swof

    dframe = pd.DataFrame(
        columns=[
            "SATNUM",
            "nw",
            "now",
            "swl",
            "a",
            "b",
            "PORO",
            "PERM",
            "sigma_COStau",
        ],
        data=[[1, 1, 1, 0.05, 3.6, -3.5, 0.25, 15, 30]],
    )
    pyscal_list = PyscalFactory.create_pyscal_list(
        PyscalFactory.load_relperm_df(dframe)
    )
    swof = pyscal_list.dump_family_1()
    assert "normalized J-function" in swof
    assert "sigma_costau" in swof
    assert "petrophysical" not in swof


def test_twophase():
    """Test interpolation for a two-phase setup water-oil"""
    # Triggers bug in pyscal <= 0.5.0
    dframe = pd.DataFrame(
        columns=["SATNUM", "CASE", "NW", "NOW", "TAG"],
        data=[
            [1, "pess", 1, 1, "thetag"],
            [1, "base", 2, 2, "thetag"],
            [1, "opt", 3, 3, "thetag"],
        ],
    )
    pyscal_list = PyscalFactory.create_scal_recommendation_list(
        PyscalFactory.load_relperm_df(dframe)
    )
    pyscal_list = pyscal_list.interpolate(-0.5)

    # This works, but provides an error message that Gas.
    # The correct usage would be to call .SWOF(), but the pyscal
    # client calls dump_family_1 on two-phase problems.
    swof = pyscal_list.dump_family_1()
    assert "thetag" in swof
    assert "SWOF" in swof
    assert "SGOF" not in swof


def test_gaswater():
    """Test list of gas-water objects"""
    dframe = pd.DataFrame(
        columns=["SATNUM", "NW", "NG", "TAG"],
        data=[[1, 2, 2, "thetag"], [2, 3, 3, "othertag"]],
    )
    pyscal_list = PyscalFactory.create_pyscal_list(
        PyscalFactory.load_relperm_df(dframe), h=0.1
    )
    assert pyscal_list.pyscaltype == GasWater
    dump = pyscal_list.dump_family_2()
    assert "SATNUM 2" in dump
    assert "SATNUM 1" in dump
    assert "SATNUM 3" not in dump
    assert "SWFN" in dump
    assert "SGFN" in dump
    assert "othertag" in dump
    assert "thetag" in dump

    with pytest.raises(ValueError):
        # Does not make sense for GasWater:
        pyscal_list.dump_family_1()


def test_gaswater_scal():
    """Test list of gas-water objects in scal recommendation"""
    dframe = pd.DataFrame(
        columns=["SATNUM", "CASE", "NW", "NG", "TAG"],
        data=[
            [1, "pess", 2, 2, "thetag"],
            [1, "base", 3, 3, "thetag"],
            [1, "opt", 4, 4, "thetag"],
        ],
    )
    pyscal_list = PyscalFactory.create_scal_recommendation_list(
        PyscalFactory.load_relperm_df(dframe), h=0.1
    )
    assert pyscal_list.pyscaltype == SCALrecommendation


def test_explicit_df():
    """Test some dataframes, check the error messages given"""

    dframe = pd.DataFrame(columns=["satnum"], data=[[1], [2]])
    with pytest.raises(ValueError):
        # SATNUM column must be upper case, or should we allow lowercase??
        relperm_data = PyscalFactory.load_relperm_df(dframe)

    dframe = pd.DataFrame(columns=["SATNUM"], data=[[0], [1]])
    with pytest.raises(ValueError):
        # SATNUM must start at 1.
        relperm_data = PyscalFactory.load_relperm_df(dframe)

    dframe = pd.DataFrame(columns=["SATNUM"], data=[[1], ["foo"]])
    with pytest.raises(ValueError):
        # SATNUM must contain only integers
        relperm_data = PyscalFactory.load_relperm_df(dframe)

    dframe = pd.DataFrame(
        columns=["SATNUM", "nw", "now"], data=[[1.01, 1, 1], [2.01, 1, 1]]
    )
    # This one will pass, as these can be converted to ints
    relperm_data = PyscalFactory.load_relperm_df(dframe)

    dframe = pd.DataFrame(
        columns=["SATNUM", "nw", "now"], data=[[1.01, 1, 1], [1.3, 1, 1]]
    )
    with pytest.raises(ValueError):
        # complains about non-uniqueness in SATNUM
        relperm_data = PyscalFactory.load_relperm_df(dframe)

    dframe = pd.DataFrame(columns=["SATNUM"], data=[[1], [2]])
    with pytest.raises(ValueError):
        # Not enough data
        relperm_data = PyscalFactory.load_relperm_df(dframe)

    # Minimal amount of data:
    dframe = pd.DataFrame(columns=["SATNUM", "Nw", "Now"], data=[[1, 2, 2]])
    relperm_data = PyscalFactory.load_relperm_df(dframe)
    p_list = PyscalFactory.create_pyscal_list(relperm_data, h=0.2)
    p_list.dump_family_1()
    with pytest.raises(ValueError):
        p_list.dump_family_2()

    # Case insensitive for parameters
    dframe = pd.DataFrame(columns=["SATNUM", "nw", "now"], data=[[1, 2, 2]])
    relperm_data = PyscalFactory.load_relperm_df(dframe)
    p_list = PyscalFactory.create_pyscal_list(relperm_data, h=0.2)
    p_list.dump_family_1()

    # Minimal wateroilgas
    dframe = pd.DataFrame(
        columns=["SATNUM", "Nw", "Now", "ng", "nOG"], data=[[1, 2, 2, 2, 2]]
    )
    relperm_data = PyscalFactory.load_relperm_df(dframe)
    p_list = PyscalFactory.create_pyscal_list(relperm_data, h=0.2)
    relperm_str = p_list.dump_family_1()
    assert "SWOF" in relperm_str
    assert "SGOF" in relperm_str
    assert "SLGOF" not in relperm_str
    assert "SOF3" not in relperm_str

    # Minimal wateroilgas with pc
    dframe = pd.DataFrame(
        columns=[
            "SATNUM",
            "swl",
            "Nw",
            "Now",
            "ng",
            "nOG",
            "a",
            "b",
            "poro_ref",
            "perm_ref",
            "drho",
        ],
        data=[[1, 0.1, 2, 2, 2, 2, 1.5, -0.5, 0.1, 100, 300]],
    )
    relperm_data = PyscalFactory.load_relperm_df(dframe)
    p_list = PyscalFactory.create_pyscal_list(relperm_data, h=0.2)
    relperm_str = p_list.dump_family_1()
    assert "SWOF" in relperm_str
    assert "Simplified" in relperm_str  # Bad practice, testing for stuff in comments
    assert "a=1.5" in relperm_str


def test_comments_df():
    """Test that we support a tag column in the dataframe, and
    that we are able to handle UTF-8 stuff nicely in py2-3
    """
    dframe = pd.DataFrame(
        columns=["SATNUM", "tag", "Nw", "Now", "ng", "nOG"],
        data=[[1, "thisisacomment", 2, 2, 2, 2]],
    )
    relperm_data = PyscalFactory.load_relperm_df(dframe)
    p_list = PyscalFactory.create_pyscal_list(relperm_data, h=0.2)
    relperm_str = p_list.dump_family_1()
    assert "thisisacomment" in relperm_str

    # tag vs comment in dataframe should not matter.
    dframe = pd.DataFrame(
        columns=["SATNUM", "comment", "Nw", "Now", "ng", "nOG"],
        data=[[1, "thisisacomment", 2, 2, 2, 2]],
    )
    relperm_data = PyscalFactory.load_relperm_df(dframe)
    p_list = PyscalFactory.create_pyscal_list(relperm_data, h=0.2)
    relperm_str = p_list.dump_family_1()
    assert relperm_str.count("thisisacomment") == 2

    # Check case insensitiveness:
    dframe = pd.DataFrame(
        columns=["SAtnUM", "coMMent", "Nw", "Now", "ng", "nOG"],
        data=[[1, "thisisacomment", 2, 2, 2, 2]],
    )
    relperm_data = PyscalFactory.load_relperm_df(dframe)
    p_list = PyscalFactory.create_pyscal_list(relperm_data, h=0.2)
    assert p_list.dump_family_1().count("thisisacomment") == 2

    # UTF-8 stuff:
    dframe = pd.DataFrame(
        columns=["SATNUM", "TAG", "Nw", "Now", "Ng", "Nog"],
        data=[[1, "æøå", 2, 2, 2, 2]],
    )
    relperm_data = PyscalFactory.load_relperm_df(dframe)
    p_list = PyscalFactory.create_pyscal_list(relperm_data, h=0.2)
    assert p_list.dump_family_1().count("æøå") == 2
