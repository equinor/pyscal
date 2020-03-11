"""Test the PyscalList module"""
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import pandas as pd

import pytest

from pyscal import WaterOil, GasOil, WaterOilGas, PyscalFactory, PyscalList

from common import sat_table_str_ok


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

    wog_list = scalrec_list.interpolate(-1)
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


def test_dump():
    """Test dumping Eclipse include dataa to file"""
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
    assert sat_table_str_ok(fam1)

    fam2 = pyscal_list.dump_family_2()
    assert sat_table_str_ok(fam2)


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
    assert "thisisacomment" in relperm_str

    # Check case insensitiveness:
    dframe = pd.DataFrame(
        columns=["SAtnUM", "coMMent", "Nw", "Now", "ng", "nOG"],
        data=[[1, "thisisacomment", 2, 2, 2, 2]],
    )
    relperm_data = PyscalFactory.load_relperm_df(dframe)
    p_list = PyscalFactory.create_pyscal_list(relperm_data, h=0.2)
    relperm_str = p_list.dump_family_1()

    # UTF-8 stuff:
    dframe = pd.DataFrame(
        columns=["SATNUM", "TAG", "Nw", "Now", "Ng", "Nog"],
        data=[[1, "æøå", 2, 2, 2, 2]],
    )
    relperm_data = PyscalFactory.load_relperm_df(dframe)
    p_list = PyscalFactory.create_pyscal_list(relperm_data, h=0.2)
    relperm_str = p_list.dump_family_1()
    assert "æøå" in relperm_str
