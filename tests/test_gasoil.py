"""Test module for GasOil objects"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

from hypothesis import given, settings
import hypothesis.strategies as st

from common import (
    float_df_checker,
    check_table,
    sat_table_str_ok,
    check_linear_sections,
)

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


@given(st.text())
def test_gasoil_tag(tag):
    """Test tagging of GasOil objects,
    that we are not able to produce something that
    can crash Eclipse"""
    gasoil = GasOil(h=0.5, tag=tag)
    gasoil.add_corey_gas()
    gasoil.add_corey_oil()
    sat_table_str_ok(gasoil.SGOF())
    sat_table_str_ok(gasoil.SGFN())


@settings(deadline=500)
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

    # Check that son is 1 at sg=0
    assert float_df_checker(gasoil.table, "sg", 0, "son", 1)

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


@settings(deadline=1000)
@given(
    st.floats(min_value=0, max_value=0.3),  # swl
    st.floats(min_value=0, max_value=0.3),  # sgcr
    st.floats(min_value=0, max_value=0.41),  # sorg (sgn collapses when >0.4)
    st.floats(min_value=0.1, max_value=1),  # kroend
    st.floats(min_value=0.1, max_value=1),  # krgend
    st.floats(min_value=0.2, max_value=1),  # krgmax
    st.floats(min_value=0.0001, max_value=1),  # h
    st.booleans(),  # fast mode
)
def test_gasoil_krendmax(swl, sgcr, sorg, kroend, krgend, krgmax, h, fast):
    """Test that krendmax gets correct in all numerical corner cases.

    The normalized sg-range is allowed to collapse to nothing in this test.
    (causes AssertionError)
    """
    try:
        gasoil = GasOil(swl=swl, sgcr=sgcr, sorg=sorg, h=h, tag="", fast=fast)
    except AssertionError:
        return
    krgend = min(krgend, krgmax)
    gasoil.add_corey_oil(kroend=kroend)
    gasoil.add_corey_gas(krgend=krgend, krgmax=krgmax)
    check_table(gasoil.table)
    check_linear_sections(gasoil)
    assert gasoil.selfcheck()
    check_endpoints(gasoil, krgend, krgmax, kroend)
    assert 0 < gasoil.crosspoint() < 1

    # Redo with krgendanchor not defaulted
    gasoil = GasOil(swl=swl, sgcr=sgcr, sorg=sorg, h=h, krgendanchor="", tag="")
    gasoil.add_corey_oil(kroend=kroend)
    gasoil.add_corey_gas(krgend=krgend, krgmax=krgmax)
    check_table(gasoil.table)
    check_linear_sections(gasoil)
    assert gasoil.selfcheck()
    check_endpoints(gasoil, krgend, krgmax, kroend)
    assert 0 < gasoil.crosspoint() < 1

    # Redo with LET:
    gasoil = GasOil(swl=swl, sgcr=sgcr, sorg=sorg, h=h, tag="")
    gasoil.add_LET_oil(t=1.1, kroend=kroend)
    gasoil.add_LET_gas(krgend=krgend, krgmax=krgmax)
    check_table(gasoil.table)
    check_linear_sections(gasoil)
    assert gasoil.selfcheck()
    check_endpoints(gasoil, krgend, krgmax, kroend)
    assert 0 < gasoil.crosspoint() < 1


def check_endpoints(gasoil, krgend, krgmax, kroend):
    """Discrete tests that endpoints get numerically correct"""
    swtol = 1 / SWINTEGERS

    # Check endpoints for oil curve:
    # krog at sgcr should be kroend
    if gasoil.sgcr > swtol:
        assert float_df_checker(gasoil.table, "son", 1.0, "krog", kroend)
    # krog at son=0 (1-sorg-swl or 1 - swl) should be zero:
    assert float_df_checker(gasoil.table, "son", 0.0, "krog", 0)

    assert float_df_checker(gasoil.table, "sg", 0, "krog", kroend)

    assert float_df_checker(gasoil.table, "sgn", 0.0, "krg", 0)
    assert float_df_checker(gasoil.table, "sg", gasoil.sgcr, "krg", 0)

    # If krgendanchor == "sorg" then krgmax is irrelevant.
    if gasoil.sorg > swtol and gasoil.sorg > gasoil.h and gasoil.krgendanchor == "sorg":
        assert float_df_checker(gasoil.table, "sgn", 1.0, "krg", krgend)
        assert np.isclose(gasoil.table["krg"].max(), krgmax)
    if gasoil.krgendanchor != "sorg":
        assert np.isclose(gasoil.table["krg"].max(), krgend)


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
    check_linear_sections(gasoil)
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
    check_linear_sections(gasoil)
    assert gasoil.selfcheck()

    # kg should be < 1 at 1 - sorg due to krgendanchor being ""
    assert (
        gasoil.table[np.isclose(gasoil.table["sg"], 1 - gasoil.sorg)]["krg"].values[0]
        < 1.0
    )
    assert gasoil.table[np.isclose(gasoil.table["sg"], 1.0)]["krg"].values[0] == 1.0


def test_linearsegments():
    """Made for testing the linear segments during
    the resolution of issue #163"""
    gasoil = GasOil(h=0.01, swl=0.1, sgcr=0.3, sorg=0.3)
    gasoil.add_corey_oil(nog=10, kroend=0.5)
    gasoil.add_corey_gas(ng=10, krgend=0.5)
    check_table(gasoil.table)
    check_linear_sections(gasoil)
    # gasoil.plotkrgkrog(marker="*")


def test_kroend():
    """Manual testing of kromax and kroend behaviour"""
    gasoil = GasOil(swirr=0.01, sgcr=0.01, h=0.01, swl=0.1, sorg=0.05)
    gasoil.add_LET_gas()
    gasoil.add_LET_oil(2, 2, 2.1)
    assert gasoil.table["krog"].max() == 1
    gasoil.add_LET_oil(2, 2, 2.1, kroend=0.5)
    check_linear_sections(gasoil)
    assert gasoil.table["krog"].max() == 0.5

    assert 0 < gasoil.crosspoint() < 1

    gasoil.add_corey_oil(2)
    assert gasoil.table["krog"].max() == 1
    gasoil.add_corey_oil(nog=2, kroend=0.5)
    assert gasoil.table["krog"].max() == 0.5


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
    sat_table_str_ok(sgofstr)

    gasoil.resetsorg()
    check_table(gasoil.table)
    check_linear_sections(gasoil)
    sgofstr = gasoil.SGOF()
    assert len(sgofstr) > 100
    sat_table_str_ok(sgofstr)


def test_comments():
    """Test that the outputters include endpoints in comments"""
    gasoil = GasOil(h=0.3)
    gasoil.add_corey_gas()
    gasoil.add_corey_oil()
    sgfn = gasoil.SGFN()
    assert "--" in sgfn
    assert "pyscal: " in sgfn  # part of version string
    assert "swirr=0" in sgfn
    assert "sgcr=0" in sgfn
    assert "swl=0" in sgfn
    assert "sorg=0" in sgfn
    assert "ng=2" in sgfn
    assert "krgend=1" in sgfn
    assert "Corey" in sgfn
    assert "krg = krog @ sg=0.5" in sgfn
    assert "Zero capillary pressure" in sgfn
    assert "SG" in sgfn
    assert "KRG" in sgfn
    assert "PC" in sgfn

    sgof = gasoil.SGOF()
    assert "--" in sgof
    assert "pyscal: " in sgof  # part of version string
    assert "swirr=0" in sgof
    assert "sgcr=0" in sgof
    assert "swl=0" in sgof
    assert "sorg=0" in sgof
    assert "ng=2" in sgof
    assert "nog=2" in sgof
    assert "krgend=1" in sgof
    assert "Corey" in sgof
    assert "krg = krog @ sg=0.5" in sgof
    assert "Zero capillary pressure" in sgof
    assert "SG" in sgof
    assert "KRG" in sgof
    assert "KROG" in sgof
    assert "PC" in sgof


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
    check_linear_sections(gasoil)
    sgofstr = gasoil.SGOF()
    assert len(sgofstr) > 100
    sat_table_str_ok(sgofstr)


def test_sgfn():
    """Test that we can call SGFN without oil relperm defined"""
    gasoil = GasOil()
    gasoil.add_corey_gas()
    sgfn_str = gasoil.SGFN()
    assert "SGFN" in sgfn_str
    assert len(sgfn_str) > 15
    print(sgfn_str)


def test_roundoff():
    """Test robustness to monotonicity issues arising from
    representation errors

    https://docs.python.org/3/tutorial/floatingpoint.html#representation-error

    The dataframe injected in this function has occured in the wild, and
    caused fatal errors in Eclipse100. The error lies in
    pd.dataframe.to_csv(float_format=".7f") which does truncation of floating points
    instead of rounding (intentional). Since we have a strict dependency on
    monotonicity properties for Eclipse to work, the data must be rounded
    before being sent to to_csv(). This is being done in the .SGOF() and SWOF() as
    it is a representation issue, not a numerical issues in the objects themselves.
    """

    gasoil = GasOil()
    # Inject a custom dataframe that has occured in the wild,
    # and given monotonicity issues in GasOil.SGOF().
    gasoil.table = pd.DataFrame(
        columns=["sg", "krg", "krog", "pc"],
        data=[
            [0.02, 0, 0.19524045000000001, 0],
            [0.040000000000000001, 0, 0.19524044999999998, 0],
            [0.059999999999999998, 0, 0.19524045000000004, 0],
            [0.080000000000000002, 0, 0.19524045000000001, 0],
            [0.10000000000000001, 0, 0.19524045000000001, 0],
            [0.16, 0, 0.19524045000000001, 0],
            [0.17999999999999999, 0, 0.19524045000000001, 0],
            [0.19999999999999998, 0, 0.19524044999999998, 0],
            [0.22, 0, 0.19524045000000001, 0],
            [1, 1, 0, 0],
        ],
    )
    gasoil.table["sgn"] = gasoil.table["sg"]
    gasoil.table["son"] = 1 - gasoil.table["sg"]
    # If this value (as string) occurs, then we are victim of floating point truncation
    # in float_format=".7f":
    assert "0.1952404" not in gasoil.SGOF()
    assert "0.1952405" in gasoil.SGOF()
    check_table(gasoil.table)  # This function allows this monotonicity hiccup.
