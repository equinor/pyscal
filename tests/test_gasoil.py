# -*- coding: utf-8 -*-
"""Test module for relperm"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from hypothesis import given, settings
import hypothesis.strategies as st

from test_wateroil import float_df_checker

from pyscal import GasOil
from pyscal.constants import SWINTEGERS


def series_decreasing(series):
    """Weaker than pd.Series.is_monotonic_decreasing,
    allows constant parts"""
    return (series.diff().dropna() < 1e-8).all()


def series_increasing(series):
    """Weaker than pd.Series.is_monotonic_increasing"""
    return (series.diff().dropna() > -1e-8).all()


def check_table(df):
    """Check sanity of important columns"""
    assert not df.empty
    assert not df.isnull().values.any()
    assert len(df["sg"].unique()) == len(df)
    assert df["sg"].is_monotonic
    assert (df["sg"] >= 0.0).all()
    assert df["sgn"].is_monotonic
    assert df["son"].is_monotonic_decreasing
    assert series_decreasing(df["krog"])
    assert series_increasing(df["krg"])
    if "pc" in df:
        assert series_decreasing(df["pc"])


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
    go = GasOil(h=1)
    go.add_corey_gas()
    go.add_corey_oil()
    assert np.isclose(go.crosspoint(), 0.5)
    assert len(go.table) == 2

    go = GasOil(swl=0.1, h=1)
    go.add_corey_gas()
    go.add_corey_oil()
    assert len(go.table) == 2
    assert np.isclose(go.crosspoint(), 0.45)
    assert np.isclose(go.table["sg"].min(), 0)
    assert np.isclose(go.table["sg"].max(), 0.9)


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
    go = GasOil(
        swirr=0.0, swl=swl, sgcr=sgcr, sorg=sorg, h=h, krgendanchor="sorg", tag=tag
    )
    assert not go.table.empty
    assert not go.table.isnull().values.any()

    # Check that son is 1 at sgcr
    assert float_df_checker(go.table, "sg", go.sgcr, "son", 1)

    # Check that son is 0 at sorg with this krgendanchor
    assert float_df_checker(go.table, "sg", 1 - go.sorg - go.swl, "son", 0)

    # Check that sgn is 0 at sgcr
    assert float_df_checker(go.table, "sg", go.sgcr, "sgn", 0)

    # Check that sgn is 1 at sorg
    assert float_df_checker(go.table, "sg", 1 - go.sorg - go.swl, "sgn", 1)

    # Redo with different krgendanchor
    go = GasOil(swirr=0.0, swl=swl, sgcr=sgcr, sorg=sorg, h=h, krgendanchor="", tag=tag)
    assert float_df_checker(go.table, "sg", 1 - go.swl, "sgn", 1)
    assert float_df_checker(go.table, "sg", go.sgcr, "sgn", 0)


@settings(deadline=500)
@given(
    st.floats(min_value=0, max_value=0.3),  # swl
    st.floats(min_value=0, max_value=0.3),  # sgcr
    st.floats(min_value=0, max_value=0.3),  # sorg
    st.floats(min_value=0.1, max_value=1),  # kroend
    st.floats(min_value=0.1, max_value=1),  # kromax
    st.floats(min_value=0.1, max_value=1),  # krgend
    st.floats(min_value=0.1, max_value=1),  # krgmax
    st.floats(min_value=0.0001, max_value=1),  # h
)
def test_gasoil_krendmax(swl, sgcr, sorg, kroend, kromax, krgend, krgmax, h):
    try:
        go = GasOil(swl=swl, sgcr=sgcr, sorg=sorg, h=h, tag="")
    except AssertionError:
        return
    kroend = min(kroend, kromax)
    krgend = min(krgend, krgmax)
    go.add_corey_oil(kroend=kroend, kromax=kromax)
    go.add_corey_gas(krgend=krgend, krgmax=krgmax)
    check_table(go.table)
    assert go.selfcheck()
    check_endpoints(go, krgend, krgmax, kroend, kromax)

    # Redo with krgendanchor not defaulted
    go = GasOil(swl=swl, sgcr=sgcr, sorg=sorg, h=h, krgendanchor="", tag="")
    go.add_corey_oil(kroend=kroend, kromax=kromax)
    go.add_corey_gas(krgend=krgend, krgmax=krgmax)
    check_table(go.table)
    assert go.selfcheck()
    check_endpoints(go, krgend, krgmax, kroend, kromax)

    # Redo with LET:
    go = GasOil(swl=swl, sgcr=sgcr, sorg=sorg, h=h, tag="")
    go.add_LET_oil(kroend=kroend, kromax=kromax)
    go.add_LET_gas(krgend=krgend, krgmax=krgmax)
    check_table(go.table)
    assert go.selfcheck()
    check_endpoints(go, krgend, krgmax, kroend, kromax)


def check_endpoints(go, krgend, krgmax, kroend, kromax):
    swtol = 1 / SWINTEGERS

    # Check endpoints for oil curve:
    # krog at sgcr should be kroend
    if go.sgcr > go.swl + swtol:
        assert float_df_checker(go.table, "son", 1.0, "krog", kroend)
    # krog at son=0 (1-sorg-swl or 1 - swl) should be zero:
    assert float_df_checker(go.table, "son", 0.0, "krog", 0)

    if go.sgcr > go.swl + swtol:
        assert float_df_checker(go.table, "sg", 0, "krog", kromax)
        assert float_df_checker(go.table, "sg", go.sgcr, "krog", kroend)
    else:
        if not np.isclose(go.table["krog"].max(), kroend):
            print(go.table.head())
        assert np.isclose(go.table["krog"].max(), kroend)

    assert float_df_checker(go.table, "sgn", 0.0, "krg", 0)
    assert float_df_checker(go.table, "sg", go.sgcr, "krg", 0)

    # If krgendanchor == "sorg" then krgmax is irrelevant.
    if go.sorg > swtol and go.sorg > go.h and go.krgendanchor == "sorg":
        assert float_df_checker(go.table, "sgn", 1.0, "krg", krgend)
        assert np.isclose(go.table["krg"].max(), krgmax)
    if go.krgendanchor != "sorg":
        assert np.isclose(go.table["krg"].max(), krgend)


def test_gasoil_kromax():
    go = GasOil(h=0.1, sgcr=0.1)
    go.add_corey_oil(2, 0.5)  # Default for kromax
    assert float_df_checker(go.table, "sg", 0.0, "krog", 1.0)
    assert float_df_checker(go.table, "sg", 0.1, "krog", 0.5)
    go.add_corey_oil(2, 0.5, 0.7)
    assert float_df_checker(go.table, "sg", 0.0, "krog", 0.7)
    assert float_df_checker(go.table, "sg", 0.1, "krog", 0.5)

    go = GasOil(h=0.1, sgcr=0.0)
    go.add_corey_oil(2, 0.5)
    assert float_df_checker(go.table, "sg", 0.0, "krog", 0.5)
    go.add_corey_oil(2, 0.5, 1)  # A warning will be given
    assert float_df_checker(go.table, "sg", 0.0, "krog", 0.5)


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

    # Test once more for LET curves:
    gasoil = GasOil(krgendanchor="sorg", sorg=0.2, h=0.1)
    assert gasoil.sorg
    gasoil.add_LET_gas(1, 1, 1)
    gasoil.add_LET_oil(1, 1, 1)

    # kg should be 1.0 at 1 - sorg due to krgendanchor == "sorg":
    assert (
        gasoil.table[np.isclose(gasoil.table["sg"], 1 - gasoil.sorg)]["krg"].values[0]
        == 1.0
    )
    assert gasoil.table[np.isclose(gasoil.table["sg"], 1.0)]["krg"].values[0] == 1.0

    gasoil = GasOil(krgendanchor="", sorg=0.2, h=0.1)
    assert gasoil.sorg
    gasoil.add_LET_gas(1, 1, 1)
    gasoil.add_LET_oil(1, 1, 1)
    assert gasoil.selfcheck()

    # kg should be < 1 at 1 - sorg due to krgendanchor being ""
    assert (
        gasoil.table[np.isclose(gasoil.table["sg"], 1 - gasoil.sorg)]["krg"].values[0]
        < 1.0
    )
    assert gasoil.table[np.isclose(gasoil.table["sg"], 1.0)]["krg"].values[0] == 1.0


def test_kromaxend():
    """Manual testing of kromax and kroend behaviour"""
    gasoil = GasOil(swirr=0.01, sgcr=0.01, h=0.01, swl=0.1, sorg=0.05)
    gasoil.add_LET_gas()
    gasoil.add_LET_oil(2, 2, 2)
    assert gasoil.table["krog"].max() == 1
    gasoil.add_LET_oil(2, 2, 2, 0.5, 0.9)
    assert gasoil.table["krog"].max() == 0.5
    # Second krog-value should be kroend, values in between will be linearly
    # interpolated in Eclipse
    assert gasoil.table.sort_values("krog")[-2:-1]["krog"].values[0] == 0.5

    gasoil.add_corey_oil(2)
    assert gasoil.table["krog"].max() == 1
    gasoil.add_corey_oil(2, 0.5, 0.9)
    assert gasoil.table["krog"].max() == 0.5
    assert gasoil.table.sort_values("krog")[-2:-1]["krog"].values[0] == 0.5


@settings(deadline=1000)
@given(st.floats(), st.floats())
def test_gasoil_corey1(ng, nog):
    go = GasOil()
    try:
        go.add_corey_oil(nog=nog)
        go.add_corey_gas(ng=ng)
    except AssertionError:
        # This happens for "invalid" input
        return

    assert "krog" in go.table
    assert "krg" in go.table
    assert isinstance(go.krgcomment, str)
    check_table(go.table)
    sgofstr = go.SGOF()
    assert len(sgofstr) > 100


@settings(deadline=1000)
@given(st.floats(), st.floats(), st.floats(), st.floats(), st.floats())
def test_gasoil_let1(l, e, t, krgend, krgmax):
    go = GasOil()
    try:
        go.add_LET_oil(l, e, t, krgend)
        go.add_LET_gas(l, e, t, krgend, krgmax)
    except AssertionError:
        # This happens for negative values f.ex.
        return
    assert "krog" in go.table
    assert "krg" in go.table
    assert isinstance(go.krgcomment, str)
    check_table(go.table)
    sgofstr = go.SGOF()
    assert len(sgofstr) > 100
