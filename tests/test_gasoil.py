# -*- coding: utf-8 -*-
"""Test module for relperm"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from hypothesis import given, settings
import hypothesis.strategies as st


from pyscal import GasOil
from pyscal.constants import SWINTEGERS


def check_table(df):
    """Check sanity of important columns"""
    assert not df.empty
    assert not df.isnull().values.any()
    assert len(df["sg"].unique()) == len(df)
    assert df["sg"].is_monotonic
    assert (df["sg"] >= 0.0).all()
    assert df["sgn"].is_monotonic
    assert df["son"].is_monotonic_decreasing
    assert df["krog"].is_monotonic_decreasing
    assert df["krg"].is_monotonic
    if "pc" in df:
        assert df["pc"].is_monotonic


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


def test_gasoil_slgof():
    """Test fine-tuned numerics for slgof"""

    # Parameter set found by hypothesis
    swl = 0.029950000000000105
    sorg = 0.0
    sgcr = 0.01994999999999992
    # Because we cut away some saturation points due to SWINTEGERS, we easily
    # end in a situation where the wrong saturation point of to "equal" ones
    # is removed (because in SLGOF, sg is flipped to sl)

    # Unrounded, this represents a numerical difficulty, when h is low enough,
    # but there is special code in slgof_df() to workaround this.
    gasoil = GasOil(swl=swl, sorg=sorg, sgcr=sgcr, h=0.001)
    gasoil.add_corey_gas()
    gasoil.add_corey_oil()
    assert gasoil.selfcheck()
    slgof = gasoil.slgof_df()
    assert np.isclose(slgof["sl"].values[0], gasoil.swl + gasoil.sorg)
    assert np.isclose(slgof["sl"].values[-1], 1.0)


@settings(deadline=1000)
@given(
    st.floats(min_value=0.0, max_value=0.3),
    st.floats(min_value=0.0, max_value=0.3),
    st.floats(min_value=0.0, max_value=0.3),
    st.floats(min_value=1.0 / float(SWINTEGERS), max_value=0.5),
)
def test_slgof_hypo(swl, sorg, sgcr, h):
    """Shotgun-testing of slgof"""
    gasoil = GasOil(swl=swl, sorg=sorg, sgcr=sgcr, h=h)
    gasoil.add_corey_gas()
    gasoil.add_corey_oil()
    assert gasoil.selfcheck()
    slgof = gasoil.slgof_df()
    # Eclipse 100 requirement from manual:
    assert np.isclose(slgof["sl"].values[0], gasoil.swl + gasoil.sorg)
    # Eclipse 100 requirement from manual:
    assert np.isclose(slgof["sl"].values[-1], 1.0)
    slgof_str = gasoil.SLGOF()
    assert isinstance(slgof_str, str)
    assert slgof_str


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
