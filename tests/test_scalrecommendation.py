# -*- coding: utf-8 -*-
"""Test module for relperm"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from hypothesis import given, settings
import hypothesis.strategies as st

import pytest

from pyscal import SCALrecommendation

# Example SCAL recommendation, low case
low_sample_let = {
    "swirr": 0.1,
    "sorw": 0.02,
    "krwend": 0.7,
    "krwmax": 0.8,
    "swl": 0.16,
    "swcr": 0.25,
    "Lw": 2.323,
    "Ew": 2.0,
    "Tw": 1.329,
    "Lo": 4.944,
    "Eo": 5.0,
    "To": 0.68,
    "Lg": 4,
    "Eg": 1,
    "Tg": 1,
    "Log": 4,
    "Eog": 1,
    "Tog": 1,
    "sorg": 0.2,
    "sgcr": 0.15,
    "krgend": 0.9,
    "krgmax": 1,
    "kroend": 1,
}
# Example SCAL recommendation, base case
base_sample_let = {
    "swirr": 0.1,
    "sorw": 0.091,
    "krwend": 0.8,
    "swl": 0.16,
    "swcr": 0.20,
    "Lw": 3.369,
    "Ew": 4.053,
    "Tw": 1.047,
    "Lo": 3.726,
    "Eo": 3.165,
    "To": 1.117,
    "Lg": 2,
    "Eg": 2,
    "Tg": 2,
    "Log": 2,
    "Eog": 2,
    "Tog": 2,
    "sorg": 0.1,
    "sgcr": 0.10,
    "krgend": 0.97,
    "kroend": 1,
}
# Example SCAL recommendation, high case
high_sample_let = {
    "swirr": 0.1,
    "sorw": 0.137,
    "krwend": 0.6,
    "swl": 0.16,
    "swcr": 0.16,
    "Lw": 4.436,
    "Ew": 8.0,
    "Tw": 0.766,
    "Lo": 2.537,
    "Eo": 2.0,
    "To": 1.549,
    "Lg": 1,
    "Eg": 2,
    "Tg": 2,
    "Log": 1,
    "Eog": 2,
    "Tog": 2,
    "sorg": 0.05,
    "sgcr": 0.0,
    "krgend": 1,
    "kroend": 1,
}

low_sample_corey = {
    "swirr": 0.25,
    "sorw": 0.25,
    "swl": 0.27,
    "nw": 1,
    "ng": 3,
    "nog": 5,
    "now": 5,
    "sorg": 0.2,  # Todo: We do not capture if sorg + sgcr > 1 - swl
    "sgcr": 0.35,
    "kroend": 0.2,
    "krwend": 0.4,
}
base_sample_corey = {
    "swirr": 0.25,
    "sorw": 0.25,
    "swl": 0.27,
    "nw": 3,
    "ng": 2,
    "nog": 3,
    "now": 3.5,
    "sorg": 0.1,
    "sgcr": 0.15,
    "kroend": 0.3,
    "krwend": 0.24,
}
high_sample_corey = {
    "swirr": 0.25,
    "sorw": 0.25,
    "swl": 0.27,
    "nw": 4,
    "ng": 1.4,
    "nog": 2,
    "now": 2,
    "sorg": 0.05,
    "sgcr": 0.05,
    "kroend": 0.4,
    "krwend": 0.13,
}



def test_example_let_cases():
    rec = SCALrecommendation(
        low_sample_let, base_sample_let, high_sample_let, "foo", h=0.1
    )

    check_table_wo(rec.low.wateroil.table)
    check_table_wo(rec.base.wateroil.table)
    check_table_wo(rec.high.wateroil.table)
    check_table_go(rec.low.gasoil.table)
    check_table_go(rec.base.gasoil.table)
    check_table_go(rec.high.gasoil.table)
    assert rec.low.threephaseconsistency() == ""
    assert rec.base.threephaseconsistency() == ""
    assert rec.high.threephaseconsistency() == ""

def test_example_corey_cases():
    rec = SCALrecommendation(
        low_sample_corey, base_sample_corey, high_sample_corey, "foo", h=0.1
    )
    print()
    print(rec.low.gasoil.table)
    print(rec.base.gasoil.table)
    check_table_wo(rec.low.wateroil.table)
    check_table_wo(rec.base.wateroil.table)
    check_table_wo(rec.high.wateroil.table)
    check_table_go(rec.low.gasoil.table)
    check_table_go(rec.base.gasoil.table)
    check_table_go(rec.high.gasoil.table)
    assert rec.low.threephaseconsistency() == ""
    assert rec.base.threephaseconsistency() == ""
    assert rec.high.threephaseconsistency() == ""




@settings(max_examples=10, deadline=1000)
@given(
    st.floats(min_value=-1.1, max_value=1.1), st.floats(min_value=-1.1, max_value=1.1)
)
def test_interpolation_let(param_wo, param_go):
    rec = SCALrecommendation(
        low_sample_let, base_sample_let, high_sample_let, "foo", h=0.1
    )
    rec.add_simple_J()  # Add default pc curve

    try:
        interpolant = rec.interpolate(param_wo, param_go, h=0.1)
    except AssertionError:
        return

    check_table_wo(interpolant.wateroil.table)
    check_table_go(interpolant.gasoil.table)

    assert len(interpolant.gasoil.SGOF()) > 100
    assert len(interpolant.gasoil.SGFN()) > 100
    assert len(interpolant.wateroil.SWFN()) > 100
    assert len(interpolant.SOF3()) > 100
    assert len(interpolant.wateroil.SWOF()) > 100
    assert interpolant.threephaseconsistency() == ""

@settings(max_examples=10, deadline=1000)
@given(
    st.floats(min_value=-1.1, max_value=1.1), st.floats(min_value=-1.1, max_value=1.1)
)
def test_interpolation_corey(param_wo, param_go):
    rec = SCALrecommendation(
        low_sample_corey, base_sample_corey, high_sample_corey, "foo", h=0.1
    )
    rec.add_simple_J()  # Add default pc curve

    try:
        interpolant = rec.interpolate(param_wo, param_go, h=0.1)
    except AssertionError:
        return

    check_table_wo(interpolant.wateroil.table)
    check_table_go(interpolant.gasoil.table)

    assert len(interpolant.gasoil.SGOF()) > 100
    assert len(interpolant.gasoil.SGFN()) > 100
    assert len(interpolant.wateroil.SWFN()) > 100
    assert len(interpolant.SOF3()) > 100
    assert len(interpolant.wateroil.SWOF()) > 100
    assert interpolant.threephaseconsistency() == ""


def test_boundary_cases():
    rec = SCALrecommendation(
        low_sample_let, base_sample_let, high_sample_let, "foo", h=0.1
    )
    # Object reference equivalence is a little bit strict,
    # because it would be perfectly fine if interpolate()
    # retured copied objects. But we don't have an equivalence operator
    # implemented.
    assert rec.interpolate(0).wateroil == rec.base.wateroil
    assert rec.interpolate(-1).wateroil == rec.low.wateroil
    assert rec.interpolate(1).wateroil == rec.high.wateroil
    assert rec.interpolate(0).gasoil == rec.base.gasoil
    assert rec.interpolate(-1).gasoil == rec.low.gasoil
    assert rec.interpolate(1).gasoil == rec.high.gasoil

    assert rec.interpolate(0, 1).wateroil == rec.base.wateroil
    assert rec.interpolate(-1, 1).wateroil == rec.low.wateroil
    assert rec.interpolate(1, 1).wateroil == rec.high.wateroil

    assert rec.interpolate(0, 1).gasoil == rec.high.gasoil
    assert rec.interpolate(-1, 1).gasoil == rec.high.gasoil
    assert rec.interpolate(1, 1).gasoil == rec.high.gasoil

    assert rec.interpolate(0, 0).gasoil == rec.base.gasoil
    assert rec.interpolate(-1, 0).gasoil == rec.base.gasoil
    assert rec.interpolate(1, 0).gasoil == rec.base.gasoil

    assert rec.interpolate(0, -1).gasoil == rec.low.gasoil
    assert rec.interpolate(-1, -1).gasoil == rec.low.gasoil
    assert rec.interpolate(1, -1).gasoil == rec.low.gasoil


def test_mangling_input():
    with pytest.raises(ValueError):
        rec = SCALrecommendation(dict(), dict(), dict(), "foo", h=0.1)

def check_table_wo(df):
    """Check sanity of important columns"""
    assert not df.empty
    assert not df.isnull().values.any()
    assert len(df["sw"].unique()) == len(df)
    assert df["sw"].is_monotonic
    assert (df["sw"] >= 0.0).all()
    assert df["swn"].is_monotonic
    assert df["son"].is_monotonic_decreasing
    assert df["swnpc"].is_monotonic
    assert df["krow"].is_monotonic_decreasing
    assert df["krw"].is_monotonic


def check_table_go(df):
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
