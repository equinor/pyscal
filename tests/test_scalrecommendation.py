"""Test module for SCAL recommendation objects"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from hypothesis import given, settings
import hypothesis.strategies as st

from pyscal import SCALrecommendation, PyscalFactory

# Example SCAL recommendation, low case
LOW_SAMPLE_LET = {
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
BASE_SAMPLE_LET = {
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
HIGH_SAMPLE_LET = {
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


@settings(max_examples=10, deadline=3000)
@given(
    st.floats(min_value=-1.1, max_value=1.1), st.floats(min_value=-1.1, max_value=1.1)
)
def test_interpolation_deprecated(param_wo, param_go):
    """Testing deprecated functionality. Remove
    this test function when SCALrecommendation class is updated"""
    rec = SCALrecommendation(
        LOW_SAMPLE_LET, BASE_SAMPLE_LET, HIGH_SAMPLE_LET, "foo", h=0.1
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


def test_make_scalrecommendation():
    """Test that we can make scal recommendation objects
    from three WaterOilGas objects"""
    low = PyscalFactory.create_water_oil_gas(LOW_SAMPLE_LET)
    base = PyscalFactory.create_water_oil_gas(BASE_SAMPLE_LET)
    high = PyscalFactory.create_water_oil_gas(HIGH_SAMPLE_LET)
    rec = SCALrecommendation(low, base, high)
    interpolant = rec.interpolate(-0.5)
    check_table_wo(interpolant.wateroil.table)
    check_table_go(interpolant.gasoil.table)


@settings(max_examples=10, deadline=1500)
@given(
    st.floats(min_value=-1.1, max_value=1.1), st.floats(min_value=-1.1, max_value=1.1)
)
def test_interpolation(param_wo, param_go):
    rec = PyscalFactory.create_scal_recommendation(
        {"low": LOW_SAMPLE_LET, "base": BASE_SAMPLE_LET, "high": HIGH_SAMPLE_LET},
        "foo",
        h=0.1,
    )
    rec.add_simple_J()  # Add default pc curve

    # Check that added pc curve is non-zero
    assert sum(rec.low.wateroil.table["pc"])
    assert sum(rec.base.wateroil.table["pc"])
    assert sum(rec.high.wateroil.table["pc"])

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

    assert sum(interpolant.wateroil.table["pc"])


def test_boundary_cases():
    """Test that interpolation is able to return the boundaries
    at +/- 1"""
    rec = PyscalFactory.create_scal_recommendation(
        {"low": LOW_SAMPLE_LET, "base": BASE_SAMPLE_LET, "high": HIGH_SAMPLE_LET},
        "foo",
        h=0.1,
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


def check_table_wo(dframe):
    """Check sanity of important columns"""
    assert not dframe.empty
    assert not dframe.isnull().values.any()
    assert len(dframe["sw"].unique()) == len(dframe)
    assert dframe["sw"].is_monotonic
    assert (dframe["sw"] >= 0.0).all()
    assert dframe["swn"].is_monotonic
    assert dframe["son"].is_monotonic_decreasing
    assert dframe["swnpc"].is_monotonic
    assert dframe["krow"].is_monotonic_decreasing
    assert dframe["krw"].is_monotonic


def check_table_go(dframe):
    """Check sanity of important columns"""
    assert not dframe.empty
    assert not dframe.isnull().values.any()
    assert len(dframe["sg"].unique()) == len(dframe)
    assert dframe["sg"].is_monotonic
    assert (dframe["sg"] >= 0.0).all()
    assert dframe["sgn"].is_monotonic
    assert dframe["son"].is_monotonic_decreasing
    assert dframe["krog"].is_monotonic_decreasing
    assert dframe["krg"].is_monotonic
    if "pc" in dframe:
        assert dframe["pc"].is_monotonic
