"""Test module for SCAL recommendation objects"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd

from hypothesis import given, settings
import hypothesis.strategies as st

from pyscal import SCALrecommendation, PyscalFactory, GasWater, WaterOilGas
from pyscal.factory import slicedict

from common import sat_table_str_ok, check_table


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
    "tag": "SATNUM X",
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
    "tag": "SATNUM X",
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
    "tag": "SATNUM X",
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
    assert rec.type == WaterOilGas
    try:
        interpolant = rec.interpolate(param_wo, param_go, h=0.1)
    except AssertionError:
        return

    check_table(interpolant.wateroil.table)
    check_table(interpolant.gasoil.table)

    assert len(interpolant.gasoil.SGOF()) > 100
    assert len(interpolant.gasoil.SGFN()) > 100
    assert len(interpolant.wateroil.SWFN()) > 100
    assert len(interpolant.SOF3()) > 100
    assert len(interpolant.wateroil.SWOF()) > 100
    if not interpolant.threephaseconsistency():
        print(interpolant.wateroil.SWOF())
        print(interpolant.gasoil.SGOF())
    assert interpolant.threephaseconsistency()


def test_deprecated_kroend():
    """Testing that the deprecated scalrecommendation can take
    both kroend and krogend/krowend"""

    low_krowend = dict(LOW_SAMPLE_LET)
    low_krowend["krowend"] = low_krowend["kroend"]
    del low_krowend["kroend"]

    base_krowend = dict(BASE_SAMPLE_LET)
    base_krowend["krowend"] = base_krowend["kroend"]
    del base_krowend["kroend"]

    high_krowend = dict(HIGH_SAMPLE_LET)
    high_krowend["krowend"] = high_krowend["kroend"]
    del high_krowend["kroend"]

    rec = SCALrecommendation(low_krowend, base_krowend, high_krowend, "foo", h=0.1)

    rec.add_simple_J()  # Add default pc curve
    interpolant = rec.interpolate(0.1, 0, h=0.1)
    check_table(interpolant.wateroil.table)
    print(interpolant.SWOF())


def test_make_scalrecommendation():
    """Test that we can make scal recommendation objects
    from three WaterOilGas objects"""

    low = PyscalFactory.create_water_oil_gas(LOW_SAMPLE_LET)
    base = PyscalFactory.create_water_oil_gas(BASE_SAMPLE_LET)
    high = PyscalFactory.create_water_oil_gas(HIGH_SAMPLE_LET)
    rec = SCALrecommendation(low, base, high)
    interpolant = rec.interpolate(-0.5, h=0.2)
    check_table(interpolant.wateroil.table)
    check_table(interpolant.gasoil.table)

    # Check preservation of tags.
    swof = interpolant.SWOF()
    assert "SCAL recommendation interpolation to -0.5" in swof
    assert "SCAL recommendation interpolation to -0.5" in interpolant.SGOF()
    assert "SCAL recommendation interpolation to -0.5" in interpolant.SWFN()
    assert "SCAL recommendation interpolation to -0.5" in interpolant.SOF3()
    assert "SCAL recommendation interpolation to -0.5" in interpolant.SGFN()

    assert swof.count("SATNUM X") == 1
    assert interpolant.SOF3().count("SATNUM X") == 1

    # Check different comment when different interpolation parameter:
    interpolant = rec.interpolate(-0.777, 0.888, h=0.2)
    assert "SCAL recommendation interpolation to -0.777" in interpolant.SWOF()
    assert "SCAL recommendation interpolation to 0.888" in interpolant.SGOF()


def test_make_scalrecommendation_wo():
    """Test that we can make scal recommendation objects
    from three WaterOilGas objects, but only with WaterOil
    objects"""

    wo_param_names = [
        "swirr",
        "sorw",
        "krwend",
        "krwmax",
        "swl",
        "swcr",
        "Lw",
        "Ew",
        "Tw",
        "Lo",
        "Eo",
        "To",
        "kroend",
    ]

    low_let_wo = slicedict(LOW_SAMPLE_LET, wo_param_names)
    low = PyscalFactory.create_water_oil_gas(low_let_wo)
    base_let_wo = slicedict(BASE_SAMPLE_LET, wo_param_names)
    base = PyscalFactory.create_water_oil_gas(base_let_wo)
    high_let_wo = slicedict(HIGH_SAMPLE_LET, wo_param_names)
    assert "Lg" not in high_let_wo
    high = PyscalFactory.create_water_oil_gas(high_let_wo)
    rec = SCALrecommendation(low, base, high)
    interpolant = rec.interpolate(-0.5)
    check_table(interpolant.wateroil.table)
    assert interpolant.gasoil is None
    sat_table_str_ok(interpolant.SWOF())
    sat_table_str_ok(interpolant.SWFN())

    # This should return empty string
    assert not interpolant.SGOF()


def test_make_scalrecommendation_go():
    """Test that we can make scal recommendation objects
    from three WaterOilGas objects, but only with GasOil
    objects"""

    go_param_names = [
        "swirr",
        "sorg",
        "krgend",
        "krgmax",
        "swl",
        "sgcr",
        "Lg",
        "Eg",
        "Tg",
        "Log",
        "Eog",
        "Tog",
        "kroend",
    ]

    low_let_go = slicedict(LOW_SAMPLE_LET, go_param_names)
    low = PyscalFactory.create_water_oil_gas(low_let_go)
    base_let_go = slicedict(BASE_SAMPLE_LET, go_param_names)
    base = PyscalFactory.create_water_oil_gas(base_let_go)
    high_let_go = slicedict(HIGH_SAMPLE_LET, go_param_names)
    assert "Lw" not in high_let_go
    high = PyscalFactory.create_water_oil_gas(high_let_go)
    rec = SCALrecommendation(low, base, high)
    assert rec.type == WaterOilGas
    interpolant = rec.interpolate(-0.5)
    check_table(interpolant.gasoil.table)
    assert interpolant.wateroil is None
    sat_table_str_ok(interpolant.SGOF())
    sat_table_str_ok(interpolant.SGFN())

    # This should return empty string
    assert not interpolant.SWOF()


@settings(max_examples=10, deadline=1500)
@given(
    st.floats(min_value=-1.1, max_value=1.1), st.floats(min_value=-1.1, max_value=1.1)
)
def test_interpolation(param_wo, param_go):
    """Test interpolation with random interpolation parameters,
    looking for numerical corner cases"""

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

    check_table(interpolant.wateroil.table)
    check_table(interpolant.gasoil.table)

    assert len(interpolant.gasoil.SGOF()) > 100
    assert len(interpolant.gasoil.SGFN()) > 100
    assert len(interpolant.wateroil.SWFN()) > 100
    assert len(interpolant.SOF3()) > 100
    assert len(interpolant.wateroil.SWOF()) > 100
    assert interpolant.threephaseconsistency()

    assert sum(interpolant.wateroil.table["pc"])


def test_boundary_cases():
    """Test that interpolation is able to return the boundaries
    at +/- 1"""
    rec = PyscalFactory.create_scal_recommendation(
        {"low": LOW_SAMPLE_LET, "base": BASE_SAMPLE_LET, "high": HIGH_SAMPLE_LET},
        "foo",
        h=0.1,
    )
    assert rec.type == WaterOilGas

    wo_cols = ["sw", "krw", "krow"]  # no Pc in this test data
    go_cols = ["sg", "krg", "krog"]
    assert (
        rec.interpolate(0)
        .wateroil.table[wo_cols]
        .equals(rec.base.wateroil.table[wo_cols])
    )
    assert (
        rec.interpolate(-1)
        .wateroil.table[wo_cols]
        .equals(rec.low.wateroil.table[wo_cols])
    )
    assert (
        rec.interpolate(1)
        .wateroil.table[wo_cols]
        .equals(rec.high.wateroil.table[wo_cols])
    )

    assert (
        rec.interpolate(0).gasoil.table[go_cols].equals(rec.base.gasoil.table[go_cols])
    )
    assert (
        rec.interpolate(-1).gasoil.table[go_cols].equals(rec.low.gasoil.table[go_cols])
    )
    assert (
        rec.interpolate(1).gasoil.table[go_cols].equals(rec.high.gasoil.table[go_cols])
    )

    assert (
        rec.interpolate(0, 1)
        .wateroil.table[wo_cols]
        .equals(rec.base.wateroil.table[wo_cols])
    )
    assert (
        rec.interpolate(-1, 1)
        .wateroil.table[wo_cols]
        .equals(rec.low.wateroil.table[wo_cols])
    )
    assert (
        rec.interpolate(1, 1)
        .wateroil.table[wo_cols]
        .equals(rec.high.wateroil.table[wo_cols])
    )

    assert (
        rec.interpolate(0, 1)
        .gasoil.table[go_cols]
        .equals(rec.high.gasoil.table[go_cols])
    )
    assert (
        rec.interpolate(-1, 1)
        .gasoil.table[go_cols]
        .equals(rec.high.gasoil.table[go_cols])
    )
    assert (
        rec.interpolate(1, 1)
        .gasoil.table[go_cols]
        .equals(rec.high.gasoil.table[go_cols])
    )

    assert (
        rec.interpolate(0, 0)
        .gasoil.table[go_cols]
        .equals(rec.base.gasoil.table[go_cols])
    )
    assert (
        rec.interpolate(-1, 0)
        .gasoil.table[go_cols]
        .equals(rec.base.gasoil.table[go_cols])
    )
    assert (
        rec.interpolate(1, 0)
        .gasoil.table[go_cols]
        .equals(rec.base.gasoil.table[go_cols])
    )

    assert (
        rec.interpolate(0, -1)
        .gasoil.table[go_cols]
        .equals(rec.low.gasoil.table[go_cols])
    )
    assert (
        rec.interpolate(-1, -1)
        .gasoil.table[go_cols]
        .equals(rec.low.gasoil.table[go_cols])
    )
    assert (
        rec.interpolate(1, -1)
        .gasoil.table[go_cols]
        .equals(rec.low.gasoil.table[go_cols])
    )


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
    rec_list = PyscalFactory.create_scal_recommendation_list(
        PyscalFactory.load_relperm_df(dframe), h=0.1
    )
    assert rec_list.pyscaltype == SCALrecommendation
    assert rec_list[1].type == GasWater
    low_list = rec_list.interpolate(-1)
    str_fam2 = low_list.dump_family_2()
    assert "SCAL recommendation interpolation to -1" in str_fam2
    assert "SGFN" in str_fam2
    assert "SWFN" in str_fam2
