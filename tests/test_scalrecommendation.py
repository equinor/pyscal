"""Test module for SCAL recommendation objects"""

import hypothesis.strategies as st
import pandas as pd
import pytest
from hypothesis import given

from pyscal import GasWater, PyscalFactory, SCALrecommendation, WaterOil, WaterOilGas
from pyscal.factory import slicedict
from pyscal.utils.testing import check_table, sat_table_str_ok, slow_hypothesis

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

    with pytest.raises(ValueError, match="Wrong arguments to SCALrecommendation"):
        SCALrecommendation([], [], [])


def test_make_scalrecommendation_wo(caplog):
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

    # Meaningless extra argument is ignored, only warning printed:
    rec.interpolate(-1, parameter2=1)
    assert "parameter2 is meaningless for water-oil only" in caplog.text


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

    with pytest.raises(ValueError, match="Interpolation parameter for gas must be in"):
        rec.interpolate(2)


@slow_hypothesis
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
    assert sum(rec.low.wateroil.table["PC"])
    assert sum(rec.base.wateroil.table["PC"])
    assert sum(rec.high.wateroil.table["PC"])

    try:
        interpolant = rec.interpolate(param_wo, param_go, h=0.1)
    except ValueError:
        return

    check_table(interpolant.wateroil.table)
    check_table(interpolant.gasoil.table)

    assert len(interpolant.gasoil.SGOF()) > 100
    assert len(interpolant.gasoil.SGFN()) > 100
    assert len(interpolant.wateroil.SWFN()) > 100
    assert len(interpolant.SOF3()) > 100
    assert len(interpolant.wateroil.SWOF()) > 100
    assert interpolant.threephaseconsistency()

    assert sum(interpolant.wateroil.table["PC"])


def test_boundary_cases():
    """Test that interpolation is able to return the boundaries
    at +/- 1"""
    rec = PyscalFactory.create_scal_recommendation(
        {"low": LOW_SAMPLE_LET, "base": BASE_SAMPLE_LET, "high": HIGH_SAMPLE_LET},
        "foo",
        h=0.1,
    )
    assert rec.type == WaterOilGas

    wo_cols = ["SW", "KRW", "KROW"]  # no Pc in this test data
    go_cols = ["SG", "KRG", "KROG"]
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


def test_corner_errors():
    """Test more or less likely error scenarios"""

    # This could possibly also have been exception scenarios:
    assert (
        SCALrecommendation(GasWater(), GasWater(), GasWater()).interpolate(0).SWFN()
        == ""
    )
    assert (
        SCALrecommendation(WaterOilGas(), WaterOilGas(), WaterOilGas())
        .interpolate(0)
        .SWFN()
        == ""
    )

    with pytest.raises(ValueError, match="Wrong arguments to SCALrecommendation"):
        SCALrecommendation(WaterOil(), WaterOil(), WaterOil())

    with pytest.raises(ValueError, match="Wrong arguments to SCALrecommendation"):
        SCALrecommendation(WaterOilGas(), WaterOilGas(), GasWater())


def test_gaswater_scal(caplog):
    """Test list of gas-water objects in scal recommendation"""
    dframe = pd.DataFrame(
        columns=["SATNUM", "CASE", "NW", "NG", "TAG"],
        data=[
            [1, "pess", 2, 2, "sometag"],
            [1, "base", 3, 3, "sometag"],
            [1, "opt", 4, 4, "sometag"],
        ],
    )
    rec_list = PyscalFactory.create_scal_recommendation_list(
        PyscalFactory.load_relperm_df(dframe), h=0.1
    )
    assert rec_list.pyscaltype == SCALrecommendation
    assert rec_list[1].type == GasWater
    low_list = rec_list.interpolate(-1)
    str_fam2 = low_list.build_eclipse_data(family=2)
    assert "SCAL recommendation interpolation to -1" in str_fam2
    assert "SGFN" in str_fam2
    assert "SWFN" in str_fam2

    # Meaningless extra argument is ignored, only warning printed:
    rec_list[1].interpolate(-1, parameter2=1)
    assert "parameter2 is meaningless for gas-water" in caplog.text
    caplog.clear()

    rec_list.interpolate(-1, int_params_go=1)
    assert "parameter2 is meaningless for gas-water" in caplog.text

    with pytest.raises(ValueError, match="Interpolation parameter must be in"):
        rec_list[1].interpolate(2)


def test_fast():
    """Test the fast option"""
    low_fast = PyscalFactory.create_water_oil_gas(LOW_SAMPLE_LET, fast=True)
    base_fast = PyscalFactory.create_water_oil_gas(BASE_SAMPLE_LET, fast=True)
    high_fast = PyscalFactory.create_water_oil_gas(HIGH_SAMPLE_LET, fast=True)

    rec = SCALrecommendation(low_fast, base_fast, high_fast)
    interp = rec.interpolate(-0.5)
    assert rec.fast
    assert interp.fast

    # test that one or more inputs not being set to fast does not trigger fast mode
    low = PyscalFactory.create_water_oil_gas(LOW_SAMPLE_LET)
    base = PyscalFactory.create_water_oil_gas(BASE_SAMPLE_LET)
    high = PyscalFactory.create_water_oil_gas(HIGH_SAMPLE_LET)

    rec = SCALrecommendation(low_fast, base_fast, high)
    interp = rec.interpolate(-0.5)
    assert not rec.fast
    assert not interp.fast

    rec = SCALrecommendation(low, base_fast, high)
    interp = rec.interpolate(-0.5)
    assert not rec.fast
    assert not interp.fast

    rec = SCALrecommendation(low, base, high)
    interp = rec.interpolate(-0.5)
    assert not rec.fast
    assert not interp.fast
