"""Test module for SLGOF export from GasOil"""

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

from pyscal import GasOil, WaterOilGas
from pyscal.constants import EPSILON, SWINTEGERS
from pyscal.utils.testing import sat_table_str_ok


def check_table(dframe):
    """Check sanity of important columns"""
    assert not dframe.empty
    assert not dframe.isnull().values.any()
    assert dframe["SL"].is_monotonic_increasing
    assert (dframe["SL"] >= 0.0).all()
    assert (dframe["SL"] <= 1.0).all()
    # Increasing, but not monotonically for slgof
    assert (dframe["KROG"].diff().dropna() > -EPSILON).all()
    assert dframe["KRG"].is_monotonic_decreasing
    if "PC" in dframe:
        assert dframe["PC"].is_monotonic_decreasing


@given(
    st.floats(min_value=0.0, max_value=0.3),
    st.floats(min_value=0.0, max_value=0.3),
    st.floats(min_value=0.0, max_value=0.2),
)
def test_slgof(swl, sorg, sgcr):
    """Test dumping SLGOF records"""
    wog = WaterOilGas(swl=swl, sorg=sorg, sgcr=sgcr, h=0.05)
    wog.wateroil.add_corey_water()
    wog.wateroil.add_corey_oil()
    wog.gasoil.add_corey_gas(krgmax=1)
    wog.gasoil.add_corey_oil()

    assert wog.selfcheck()

    slgof = wog.gasoil.slgof_df()
    assert "SL" in slgof
    assert "KRG" in slgof
    assert "KROG" in slgof
    assert not slgof.empty

    check_table(slgof)
    sat_table_str_ok(wog.SLGOF())

    # Requirements from E100 manual:
    assert np.isclose(slgof["SL"].values[0], wog.gasoil.swl + wog.gasoil.sorg)
    assert np.isclose(slgof["KRG"].values[-1], 0)
    assert np.isclose(slgof["KROG"].values[0], 0)

    # If we ruin the object, SLGOF() will return an empty string:
    wog.gasoil.table.drop("KRG", axis="columns", inplace=True)
    assert wog.gasoil.SLGOF() == ""
    assert wog.SLGOF() == ""


@pytest.mark.parametrize(
    "swl, sorg, sgcr",
    [
        (0.0, 0.0, 0.1),
        (1e-18, 0.0, 0.1),
        (1e-17, 0.0, 0.1),
        (1e-16, 0.0, 0.1),
        (1e-15, 0.0, 0.1),
        (1e-14, 0.0, 0.1),
        (1e-13, 0.0, 0.1),
        (1e-12, 0.0, 0.1),
        (1e-11, 0.0, 0.1),
        (1e-10, 0.0, 0.1),
        #
        (0.1, 0.0, 0.1),
        (0.1, 1e-18, 0.1),
        (0.1, 1e-17, 0.1),
        (0.1, 1e-16, 0.1),
        (0.1, 1e-15, 0.1),
        (0.1, 1e-14, 0.1),
        (0.1, 1e-13, 0.1),
        (0.1, 1e-12, 0.1),
        (0.1, 1e-11, 0.1),
        (0.1, 1e-10, 0.1),
        #
        (0.1, 0.0, 0.0),
        (0.1, 0.0, 1e-18),
        (0.1, 0.0, 1e-17),
        (0.1, 0.0, 1e-16),
        (0.1, 0.0, 1e-15),
        (0.1, 0.0, 1e-14),
        (0.1, 0.0, 1e-13),
        (0.1, 0.0, 1e-12),
        (0.1, 0.0, 1e-11),
        (0.1, 0.0, 1e-10),
        #
        (0.1 - 0.0, 0.0, 0.1),
        (0.1 - 1e-18, 0.0, 0.1),
        (0.1 - 1e-17, 0.0, 0.1),
        (0.1 - 1e-16, 0.0, 0.1),
        (0.1 - 1e-15, 0.0, 0.1),
        (0.1 - 1e-14, 0.0, 0.1),
        (0.1 - 1e-13, 0.0, 0.1),
        (0.1 - 1e-12, 0.0, 0.1),
        (0.1 - 1e-11, 0.0, 0.1),
        (0.1 - 1e-10, 0.0, 0.1),
    ],
)
def test_numerical_problems(swl, sorg, sgcr):
    """Test fine-tuned numerics for slgof"""

    # Because we cut away some saturation points due to SWINTEGERS, we easily
    # end in a situation where the wrong saturation point of to "equal" ones
    # is removed (because in SLGOF, sg is flipped to sl)

    gasoil = GasOil(swl=swl, sorg=sorg, sgcr=sgcr, h=0.001)
    gasoil.add_corey_gas()
    gasoil.add_corey_oil()
    assert gasoil.selfcheck()
    slgof = gasoil.slgof_df()
    assert np.isclose(slgof["SL"].values[0], gasoil.swl + gasoil.sorg)
    assert np.isclose(slgof["SL"].values[-1], 1.0)
    check_table(slgof)


@given(
    st.floats(min_value=0.0, max_value=0.3),
    st.floats(min_value=0.0, max_value=0.3),
    st.floats(min_value=0.0, max_value=0.3),
    st.floats(min_value=1.0 / float(1000 * SWINTEGERS), max_value=0.5),
)
def test_slgof_hypo(swl, sorg, sgcr, h):
    """Shotgun-testing of slgof"""
    gasoil = GasOil(swl=swl, sorg=sorg, sgcr=sgcr, h=h)
    gasoil.add_corey_gas()
    gasoil.add_corey_oil()
    assert gasoil.selfcheck()
    slgof = gasoil.slgof_df()
    check_table(slgof)
    # Eclipse 100 requirement from manual:
    assert np.isclose(slgof["SL"].values[0], gasoil.swl + gasoil.sorg)
    # Eclipse 100 requirement from manual:
    assert np.isclose(slgof["SL"].values[-1], 1.0)
    slgof_str = gasoil.SLGOF()
    assert isinstance(slgof_str, str)
    assert slgof_str
    sat_table_str_ok(slgof_str)


@given(
    st.floats(min_value=0.0, max_value=0.3),
    st.floats(min_value=0.0, max_value=0.3),
    st.floats(min_value=1.0 / float(1000 * SWINTEGERS), max_value=0.5),
)
def test_slgof_sl_mismatch(swl, sorg, h):
    """Test numerical robustness on slgof table creation."""
    gasoil = GasOil(h=h, swl=swl, sorg=sorg)
    gasoil.add_corey_gas()
    gasoil.add_corey_oil()

    # It is a strict requirement that the first sl value should be swl + sorg,
    # but GasOil might have truncated the values.
    assert np.isclose(gasoil.slgof_df()["SL"].values[0] - (gasoil.swl + gasoil.sorg), 0)
