"""Test module for SLGOF export from GasOil"""

from hypothesis import given, settings
import hypothesis.strategies as st

import numpy as np

import pytest

from pyscal import WaterOilGas, GasOil
from pyscal.constants import SWINTEGERS, EPSILON

from pyscal.utils.testing import sat_table_str_ok


def check_table(dframe):
    """Check sanity of important columns"""
    assert not dframe.empty
    assert not dframe.isnull().values.any()
    assert dframe["SL"].is_monotonic
    assert (dframe["SL"] >= 0.0).all()
    assert (dframe["SL"] <= 1.0).all()
    # Increasing, but not monotonically for slgof
    assert (dframe["KROG"].diff().dropna() > -EPSILON).all()
    assert dframe["KRG"].is_monotonic_decreasing
    if "PC" in dframe:
        assert dframe["PC"].is_monotonic_decreasing


@settings(deadline=1000)
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


@pytest.mark.parametrize(
    "swl, sorg, sgcr",
    [
        # Parameter combinations exposed by pytest.hypothesis:
        (0.029950000000000105, 0.0, 0.01994999999999992),
        (0.285053445121882, 0.24900257734119435, 0.1660017182274629),
    ],
)
def test_numerical_problems(swl, sorg, sgcr):
    """Test fine-tuned numerics for slgof, this function should
    trigger the code path in slgof_df() where slgof_sl_mismatch is small.

    Note: The code path taken may depend on hardware/OS etc.
    """

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
    assert np.isclose(slgof["SL"].values[0], gasoil.swl + gasoil.sorg)
    assert np.isclose(slgof["SL"].values[-1], 1.0)
    check_table(slgof)


@settings(deadline=2000)  # This is slow for small h
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
    check_table(slgof)
    # Eclipse 100 requirement from manual:
    assert np.isclose(slgof["SL"].values[0], gasoil.swl + gasoil.sorg)
    # Eclipse 100 requirement from manual:
    assert np.isclose(slgof["SL"].values[-1], 1.0)
    slgof_str = gasoil.SLGOF()
    assert isinstance(slgof_str, str)
    assert slgof_str
    sat_table_str_ok(slgof_str)
