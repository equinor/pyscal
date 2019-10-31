# -*- coding: utf-8 -*-
"""Test module for relperm"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from hypothesis import given, settings
import hypothesis.strategies as st

import numpy as np

from pyscal import WaterOilGas, GasOil
from pyscal.constants import SWINTEGERS, EPSILON


def check_table(df):
    """Check sanity of important columns"""
    assert not df.empty
    assert not df.isnull().values.any()
    assert df["sl"].is_monotonic
    assert (df["sl"] >= 0.0).all()
    assert (df["sl"] <= 1.0).all()
    # Increasing, but not monotonically for slgof
    assert (df["krog"].diff().dropna() > -EPSILON).all()
    assert df["krg"].is_monotonic_decreasing
    if "pc" in df:
        assert df["pc"].is_monotonic_decreasing


@settings(deadline=1000)
@given(
    st.floats(min_value=0.0, max_value=0.3),
    st.floats(min_value=0.0, max_value=0.3),
    st.floats(min_value=0.0, max_value=0.2),
)
def test_slgof(swl, sorg, sgcr):
    wog = WaterOilGas(swl=swl, sorg=sorg, sgcr=sgcr, h=0.05)
    wog.wateroil.add_corey_water()
    wog.wateroil.add_corey_oil()
    wog.gasoil.add_corey_gas(krgmax=1)
    wog.gasoil.add_corey_oil()

    assert wog.selfcheck()

    slgof = wog.gasoil.slgof_df()
    assert "sl" in slgof
    assert "krg" in slgof
    assert "krog" in slgof
    assert len(slgof)

    check_table(slgof)

    # Requirements from E100 manual:
    assert np.isclose(slgof["sl"].values[0], wog.gasoil.swl + wog.gasoil.sorg)
    assert np.isclose(slgof["krg"].values[-1], 0)
    assert np.isclose(slgof["krog"].values[0], 0)


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
    assert np.isclose(slgof["sl"].values[0], gasoil.swl + gasoil.sorg)
    # Eclipse 100 requirement from manual:
    assert np.isclose(slgof["sl"].values[-1], 1.0)
    slgof_str = gasoil.SLGOF()
    assert isinstance(slgof_str, str)
    assert slgof_str
