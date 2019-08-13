# -*- coding: utf-8 -*-
"""Test module for relperm"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from hypothesis import given, settings
import hypothesis.strategies as st

import numpy as np

from pyscal import WaterOilGas


def check_table(df):
    """Check sanity of important columns"""
    assert not df.empty
    assert not df.isnull().values.any()
    assert len(df["sl"].unique()) == len(df)
    assert df["sl"].is_monotonic
    assert (df["sl"] >= 0.0).all()
    assert (df["sl"] <= 1.0).all()
    assert df["krog"].is_monotonic_increasing
    assert df["krg"].is_monotonic_decreasing
    if "pc" in df:
        assert df["pc"].is_monotonic_decreasing


@settings(deadline=1000)
@given(st.floats(min_value=0.0, max_value=0.3), st.floats(min_value=0.0, max_value=0.3),
        st.floats(min_value=0.0, max_value=0.2))
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
    assert np.isclose(slgof["sl"].values[0], swl + sorg)
    assert np.isclose(slgof["krg"].values[-1], 0)
    assert np.isclose(slgof["krog"].values[0], 0)
