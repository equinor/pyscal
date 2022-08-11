"""Test module for capillary pressure in WaterOil"""

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

from pyscal import WaterOil
from pyscal.constants import MAX_EXPONENT
from pyscal.utils.testing import check_table, float_df_checker, sat_table_str_ok


def test_simple_j():
    """Simple test of the simple J function correlation"""
    wateroil = WaterOil(swl=0.01)
    wateroil.add_simple_J()  # swl set to zero will give infinite pc
    check_table(wateroil.table)
    assert wateroil.pccomment

    wateroil = WaterOil(swl=0)
    with pytest.raises(ValueError, match="swl must be larger than zero"):
        wateroil.add_simple_J()

    # Zero gravity:
    wateroil = WaterOil(swl=0.01)
    wateroil.add_simple_J(g=0)
    assert wateroil.table["PC"].unique() == 0.0

    # This should give back Sw:
    # This ensures that density and gravity scaling is correct
    wateroil.add_simple_J(a=1, b=1, poro_ref=1, perm_ref=1, drho=1000, g=100)
    assert (wateroil.table["PC"] - wateroil.table["SW"]).sum() < 0.00001
    # (check_table() will fail on this, when b > 0)

    # Some values seen in real life:
    wateroil.add_simple_J(a=100, b=-1.5, poro_ref=0.12, perm_ref=100, drho=200)
    check_table(wateroil.table)
    assert "Simplified" in wateroil.pccomment
    assert "a=100" in wateroil.pccomment
    assert "b=-1.5" in wateroil.pccomment
    wateroil.add_corey_oil()
    wateroil.add_corey_water()
    swof = wateroil.SWOF()
    assert isinstance(swof, str)
    assert swof
    sat_table_str_ok(swof)
    sat_table_str_ok(wateroil.SWFN())


def test_simple_j_petro():
    """Simple test of the simple J petrophysical function correlation"""
    wateroil = WaterOil(swl=0)
    with pytest.raises(ValueError, match="swl must be larger than zero"):
        wateroil.add_simple_J_petro(a=1, b=-2)

    wateroil = WaterOil(swl=0.01)
    wateroil.add_simple_J_petro(a=1, b=-2)
    check_table(wateroil.table)
    assert wateroil.pccomment
    assert "etrophysic" in wateroil.pccomment

    # Zero gravity:
    wateroil.add_simple_J_petro(a=1, b=-2, g=0)
    assert wateroil.table["PC"].unique() == 0.0

    # Numerical test from sample numbers calculated independently in different tool:
    wateroil = WaterOil(swl=0.05, h=0.025)
    wateroil.add_simple_J_petro(
        a=1.45, b=-0.285, drho=143, g=9.81, perm_ref=15, poro_ref=0.27
    )
    float_df_checker(wateroil.table, "SW", 0.1, "PC", 22.36746)
    assert "Simplified" in wateroil.pccomment
    assert "etrophysic" in wateroil.pccomment
    wateroil.add_corey_oil()
    wateroil.add_corey_water()
    swof = wateroil.SWOF()
    assert isinstance(swof, str)
    assert swof
    sat_table_str_ok(swof)
    sat_table_str_ok(wateroil.SWFN())

    with pytest.raises(ValueError, match="positive b"):
        wateroil = WaterOil(swl=0.01)
        wateroil.add_simple_J_petro(a=1, b=2)


@given(
    st.floats(min_value=0.001, max_value=1000000),
    st.floats(min_value=-0.9 * MAX_EXPONENT, max_value=-0.001),
    st.floats(min_value=0.01, max_value=0.5),
    st.floats(min_value=0.001, max_value=10),
    st.floats(min_value=0.01, max_value=1000000),
    st.floats(min_value=0.001, max_value=10000000),
)
def test_simple_j_random(a, b, poro_ref, perm_ref, drho, g):
    """Test different J-function parameters.

    Parameter ranges tested through hypothesis are limited, as not
    every number is realistic. Way outside the tested intervals, you
    can get AssertionErrors or the capillary pressure may not be
    monotonically decreasing within machine precision.
    """
    wateroil = WaterOil(swl=0.01)
    wateroil.add_simple_J(
        a=a, b=b, poro_ref=poro_ref, perm_ref=perm_ref, drho=drho, g=g
    )
    check_table(wateroil.table)


def test_normalized_j(caplog):
    """Test the normalized J-function correlation for capillary pressure"""
    wateroil = WaterOil(swirr=0.1, h=0.1)
    with pytest.raises(ValueError):
        wateroil.add_normalized_J(a=0.5, b=-0.2, poro=0.2, perm=10, sigma_costau=30)

    wateroil = WaterOil(swirr=0, swl=0.1, h=0.1)
    wateroil.add_normalized_J(a=0.5, b=-0.2, poro=0.2, perm=10, sigma_costau=30)
    check_table(wateroil.table)

    # Sample numerical tests taken from a prior implementation
    # NB: Prior implementation created Pc in atm, we create in bar
    bar_to_atm = 1.0 / 1.01325
    wateroil.add_normalized_J(a=0.22, b=-0.5, perm=100, poro=0.2, sigma_costau=30)
    float_df_checker(wateroil.table, "SW", 0.1, "PC", 2.039969 * bar_to_atm)
    float_df_checker(wateroil.table, "SW", 0.6, "PC", 0.056666 * bar_to_atm)
    float_df_checker(wateroil.table, "SW", 1.0, "PC", 0.02040 * bar_to_atm)

    wateroil = WaterOil(swirr=0.1, swl=0.11, h=0.1)
    wateroil.add_normalized_J(a=0.5, b=-0.001, poro=0.2, perm=10, sigma_costau=30)
    assert "b exponent is very small" in caplog.text

    wateroil = WaterOil(swirr=0.1, swl=0.11, h=0.1)
    wateroil.add_normalized_J(a=0.001, b=-2, poro=0.2, perm=10, sigma_costau=30)
    assert "a parameter is very small" in caplog.text

    wateroil = WaterOil(swirr=0.1, swl=0.11, h=0.1)
    wateroil.add_normalized_J(a=9, b=-2, poro=0.2, perm=10, sigma_costau=30)
    assert "a parameter is very high" in caplog.text


@given(
    st.floats(min_value=0, max_value=0.1),  # swirr
    st.floats(min_value=0.01, max_value=0.1),  # swl - swirr
    st.floats(min_value=0.01, max_value=5),  # a
    st.floats(min_value=-0.9 * MAX_EXPONENT, max_value=-0.01),  # b
    st.floats(min_value=-1, max_value=1.5),  # poro
    st.floats(min_value=0.0001, max_value=1000000000),  # perm
    st.floats(min_value=0, max_value=100000),  # sigma_costau
)
def test_norm_j_pc_random(swirr, swl, a_pc, b_pc, poro, perm, sigma_costau):
    """Test many possibilities of Pc-parameters.

    Outside of the tested range, there are many combination of parameters
    that can give infinite capillary pressure"""

    swl = swirr + swl  # No point in getting too many AssertionErrors
    wateroil = WaterOil(swirr=swirr, swl=swl, h=0.01)
    try:
        wateroil.add_normalized_J(
            a=a_pc, b=b_pc, perm=perm, poro=poro, sigma_costau=sigma_costau
        )
    except (AssertionError, ValueError):  # when poro is < 0 f.ex.
        return
    check_table(wateroil.table)
    wateroil.add_corey_water()
    wateroil.add_corey_oil()
    sat_table_str_ok(wateroil.SWOF())


def test_let_pc_pd():
    """Test LET formulation for primary drainage capillary pressure"""
    wateroil = WaterOil(swirr=0.1)
    wateroil.add_LET_pc_pd(Lp=1, Ep=1, Tp=1, Lt=1, Et=1, Tt=1, Pcmax=10, Pct=5)
    assert np.isclose(wateroil.table["PC"].max(), 10)
    assert np.isclose(wateroil.table["PC"].min(), 0)
    # (everything is linear)

    wateroil.add_LET_pc_pd(Lp=10, Ep=10, Tp=10, Lt=10, Et=10, Tt=10, Pcmax=10, Pct=5)
    assert np.isclose(wateroil.table["PC"].max(), 10)
    assert np.isclose(wateroil.table["PC"].min(), 0)
    # On a plot, you can see a kink at Pc=5.
    # wateroil.plotpc()

    wateroil = WaterOil(swirr=0.1, sorw=0.4)
    wateroil.add_LET_pc_pd(Lp=10, Ep=10, Tp=10, Lt=10, Et=10, Tt=10, Pcmax=5, Pct=2)
    assert np.isclose(wateroil.table["PC"].max(), 5)
    assert np.isclose(wateroil.table["PC"].min(), 0)
    # On plot: hard-to-see kink at Pc=2. .
    # wateroil.plotpc()
    wateroil.add_corey_water()
    wateroil.add_corey_oil()
    sat_table_str_ok(wateroil.SWOF())


def test_let_pc_imb():
    """Test the LET formulation for imbibition capillary pressures"""
    wateroil = WaterOil(swirr=0.1)
    wateroil.add_LET_pc_imb(
        Ls=1, Es=1, Ts=1, Lf=1, Ef=1, Tf=1, Pcmax=10, Pcmin=-10, Pct=3
    )
    assert np.isclose(wateroil.table["PC"].max(), 10)
    assert np.isclose(wateroil.table["PC"].min(), -10)

    wateroil = WaterOil(swirr=0.1)
    wateroil.add_LET_pc_imb(Ls=5, Es=5, Ts=5, Lf=5, Ef=5, Tf=5, Pcmax=5, Pcmin=1, Pct=4)
    assert np.isclose(wateroil.table["PC"].max(), 5)
    assert np.isclose(wateroil.table["PC"].min(), 1)

    wateroil = WaterOil(swirr=0.1, sorw=0.3)
    wateroil.add_LET_pc_imb(Ls=5, Es=5, Ts=5, Lf=5, Ef=5, Tf=5, Pcmax=5, Pcmin=1, Pct=4)
    assert np.isclose(wateroil.table["PC"].max(), 5)
    assert np.isclose(wateroil.table["PC"].min(), 1)
    wateroil.add_corey_water()
    wateroil.add_corey_oil()
    sat_table_str_ok(wateroil.SWOF())


def test_skjaeveland_pc():
    """Simple test of Skjæveland capillary pressure correlation"""
    wateroil = WaterOil(h=0.3, swl=0.2, swirr=0.1)
    wateroil.add_corey_water()
    wateroil.add_skjaeveland_pc(swr=0.1, cw=0.1, co=-0.1, aw=0.1, ao=0.1)
    check_table(wateroil.table)
    swfn = wateroil.SWFN()

    # Defaulting swr should give the same since swr==swirr above:
    wateroil.add_skjaeveland_pc(swr=None, cw=0.1, co=-0.1, aw=0.1, ao=0.1)
    assert wateroil.SWFN() == swfn

    # Add with wrong numbers
    wateroil = WaterOil(h=0.3, swl=0.2, sorw=0.3)
    with pytest.raises(ValueError, match="cw must be larger"):
        wateroil.add_skjaeveland_pc(swr=0.1, cw=-0.1, co=-0.1, aw=0.1, ao=0.1)

    with pytest.raises(ValueError, match="co must be less"):
        wateroil.add_skjaeveland_pc(swr=0.1, cw=0.1, co=0.1, aw=0.1, ao=0.1)

    with pytest.raises(ValueError, match="aw must be larger"):
        wateroil.add_skjaeveland_pc(swr=0.1, cw=0.1, co=-0.1, aw=-0.1, ao=0.1)

    with pytest.raises(ValueError, match="ao must be larger"):
        wateroil.add_skjaeveland_pc(swr=0.1, sor=0.3, cw=0.1, co=-0.1, aw=0.1, ao=-0.1)

    with pytest.raises(ValueError, match=r"swr \(swirr\) must be less than 1 - sor"):
        wateroil.add_skjaeveland_pc(swr=0.8, sor=0.3, cw=0.1, co=-0.1, aw=0.1, ao=0.1)

    with pytest.raises(ValueError, match="swr must be contained in"):
        wateroil.add_skjaeveland_pc(swr=-1.2, sor=0.3, cw=0.1, co=-0.1, aw=0.1, ao=0.1)

    with pytest.raises(ValueError, match="swr must be contained in"):
        wateroil.add_skjaeveland_pc(swr=1.2, sor=0.3, cw=0.1, co=-0.1, aw=0.1, ao=0.1)

    with pytest.raises(ValueError, match="sor must be contained in"):
        wateroil.add_skjaeveland_pc(swr=0.1, sor=-0.2, cw=0.1, co=-0.1, aw=0.1, ao=0.1)

    with pytest.raises(ValueError, match="sor must be contained in"):
        wateroil.add_skjaeveland_pc(swr=0.1, sor=1.2, cw=0.1, co=-0.1, aw=0.1, ao=0.1)
