"""Test module for capillary pressure support code in pyscal"""
import math

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

from pyscal.constants import EPSILON
from pyscal.utils import capillarypressure

PASCAL = 1e-05  # One pascal in bar.

# pylint: disable=protected-access  # Private functions should be tested too


@pytest.mark.parametrize(
    "sw, a, b, poro_ref, perm_ref, drho, g, expected",
    [
        (0, 0, 0, 0, 1, 0, 0, 0),
        (1, 1, 1, 1, 1, 1, 1, PASCAL),
        # Linear in gravity:
        (1, 1, 1, 1, 1, 1, 10, 10 * PASCAL),
        # Linear in density difference
        (1, 1, 1, 1, 1, 10, 1, 10 * PASCAL),
        # Nonlinear in permeabilty
        (1, 1, 1, 1, 10, 1, 1, math.sqrt(1.0 / 10.0) * PASCAL),
        # Nonlinear in porosity:
        (1, 1, 1, 0.5, 1, 1, 1, math.sqrt(0.5) * PASCAL),
        # Linear in a:
        (1, 2, 1, 1, 1, 1, 1, 2 * PASCAL),
        # Nonlinear in b:
        (1, 1, 2, 1, 1, 1, 1, PASCAL),
        (0.5, 1, 2, 1, 1, 1, 1, 0.5**2 * PASCAL),
        # Vector support in sw:
        (np.array([0, 1]), 1, 1, 1, 1, 1, 1, np.array([0, PASCAL])),
    ],
)
def test_simple_J(sw, a, b, poro_ref, perm_ref, drho, g, expected):
    """Test the simple J formula implementation"""
    # pylint: disable=invalid-name,too-many-arguments
    result = capillarypressure.simple_J(sw, a, b, poro_ref, perm_ref, drho, g)

    if isinstance(result, (list, np.ndarray)):
        assert np.isclose(result, expected).all()
    else:
        assert np.isclose(result, expected)


@pytest.mark.parametrize(
    "swlheight, swirr, a, b, poro_ref, perm_ref, expected",
    [
        (1, 0, 1, -1, 1, 1, 1),
        (1, 0.1, 1, -1, 1, 1, 1),
    ],
)
def test_swl_from_height_simple_J(swlheight, swirr, a, b, poro_ref, perm_ref, expected):
    """Test the calculation of swlheight from input parameters"""
    # pylint: disable=invalid-name,too-many-arguments
    result = capillarypressure.swl_from_height_simpleJ(
        swlheight, swirr, a, b, poro_ref, perm_ref
    )

    if isinstance(result, (list, np.ndarray)):
        assert np.isclose(result, expected).all()
    else:
        assert np.isclose(result, expected)


@given(
    st.floats(min_value=EPSILON, max_value=100),
    st.floats(min_value=0.001, max_value=1000000),
    st.floats(min_value=-9, max_value=-0.1),
    # Higher b gives OverflowError: math range error
)
def test_inverses_sw_simpleJ(j_value, a, b):
    """Ensure that the pair of functions going from sw to J and back
    are truly inverses of each other"""
    # pylint: disable=invalid-name  # simpleJ
    sw = capillarypressure._simpleJ_to_sw(j_value, a, b)
    assert np.isclose(capillarypressure._sw_to_simpleJ(sw, a, b), j_value)


@given(
    st.floats(min_value=EPSILON, max_value=1),
    st.floats(min_value=0.001, max_value=1000000),
    st.floats(min_value=-9, max_value=-0.1),
)
def test_inverses_simpleJ_sw(sw_value, a, b):
    """Inverse of the test function above"""
    # pylint: disable=invalid-name  # simpleJ
    result = capillarypressure._simpleJ_to_sw(
        capillarypressure._sw_to_simpleJ(sw_value, a, b), a, b
    )
    assert np.isclose(result, sw_value)


@given(
    st.floats(min_value=-10000000, max_value=10000000),  # J
    st.floats(min_value=0.0001, max_value=1),  # poro_ref
    st.floats(min_value=0.1, max_value=10000),  # perm_ref
)
def test_inverses_simpleJ_height(J, poro_ref, perm_ref):
    """Test round-trip calculation of J-value from height"""
    # pylint: disable=invalid-name
    result = capillarypressure._height_to_simpleJ(
        capillarypressure._simpleJ_to_height(J, poro_ref, perm_ref), poro_ref, perm_ref
    )
    assert np.isclose(result, J)


@given(
    st.floats(min_value=-100, max_value=100),  # height
    st.floats(min_value=0.0001, max_value=1),  # poro_ref
    st.floats(min_value=0.1, max_value=10000),  # perm_ref
)
def test_inverses_height_simpleJ(height, poro_ref, perm_ref):
    """Test round-trip calculation of height-value from J"""
    # pylint: disable=invalid-name
    result = capillarypressure._simpleJ_to_height(
        capillarypressure._height_to_simpleJ(height, poro_ref, perm_ref),
        poro_ref,
        perm_ref,
    )
    assert np.isclose(result, height)


def test_reference_implementation_swl_from_height():
    """Test the reference implementation that was copied from Drogon example project"""
    swlheight = 300
    permref = 10
    pororef = 0.3
    a = 1
    b = -2.2
    swirr = 0.02

    # Lines copied from Drogon:
    j_swlheight = swlheight * math.sqrt(permref / pororef)  # J at swlheight
    swn_swlheight = math.pow(j_swlheight / a, 1 / b)  # swn at swlheigh_
    ref_swl = swirr + (1 - swirr) * swn_swlheight  # swl = sw at swlheight

    swl = capillarypressure.swl_from_height_simpleJ(
        swlheight, swirr, a, b, pororef, permref
    )

    assert np.isclose(swl, ref_swl)
