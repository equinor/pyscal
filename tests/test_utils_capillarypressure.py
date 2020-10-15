"""Test module for capillary pressure support code in pyscal"""
import math

import numpy as np
import pytest

from pyscal.utils import capillarypressure

# pyscal.utils.relperm.crosspoint() is tested by test_wateroil and test_gasoil.

PASCAL = 1e-05  # One pascal in bar.


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
        (0.5, 1, 2, 1, 1, 1, 1, 0.5 ** 2 * PASCAL),
        # Vector support in sw:
        (np.array([0, 1]), 1, 1, 1, 1, 1, 1, np.array([0, PASCAL])),
    ],
)
def test_simple_J(sw, a, b, poro_ref, perm_ref, drho, g, expected):
    """Test the simple J formula implementation"""
    # pylint: disable=invalid-name,too-many-arguments
    assert np.isclose(
        capillarypressure.simple_J(sw, a, b, poro_ref, perm_ref, drho, g), expected
    ).all()
