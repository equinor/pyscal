# -*- coding: utf-8 -*-
"""Support functions for capillary pressure"""

import math

from pyscal.constants import MAX_EXPONENT


def simple_J(sw, a, b, poro_ref, perm_ref, drho, g=9.81):
    # pylint: disable=invalid-name
    """Calculate capillary pressure with bar as unit

    RMS version:

    .. math::

        J = a S_w^b

    *J* is not dimensionless in this equation.

    This is identical to the also seen formula

    .. math::

       J = 10^{b \log(S_w) + \log(a)}


    Args:
        sw: float, np.array or similar, water saturation value to be used.
            Normalize when needed.
        a (float): a coefficient
        b (float): b coefficient
        poro_ref (float): Reference porosity for scaling to Pc, between 0 and 1
        perm_ref (float): Reference permeability for scaling to Pc, in milliDarcy
        drho (float): Density difference between water and oil, in SI units kg/m³.
            Default value is 300
        g (float): Gravitational acceleration, in SI units m/s²,
            default value is 9.81

    Returns:
        capillary pressure, same type as swnpc input argument.
    """  # noqa
    assert g >= 0
    assert b < MAX_EXPONENT
    assert b > -MAX_EXPONENT
    assert 0.0 <= poro_ref <= 1.0
    assert perm_ref > 0.0

    J = a * sw ** b
    H = J * math.sqrt(poro_ref / perm_ref)
    # Scale drho and g from SI units to g/cc and m/s²100
    return H * drho / 1000 * g / 100.0
