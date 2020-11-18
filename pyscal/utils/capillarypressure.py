"""Support functions for capillary pressure"""

import math

from pyscal.constants import MAX_EXPONENT


def simple_J(sw, a, b, poro_ref, perm_ref, drho, g=9.81):
    # pylint: disable=invalid-name,anomalous-backslash-in-string
    r"""Calculate capillary pressure with bar as unit

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

    J = _sw_to_simpleJ(sw, a, b)
    height = _simpleJ_to_height(J, poro_ref, perm_ref)
    return _height_to_pc(height, drho, g)


def _height_to_pc(height, drho, g):
    """From height above free water level, multiplication
    with density difference and gravity gives capillary pressure.

    Args:
        height (float): Height above free water level in meters.
        drho (float): density difference in g/cc
        g (float): gravitational acceleration, in m/s²

    Returns:
        float: capillary pressure at given height, in bars.
    """
    return height * drho / 1000 * g / 100.0


def _sw_to_simpleJ(sw, a, b):
    """Convert a water saturation value to the associated J-value,
    using RMS simple-J"""
    return float(a) * sw ** float(b)


def _simpleJ_to_sw(J, a, b):
    """Convert a J-function-value to a water saturation value,
    using RMS simple-J"""
    return math.pow(J / float(a), 1.0 / float(b))


def _simpleJ_to_height(J, poro_ref, perm_ref):
    """Convert a J-function value to a height-value in meters

    This scales the J-value with the inverse of characteristic
    length of the capillaries radii, estimated as sqrt(k/phi).

    https://en.wikipedia.org/wiki/Leverett_J-function

    Args:
        J (float): J-function value.
        poro_ref (float): Porosity between 0 and 1
        perm_ref (float): Permeability in milliDarcy
    """
    return J * math.sqrt(float(poro_ref) / float(perm_ref))


def _height_to_simpleJ(H, poro_ref, perm_ref):
    """Convert a height value (in meters) to a corresponding J-function

    This scales the J-value with the characteristic
    length of the capillaries radii, estimated as sqrt(k/phi).

    https://en.wikipedia.org/wiki/Leverett_J-function

    Args:
        H (float): Height in meters.
        poro_ref (float): Porosity between 0 and 1
        perm_ref (float): Permeability in milliDarcy
    """
    return H * math.sqrt(float(perm_ref) / float(poro_ref))


def swl_from_height_simpleJ(swlheight, swirr, a, b, poro_ref, perm_ref):
    # pylint: disable=invalid-name
    """Calculate a swl value based on a height parameter.

    The height parameter is typically meters above free water level (FWL)

    Args:
        swlheight (float): Height above free water level, in meters,
            where we want swl to be.
        swirr (float): Asymptotic irreducible water saturation. This is
            used to normalized the outputted swl.
        a (float): a coefficient in RMS simplified J function
        b (float): b coefficient in RMS simplified J function
        poro_ref (float): Reference porosity for scaling to Pc, between 0 and 1
        perm_ref (float): Reference permeability for scaling to Pc, in milliDarcy

    Returns:
        swl (float): A value that can be used for swl.
    """
    # Calculate the value of J() far above free water level:
    j_value_at_swlheight = _height_to_simpleJ(swlheight, poro_ref, perm_ref)

    # Calculate the water saturation value that corresponds to this J-value,
    # un-normalized:
    swn_swlheight = _simpleJ_to_sw(j_value_at_swlheight, a, b)

    # Normalize with respect the the asymptotic swirr and return
    return swirr + (1 - swirr) * swn_swlheight
