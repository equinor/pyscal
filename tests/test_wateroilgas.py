"""Test module for the WaterOilGas object"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyscal import WaterOilGas, WaterOil, GasOil


def test_threephasecheck():
    """Test three phase consistency checks"""
    wog = WaterOilGas()
    wog.wateroil.add_corey_water(nw=2)
    wog.wateroil.add_corey_oil(now=2, kroend=0.9)
    wog.gasoil.add_corey_gas(ng=2)
    wog.gasoil.add_corey_oil(nog=2, kroend=1)
    assert not wog.threephaseconsistency()

    wog = WaterOilGas()
    # To trigger this, we need to hack the WaterOilGas object
    # by overriding the effect of its __init__
    wog.wateroil = WaterOil(swl=0.4)
    wog.gasoil = GasOil(swl=0.2)
    wog.wateroil.add_corey_water(nw=2)
    wog.wateroil.add_corey_oil(now=2, kroend=0.9)
    wog.gasoil.add_corey_gas(ng=2)
    wog.gasoil.add_corey_oil(nog=2, kroend=1)
    assert not wog.threephaseconsistency()
