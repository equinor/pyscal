# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyscal import WaterOil, PyscalFactory


def test_factory_wateroil():
    factory = PyscalFactory()
    wo = factory.createWaterOil(dict(swirr=0.01, swl=0.1, bogus="foobar"))
    assert isinstance(wo, WaterOil)
    assert wo.swirr == 0.01
    assert wo.swl == 0.1

    wo = factory.createWaterOil(dict(nw=3, krwend=0.2, krwmax=0.2))
    assert isinstance(wo, WaterOil)
    assert "krw" in wo.table
    assert wo.table["krw"].max() == 0.2
