# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import pandas as pd

import pytest

from pyscal import WaterOil, PyscalFactory


def test_factory_wateroil():
    """Test that we can create curves from dictionaries of parameters"""

    logging.getLogger().setLevel("INFO")

    factory = PyscalFactory()

    wo = factory.create_water_oil()
    assert isinstance(wo, WaterOil)

    with pytest.raises(TypeError):
        factory.create_water_oil(swirr=0.01)  # Must be a dictionary

    wo = factory.create_water_oil(dict(tag="Good sand"))
    assert wo.tag == "Good sand"

    wo = factory.create_water_oil(dict(swirr=0.01, swl=0.1, bogus="foobar"))
    assert isinstance(wo, WaterOil)
    assert wo.swirr == 0.01
    assert wo.swl == 0.1
    assert "krw" not in wo.table  # A zero column would be okay though.

    # (static functions)
    wo = PyscalFactory.create_water_oil(dict(nw=3, krwend=0.2, krwmax=0.5))
    assert isinstance(wo, WaterOil)
    assert "krw" in wo.table
    assert "Corey" in wo.krwcomment
    assert wo.table["krw"].max() == 0.2  # Because sorw==0 by default

    wo = factory.create_water_oil(dict(nw=3, sorw=0.1, krwend=0.2, krwmax=0.5))
    assert isinstance(wo, WaterOil)
    assert "krw" in wo.table
    assert "Corey" in wo.krwcomment
    assert wo.table["krw"].max() == 0.5

    # Ambiguous works, but we don't guarantee that this results
    # in LET or Corey.
    wo = factory.create_water_oil(dict(nw=3, Lw=2, Ew=2, Tw=2))
    assert "krw" in wo.table
    assert "Corey" in wo.krwcomment or "LET" in wo.krwcomment

    wo = factory.create_water_oil(dict(Lw=2, Ew=2, Tw=2, krwend=1))
    assert isinstance(wo, WaterOil)
    assert "krw" in wo.table
    assert wo.table["krw"].max() == 1.0
    assert "LET" in wo.krwcomment

    wo = factory.create_water_oil(
        dict(Lw=2, Ew=2, Tw=2, Low=3, Eow=3, Tow=3, krwend=0.5)
    )
    assert isinstance(wo, WaterOil)
    assert "krw" in wo.table
    assert "krow" in wo.table
    assert wo.table["krw"].max() == 0.5
    assert wo.table["krow"].max() == 1
    assert "LET" in wo.krwcomment
    assert "LET" in wo.krowcomment

    # Some missing parameters, will give warnings.
    wo = factory.create_water_oil(dict(Lw=2, Tw=2, Low=3, Eow=3))
    assert isinstance(wo, WaterOil)
    assert "krw" not in wo.table
    assert "krow" not in wo.table

    # Add capillary pressure
    wo = factory.create_water_oil(
        dict(swl=0.1, nw=1, now=1, a=2, b=-1, poro_ref=0.2, perm_ref=100, drho=200)
    )
    assert "pc" in wo.table
    assert wo.table["pc"].max() > 0.0
    assert "Simplified J" in wo.pccomment

    # Test that the optional gravity g is picked up:
    wo = factory.create_water_oil(
        dict(swl=0.1, nw=1, now=1, a=2, b=-1, poro_ref=0.2, perm_ref=100, drho=200, g=0)
    )
    assert "pc" in wo.table
    assert wo.table["pc"].max() == 0.0

    # One pc param missing:
    wo = factory.create_water_oil(
        dict(swl=0.1, nw=1, now=1, a=2, b=-1, perm_ref=100, drho=200, g=0)
    )
    assert "pc" not in wo.table


def test_xls_factory():
    """Test/demonstrate how to go from data in an excel row to pyscal objects"""

    if "__file__" in globals():
        # Easen up copying test code into interactive sessions
        testdir = os.path.dirname(os.path.abspath(__file__))
    else:
        testdir = os.path.abspath(".")

    xlsxfile = testdir + "/data/scal-pc-input-example.xlsx"

    scalinput = pd.read_excel(xlsxfile).set_index(["SATNUM", "CASE"])

    satnum1 = PyscalFactory.create_water_oil(dict(scalinput.loc[1, "low"]))
    swof1 = satnum1.SWOF()
    assert "LET krw" in swof1
    assert "LET krow" in swof1
    assert "Simplified J" in swof1
