# -*- coding: utf-8 -*-
"""Test module for relperm"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pytest
from hypothesis import given
import hypothesis.strategies as st

import pandas as pd

from pyscal import WaterOil


def check_table(df):
    assert not df.empty
    assert not df.isnull().values.any()
    assert len(df["sw"].unique()) == len(df)
    assert df["sw"].is_monotonic
    assert (df["sw"] >= 0.0).all()
    assert df["swn"].is_monotonic
    # assert df['son'].is_monotonic_decreasing


@given(st.floats(), st.floats(), st.floats(), st.floats(), st.floats(), st.text())
def test_wateroil_random(swirr, swl, swcr, sorw, h, tag):
    """Shoot wildly with arguments, the code should throw ValueError
    or AssertionError when input is invalid, but we don't want other crashes"""
    try:
        wo = WaterOil(swirr=swirr, swl=swl, swcr=swcr, sorw=sorw, h=h, tag=tag)
        assert not wo.table.empty
        assert not wo.table.isnull().values.any()
    except AssertionError:
        pass


@given(st.floats(min_value=0, max_value=1))
def test_wateroil_swir(swirr):
    wo = WaterOil(swirr=swirr)
    check_table(wo.table)


@given(st.floats(min_value=0, max_value=1))
def test_wateroil_swl(swl):
    wo = WaterOil(swl=swl)
    check_table(wo.table)


@given(st.floats(min_value=0, max_value=1))
def test_wateroil_sensible(swcr):
    wo = WaterOil(swcr=swcr)
    check_table(wo.table)


@given(st.floats(min_value=0, max_value=1))
def test_wateroil_sensible(sorw):
    wo = WaterOil(sorw=sorw)
    check_table(wo.table)
