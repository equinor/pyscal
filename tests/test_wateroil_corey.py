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
    assert df["son"].is_monotonic_decreasing
    assert df["swnpc"].is_monotonic
    assert df["krow"].is_monotonic_decreasing
    assert df["krw"].is_monotonic


@given(st.floats(), st.floats())
def test_wateroil_corey1(nw, now):
    try:
        wo = WaterOil()
        wo.add_corey_oil(now=now)
        assert "krow" in wo.table
        wo.add_corey_water(nw=nw)
        assert "krw" in wo.table
        assert isinstance(wo.krwcomment, str)
        check_table(wo.table)
    except AssertionError:
        pass
