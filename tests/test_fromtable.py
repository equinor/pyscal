# -*- coding: utf-8 -*-
"""Test module for pyscal...fromtable()"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np

import pytest

from hypothesis import given, settings
import hypothesis.strategies as st

from pyscal import WaterOil, GasOil

from test_wateroil import float_df_checker

from test_gasoil import check_table as check_go_table
from test_wateroil import check_table as check_wo_table


def test_ow_fromtable_simple():
    df1 = pd.DataFrame(
        columns=["SW", "KRW", "KROW", "PC"], data=[[0, 0, 1, 2], [1, 1, 0, 0]]
    )
    wo = WaterOil(h=0.1)
    # With wrong names:
    with pytest.raises(ValueError):
        # Here we also get a deprecation warning
        wo.add_oilwater_fromtable(df1)

    # Set names:
    wo.add_fromtable(df1, swcolname="SW")
    # This didn't do anything, because we did not provide any relperm names
    assert "krw" not in wo.table.columns
    assert "krow" not in wo.table.columns
    assert "pc" not in wo.table.columns

    # Try again:
    wo.add_fromtable(
        df1, swcolname="SW", krwcolname="KRW", krowcolname="KROW", pccolname="PC"
    )
    assert "krw" in wo.table.columns
    assert "krow" in wo.table.columns
    assert "pc" in wo.table.columns
    check_wo_table(wo.table)


def test_go_fromtable_simple():
    df1 = pd.DataFrame(
        columns=["SG", "KRG", "KROG", "PC"], data=[[0, 0, 1, 2], [1, 1, 0, 0]]
    )
    go = GasOil(h=0.1)
    go.add_fromtable(
        df1, sgcolname="SG", krgcolname="KRG", krogcolname="KROG", pccolname="PC"
    )
    check_go_table(go.table)


def test_ow_fromtable_multiindex():
    """Test that we accept multiindex dataframes,
    (but a warning will be issued)"""
    # Test an example dataframe that easily gets sent in from ecl2df.satfunc:
    df1 = pd.DataFrame(
        columns=["KEYWORD", "SATNUM", "SW", "KRW", "KROW", "PC"],
        data=[
            ["SWOF", 1, 0, 0, 1, 2],
            ["SWOF", 1, 0.5, 0.5, 0.5, 1],
            ["SWOF", 1, 1, 1, 0, 0],
        ],
    ).set_index(["KEYWORD", "SATNUM"])

    # Check that we have a MultiIndex:
    assert len(df1.index.names) == 2

    wo = WaterOil(h=0.1)
    wo.add_fromtable(
        df1, swcolname="SW", krwcolname="KRW", krowcolname="KROW", pccolname="PC"
    )
    assert "krw" in wo.table.columns
    assert "krow" in wo.table.columns
    assert "pc" in wo.table.columns
    check_wo_table(wo.table)


def test_go_fromtable_problems():
    df1 = pd.DataFrame(
        columns=["Sg", "KRG", "KROG", "PCOG"], data=[[0.1, 0, 1, 2], [0.9, 1, 0, 0]]
    )
    # Now sgcr and swl is wrong:
    go = GasOil(h=0.1)
    with pytest.raises(ValueError):
        # Should say sg must start at zero.
        go.add_fromtable(df1, pccolname="PCOG")
    df2 = pd.DataFrame(
        columns=["Sg", "KRG", "KROG", "PCOG"], data=[[0.0, 0, 1, 2], [0.9, 0.8, 0, 0]]
    )
    with pytest.raises(ValueError):
        # should say too large swl for pcog interpolation
        go = GasOil(h=0.1)
        go.add_fromtable(df2, pccolname="PCOG")
    go = GasOil(h=0.1, swl=0.1)
    go.add_fromtable(df2, pccolname="PCOG")
    assert np.isclose(go.table["pc"].max(), 2.0)
    assert np.isclose(go.table["pc"].min(), 0.0)
    assert np.isclose(go.table["sg"].max(), 0.9)

    go = GasOil(h=0.1)
    with pytest.raises(ValueError):
        # sg must start at zero
        go.add_fromtable(df1, krgcolname="KRG", krogcolname="KROG")
    # This works fine, but we get warnings on swl not seemingly correct
    go.add_fromtable(df2, krgcolname="KRG", krogcolname="KROG")
    # krg will be extrapolated to sg=1
    float_df_checker(go.table, "sg", 1.0, "krg", 0.8)
    float_df_checker(go.table, "sg", 1.0, "krog", 0.0)
    go = GasOil(h=0.1, swl=0.1)
    go.add_fromtable(df2, krgcolname="KRG", krogcolname="KROG")


def test_ow_singlecolumns():
    krw = pd.DataFrame(columns=["Sw", "krw"], data=[[0.15, 0], [0.89, 1], [1, 1]])
    krow = pd.DataFrame(columns=["Sw", "krow"], data=[[0.15, 1], [0.89, 0], [1, 0]])
    pc = pd.DataFrame(columns=["Sw", "pcow"], data=[[0.15, 3], [0.89, 0.1], [1, 0]])
    wo = WaterOil(h=0.1, swl=0.15, sorw=1 - 0.89)
    wo.add_fromtable(krw)
    assert "krw" in wo.table
    assert "krow" not in wo.table
    wo.add_fromtable(krow)
    assert "krow" in wo.table
    wo.add_fromtable(pc)
    assert "pc" in wo.table

    # We want to allow a pc dataframe where sw starts from zero:
    # then should not preserve pc(sw=0)
    pc2 = pd.DataFrame(columns=["Sw", "pcow"], data=[[0, 3], [0.5, 0.1], [1, 0]])
    wo.add_fromtable(pc2)
    assert "pc" in wo.table
    assert wo.table["sw"].min() == 0.15
    assert wo.table["pc"].max() < 3

    # But we should refuse a pc dataframe not covering our sw range:
    wo = WaterOil(h=0.1, swl=0)
    with pytest.raises(ValueError):
        pc3 = pd.DataFrame(columns=["Sw", "pcow"], data=[[0.1, 3], [0.5, 0.1], [1, 0]])
        wo.add_fromtable(pc3)
    with pytest.raises(ValueError):
        pc3 = pd.DataFrame(columns=["Sw", "pcow"], data=[[0, 3], [0.5, 0.1]])
        wo.add_fromtable(pc3)

    # Disallow non-monotonous capillary pressure:
    pc4 = pd.DataFrame(columns=["Sw", "pcow"], data=[[0.15, 3], [0.89, 0.1], [1, 0.1]])
    with pytest.raises(ValueError):
        wo.add_fromtable(pc4)

    # But if capillary pressure is all zero, accept it:
    pc5 = pd.DataFrame(columns=["Sw", "pcow"], data=[[0, 0], [1, 0]])
    wo = WaterOil(h=0.1)
    wo.add_fromtable(pc5)
    assert wo.table["pc"].sum() == 0


def test_ow_invalidcurves():
    # Sw data not ordered:
    krw1 = pd.DataFrame(columns=["Sw", "krw"], data=[[0.15, 0], [0.1, 1], [1, 1]])
    wo = WaterOil(swl=krw1["Sw"].min(), h=0.1)
    with pytest.raises(ValueError):
        # pchip-interpolator raises this error;
        # x coordinates are not in increasing order
        wo.add_fromtable(krw1, krwcolname="krw")

    krw2 = pd.DataFrame(
        columns=["Sw", "krw"], data=[[0.15, 0], [0.4, 0.6], [0.6, 0.4], [1, 1]]
    )
    wo = WaterOil(swl=krw2["Sw"].min(), h=0.1)
    with pytest.raises(ValueError):
        # Should get notified that krw is not monotonous
        wo.add_fromtable(krw2, krwcolname="krw")
    krow2 = pd.DataFrame(
        columns=["Sw", "krow"], data=[[0.15, 1], [0.4, 0.4], [0.6, 0.6], [1, 0]]
    )
    wo = WaterOil(swl=krow2["Sw"].min(), h=0.1)
    with pytest.raises(ValueError):
        # Should get notified that krow is not monotonous
        wo.add_fromtable(krow2, krowcolname="krow")
    pc2 = pd.DataFrame(
        columns=["Sw", "pc"], data=[[0.15, 1], [0.4, 0.4], [0.6, 0.6], [1, 0]]
    )
    wo = WaterOil(swl=pc2["Sw"].min(), h=0.1)
    with pytest.raises(ValueError):
        # Should get notified that pc is not monotonous
        wo.add_fromtable(pc2, pccolname="pc")

    pc3 = pd.DataFrame(
        columns=["Sw", "pc"],
        data=[[0, np.inf], [0.1, 1], [0.4, 0.4], [0.6, 0.2], [1, 0]],
    )
    wo = WaterOil(swl=pc3["Sw"].min(), h=0.1)
    # Will get warning that infinite numbers are ignored.
    # In this case the extrapolation is quite bad.
    wo.add_fromtable(pc3, pccolname="pc")

    # But when we later set swl larger, then we should
    # not bother about the infinity:
    wo = WaterOil(swl=0.1, h=0.1)
    wo.add_fromtable(pc3, pccolname="pc")
    assert np.isclose(wo.table["pc"].max(), pc3.iloc[1:]["pc"].max())

    # Choosing endpoint slightly to the left of 0.1 incurs
    # extrapolation. A warning will be given
    wo = WaterOil(swl=0.05, h=0.1)
    wo.add_fromtable(pc3, pccolname="pc")
    # Inequality due to extrapolation:
    assert wo.table["pc"].max() > pc3.iloc[1:]["pc"].max()


def test_go_invalidcurves():
    # Sw data not ordered:
    krg1 = pd.DataFrame(columns=["Sg", "krg"], data=[[0.15, 0], [0.1, 1], [1, 1]])
    go = GasOil(h=0.1)
    with pytest.raises(ValueError):
        # pchip-interpolator raises this error;
        # x coordinates are not in increasing order
        go.add_fromtable(krg1, krgcolname="krg")

    krg2 = pd.DataFrame(
        columns=["Sg", "krg"], data=[[0.15, 0], [0.4, 0.6], [0.6, 0.4], [1, 1]]
    )
    go = GasOil(h=0.1)
    with pytest.raises(ValueError):
        # Should get notified that krg is not monotonous
        go.add_fromtable(krg2, krgcolname="krg")
    krog2 = pd.DataFrame(
        columns=["Sg", "krog"], data=[[0.15, 1], [0.4, 0.4], [0.6, 0.6], [1, 0]]
    )
    go = GasOil(h=0.1)
    with pytest.raises(ValueError):
        # Should get notified that krog is not monotonous
        go.add_fromtable(krog2, krogcolname="krog")
    pc2 = pd.DataFrame(
        columns=["Sg", "pc"], data=[[0.15, 1], [0.4, 0.4], [0.6, 0.6], [1, 0]]
    )
    go = GasOil(h=0.1)
    with pytest.raises(ValueError):
        # Should get notified that pc is not monotonous
        go.add_fromtable(pc2, pccolname="pc")


def test_ow_fromtable_problems():
    # Implicit swl and sorw in the input, how do we handle that?
    df1 = pd.DataFrame(
        columns=["Sw", "krw", "krow", "pcow"],
        data=[[0.15, 0, 1, 3], [0.89, 1, 0, 0.1], [1, 1, 0, 0]],
    )
    # With default object:
    wo = WaterOil(h=0.1)
    with pytest.raises(ValueError):
        wo.add_fromtable(df1)
        # This results in krw and krow overshooting 0 and 1
    # Fix left endpoint:
    wo = WaterOil(h=0.1, swl=df1["Sw"].min())
    wo.add_fromtable(df1)
    # The table is now valid, but we did not preserve the 0.89 point
    check_wo_table(wo.table)

    # If we also tell the WaterOil object about sorw, we are guaranteed
    # to have it expclitly included:
    wo = WaterOil(h=0.1, swl=df1["Sw"].min(), sorw=1 - 0.89)
    wo.add_fromtable(df1)
    check_wo_table(wo.table)
    # For low enough h, this will however NOT matter.


@settings(deadline=2000)
@given(st.floats(min_value=1e-5, max_value=1))
def test_ow_fromtable_h(h):
    df1 = pd.DataFrame(
        columns=["Sw", "krw", "krow", "pcow"],
        data=[[0.15, 0, 1, 3], [0.89, 1, 0, 0.1], [1, 1, 0, 0]],
    )
    wo = WaterOil(h=h, swl=0.15, sorw=1 - 0.89)
    wo.add_fromtable(df1)
    check_wo_table(wo.table)
