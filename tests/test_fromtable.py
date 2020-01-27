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

from common import check_table, float_df_checker


def test_wo_fromtable_simple():
    """Test loading a simple curve from a table"""
    df1 = pd.DataFrame(
        columns=["SW", "KRW", "KROW", "PC"], data=[[0, 0, 1, 2], [1, 1, 0, 0]]
    )
    wateroil = WaterOil(h=0.1)
    # With wrong names:
    with pytest.raises(ValueError):
        # Here we also get a deprecation warning
        wateroil.add_oilwater_fromtable(df1)

    # Set names:
    wateroil.add_fromtable(df1, swcolname="SW")
    # This didn't do anything, because we did not provide any relperm names
    assert "krw" not in wateroil.table.columns
    assert "krow" not in wateroil.table.columns
    assert "pc" not in wateroil.table.columns

    # Try again:
    wateroil.add_fromtable(
        df1, swcolname="SW", krwcolname="KRW", krowcolname="KROW", pccolname="PC"
    )
    assert "krw" in wateroil.table.columns
    assert "krow" in wateroil.table.columns
    assert "pc" in wateroil.table.columns
    check_table(wateroil.table)


def test_go_fromtable_simple():
    """Test reading of a simple gasoil table"""
    df1 = pd.DataFrame(
        columns=["SG", "KRG", "KROG", "PC"], data=[[0, 0, 1, 2], [1, 1, 0, 0]]
    )
    gasoil = GasOil(h=0.1)
    gasoil.add_fromtable(
        df1, sgcolname="SG", krgcolname="KRG", krogcolname="KROG", pccolname="PC"
    )
    check_table(gasoil.table)


def test_wo_fromtable_multiindex():
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

    wateroil = WaterOil(h=0.1)
    wateroil.add_fromtable(
        df1, swcolname="SW", krwcolname="KRW", krowcolname="KROW", pccolname="PC"
    )
    assert "krw" in wateroil.table.columns
    assert "krow" in wateroil.table.columns
    assert "pc" in wateroil.table.columns
    check_table(wateroil.table)


def test_go_fromtable_problems():
    """Test loading from a table where there should be problems"""
    df1 = pd.DataFrame(
        columns=["Sg", "KRG", "KROG", "PCOG"], data=[[0.1, 0, 1, 2], [0.9, 1, 0, 0]]
    )
    # Now sgcr and swl is wrong:
    gasoil = GasOil(h=0.1)
    with pytest.raises(ValueError):
        # Should say sg must start at zero.
        gasoil.add_fromtable(df1, pccolname="PCOG")
    df2 = pd.DataFrame(
        columns=["Sg", "KRG", "KROG", "PCOG"], data=[[0.0, 0, 1, 2], [0.9, 0.8, 0, 0]]
    )
    with pytest.raises(ValueError):
        # should say too large swl for pcog interpolation
        gasoil = GasOil(h=0.1)
        gasoil.add_fromtable(df2, pccolname="PCOG")
    gasoil = GasOil(h=0.1, swl=0.1)
    gasoil.add_fromtable(df2, pccolname="PCOG")
    assert np.isclose(gasoil.table["pc"].max(), 2.0)
    assert np.isclose(gasoil.table["pc"].min(), 0.0)
    assert np.isclose(gasoil.table["sg"].max(), 0.9)

    gasoil = GasOil(h=0.1)
    with pytest.raises(ValueError):
        # sg must start at zero
        gasoil.add_fromtable(df1, krgcolname="KRG", krogcolname="KROG")
    # This works fine, but we get warnings on swl not seemingly correct
    gasoil.add_fromtable(df2, krgcolname="KRG", krogcolname="KROG")
    # krg will be extrapolated to sg=1
    float_df_checker(gasoil.table, "sg", 1.0, "krg", 0.8)
    float_df_checker(gasoil.table, "sg", 1.0, "krog", 0.0)
    gasoil = GasOil(h=0.1, swl=0.1)
    gasoil.add_fromtable(df2, krgcolname="KRG", krogcolname="KROG")


def test_wo_singlecolumns():
    """Test that we can load single columns from individual dataframes"""
    krw = pd.DataFrame(columns=["Sw", "krw"], data=[[0.15, 0], [0.89, 1], [1, 1]])
    krow = pd.DataFrame(columns=["Sw", "krow"], data=[[0.15, 1], [0.89, 0], [1, 0]])
    pc1 = pd.DataFrame(columns=["Sw", "pcow"], data=[[0.15, 3], [0.89, 0.1], [1, 0]])
    wateroil = WaterOil(h=0.1, swl=0.15, sorw=1 - 0.89)
    wateroil.add_fromtable(krw)
    assert "krw" in wateroil.table
    assert "krow" not in wateroil.table
    wateroil.add_fromtable(krow)
    assert "krow" in wateroil.table
    wateroil.add_fromtable(pc1)
    assert "pc" in wateroil.table

    # We want to allow a pc dataframe where sw starts from zero:
    # then should not preserve pc(sw=0)
    pc2 = pd.DataFrame(columns=["Sw", "pcow"], data=[[0, 3], [0.5, 0.1], [1, 0]])
    wateroil.add_fromtable(pc2)
    assert "pc" in wateroil.table
    assert wateroil.table["sw"].min() == 0.15
    assert wateroil.table["pc"].max() < 3

    # But we should refuse a pc dataframe not covering our sw range:
    wateroil = WaterOil(h=0.1, swl=0)
    with pytest.raises(ValueError):
        pc3 = pd.DataFrame(columns=["Sw", "pcow"], data=[[0.1, 3], [0.5, 0.1], [1, 0]])
        wateroil.add_fromtable(pc3)
    with pytest.raises(ValueError):
        pc3 = pd.DataFrame(columns=["Sw", "pcow"], data=[[0, 3], [0.5, 0.1]])
        wateroil.add_fromtable(pc3)

    # Disallow non-monotonous capillary pressure:
    pc4 = pd.DataFrame(columns=["Sw", "pcow"], data=[[0.15, 3], [0.89, 0.1], [1, 0.1]])
    with pytest.raises(ValueError):
        wateroil.add_fromtable(pc4)

    # But if capillary pressure is all zero, accept it:
    pc5 = pd.DataFrame(columns=["Sw", "pcow"], data=[[0, 0], [1, 0]])
    wateroil = WaterOil(h=0.1)
    wateroil.add_fromtable(pc5)
    assert wateroil.table["pc"].sum() == 0


def test_wo_invalidcurves():
    """Test what happens when we give in invalid data"""
    # Sw data not ordered:
    krw1 = pd.DataFrame(columns=["Sw", "krw"], data=[[0.15, 0], [0.1, 1], [1, 1]])
    wateroil = WaterOil(swl=krw1["Sw"].min(), h=0.1)
    with pytest.raises(ValueError):
        # pchip-interpolator raises this error;
        # x coordinates are not in increasing order
        wateroil.add_fromtable(krw1, krwcolname="krw")

    krw2 = pd.DataFrame(
        columns=["Sw", "krw"], data=[[0.15, 0], [0.4, 0.6], [0.6, 0.4], [1, 1]]
    )
    wateroil = WaterOil(swl=krw2["Sw"].min(), h=0.1)
    with pytest.raises(ValueError):
        # Should get notified that krw is not monotonous
        wateroil.add_fromtable(krw2, krwcolname="krw")
    krow2 = pd.DataFrame(
        columns=["Sw", "krow"], data=[[0.15, 1], [0.4, 0.4], [0.6, 0.6], [1, 0]]
    )
    wateroil = WaterOil(swl=krow2["Sw"].min(), h=0.1)
    with pytest.raises(ValueError):
        # Should get notified that krow is not monotonous
        wateroil.add_fromtable(krow2, krowcolname="krow")
    pc2 = pd.DataFrame(
        columns=["Sw", "pc"], data=[[0.15, 1], [0.4, 0.4], [0.6, 0.6], [1, 0]]
    )
    wateroil = WaterOil(swl=pc2["Sw"].min(), h=0.1)
    with pytest.raises(ValueError):
        # Should get notified that pc is not monotonous
        wateroil.add_fromtable(pc2, pccolname="pc")

    pc3 = pd.DataFrame(
        columns=["Sw", "pc"],
        data=[[0, np.inf], [0.1, 1], [0.4, 0.4], [0.6, 0.2], [1, 0]],
    )
    wateroil = WaterOil(swl=pc3["Sw"].min(), h=0.1)
    # Will get warning that infinite numbers are ignored.
    # In this case the extrapolation is quite bad.
    wateroil.add_fromtable(pc3, pccolname="pc")

    # But when we later set swl larger, then we should
    # not bother about the infinity:
    wateroil = WaterOil(swl=0.1, h=0.1)
    wateroil.add_fromtable(pc3, pccolname="pc")
    assert np.isclose(wateroil.table["pc"].max(), pc3.iloc[1:]["pc"].max())

    # Choosing endpoint slightly to the left of 0.1 incurs
    # extrapolation. A warning will be given
    wateroil = WaterOil(swl=0.05, h=0.1)
    wateroil.add_fromtable(pc3, pccolname="pc")
    # Inequality due to extrapolation:
    assert wateroil.table["pc"].max() > pc3.iloc[1:]["pc"].max()


def test_go_invalidcurves():
    """Test  fromtable on invalid gasoil data"""
    # Sw data not ordered:
    krg1 = pd.DataFrame(columns=["Sg", "krg"], data=[[0.15, 0], [0.1, 1], [1, 1]])
    gasoil = GasOil(h=0.1)
    with pytest.raises(ValueError):
        # pchip-interpolator raises this error;
        # x coordinates are not in increasing order
        gasoil.add_fromtable(krg1, krgcolname="krg")

    krg2 = pd.DataFrame(
        columns=["Sg", "krg"], data=[[0.15, 0], [0.4, 0.6], [0.6, 0.4], [1, 1]]
    )
    gasoil = GasOil(h=0.1)
    with pytest.raises(ValueError):
        # Should get notified that krg is not monotonous
        gasoil.add_fromtable(krg2, krgcolname="krg")
    krog2 = pd.DataFrame(
        columns=["Sg", "krog"], data=[[0.15, 1], [0.4, 0.4], [0.6, 0.6], [1, 0]]
    )
    gasoil = GasOil(h=0.1)
    with pytest.raises(ValueError):
        # Should get notified that krog is not monotonous
        gasoil.add_fromtable(krog2, krogcolname="krog")
    pc2 = pd.DataFrame(
        columns=["Sg", "pc"], data=[[0.15, 1], [0.4, 0.4], [0.6, 0.6], [1, 0]]
    )
    gasoil = GasOil(h=0.1)
    with pytest.raises(ValueError):
        # Should get notified that pc is not monotonous
        gasoil.add_fromtable(pc2, pccolname="pc")


def test_wo_fromtable_problems():
    """Test wateroil from tables with problematic data"""
    # Implicit swl and sorw in the input, how do we handle that?
    df1 = pd.DataFrame(
        columns=["Sw", "krw", "krow", "pcow"],
        data=[[0.15, 0, 1, 3], [0.89, 1, 0, 0.1], [1, 1, 0, 0]],
    )
    # With default object:
    wateroil = WaterOil(h=0.1)
    with pytest.raises(ValueError):
        wateroil.add_fromtable(df1)
        # This results in krw and krow overshooting 0 and 1
    # Fix left endpoint:
    wateroil = WaterOil(h=0.1, swl=df1["Sw"].min())
    wateroil.add_fromtable(df1)
    # The table is now valid, but we did not preserve the 0.89 point
    check_table(wateroil.table)

    # If we also tell the WaterOil object about sorw, we are guaranteed
    # to have it expclitly included:
    wateroil = WaterOil(h=0.1, swl=df1["Sw"].min(), sorw=1 - 0.89)
    wateroil.add_fromtable(df1)
    check_table(wateroil.table)
    # For low enough h, this will however NOT matter.


@settings(deadline=2000)
@given(st.floats(min_value=1e-5, max_value=1))
def test_wo_fromtable_h(h):
    """Test making curves from tabular data with random stepsize h"""
    df1 = pd.DataFrame(
        columns=["Sw", "krw", "krow", "pcow"],
        data=[[0.15, 0, 1, 3], [0.89, 1, 0, 0.1], [1, 1, 0, 0]],
    )
    wateroil = WaterOil(h=h, swl=0.15, sorw=1 - 0.89)
    wateroil.add_fromtable(df1)
    check_table(wateroil.table)
