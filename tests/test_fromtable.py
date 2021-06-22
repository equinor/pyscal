"""Test module for pyscal.*.add_fromtable()"""

import pandas as pd
import numpy as np

import pytest

from hypothesis import given, settings
import hypothesis.strategies as st

from pyscal import WaterOil, GasOil, WaterOilGas

from pyscal.utils.testing import check_table, float_df_checker


def test_wo_fromtable_simple():
    """Test loading a simple curve from a table"""
    df1 = pd.DataFrame(
        columns=["SW", "KRW", "KROW", "PC"], data=[[0, 0, 1, 2], [1, 1, 0, 0]]
    )
    wateroil = WaterOil(h=0.1)
    # With wrong names:
    with pytest.raises(ValueError):
        wateroil.add_fromtable(df1, swcolname="sw")

    # Set names:
    wateroil.add_fromtable(df1, swcolname="SW", pccolname="PC")
    assert "KRW" in wateroil.table.columns
    assert "KROW" in wateroil.table.columns
    assert "PC" in wateroil.table.columns
    assert sum(wateroil.table["KRW"]) > 0
    assert sum(wateroil.table["KROW"]) > 0
    assert np.isclose(sum(wateroil.table["PC"]), 11)  # Linearly increasing PC
    check_table(wateroil.table)


def test_go_fromtable_simple():
    """Test reading of a simple gasoil table"""
    df1 = pd.DataFrame(
        columns=["SG", "KRG", "KROG", "PC"], data=[[0, 0, 1, 0], [1, 1, 0, 2]]
    )
    gasoil = GasOil(h=0.1)
    gasoil.add_fromtable(
        df1, sgcolname="SG", krgcolname="KRG", krogcolname="KROG", pccolname="PC"
    )
    assert sum(gasoil.table["KRG"]) > 0
    assert sum(gasoil.table["KROG"]) > 0
    assert np.isclose(sum(gasoil.table["PC"]), 11)  # Linearly increasing PCOG
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
    assert "KRW" in wateroil.table.columns
    assert "KROW" in wateroil.table.columns
    assert "PC" in wateroil.table.columns
    check_table(wateroil.table)


def test_go_fromtable_problems():
    """Test loading from a table where there should be problems"""
    df1 = pd.DataFrame(
        columns=["SG", "KRG", "KROG", "PCOG"], data=[[0.1, 0, 1, 0], [0.9, 1, 0, 2]]
    )
    # Now sgcr and swl is wrong:
    gasoil = GasOil(h=0.1)
    with pytest.raises(ValueError):
        # Should say sg must start at zero.
        gasoil.add_fromtable(df1, pccolname="PCOG")
    df2 = pd.DataFrame(
        columns=["SG", "KRG", "KROG", "PCOG"], data=[[0.0, 0, 1, 0], [0.9, 0.8, 0, 2]]
    )
    with pytest.raises(ValueError):
        # should say too large swl for pcog interpolation
        gasoil = GasOil(h=0.1)
        gasoil.add_fromtable(df2, pccolname="PCOG")
    gasoil = GasOil(h=0.1, swl=0.1)
    gasoil.add_fromtable(df2, pccolname="PCOG")
    assert np.isclose(gasoil.table["PC"].max(), 2.0)
    assert np.isclose(gasoil.table["PC"].min(), 0.0)
    assert np.isclose(gasoil.table["SG"].max(), 0.9)

    gasoil = GasOil(h=0.1)
    with pytest.raises(ValueError):
        # sg must start at zero
        gasoil.add_fromtable(df1, krgcolname="KRG", krogcolname="KROG")
    # This works fine, but we get warnings on swl not seemingly correct
    gasoil.add_fromtable(df2, krgcolname="KRG", krogcolname="KROG", pccolname=None)
    # krg will be extrapolated to sg=1
    float_df_checker(gasoil.table, "SG", 1.0, "KRG", 0.8)
    float_df_checker(gasoil.table, "SG", 1.0, "KROG", 0.0)
    gasoil = GasOil(h=0.1, swl=0.1)
    gasoil.add_fromtable(df2, krgcolname="KRG", krogcolname="KROG")

    df3 = pd.DataFrame(
        columns=["SG", "KRG", "KROG", "PCOG"], data=[[0, -0.01, 1, 0], [1, 1, 0, 0]]
    )
    gasoil = GasOil(h=0.1)
    with pytest.raises(ValueError):
        # Should say krg is negative
        gasoil.add_fromtable(df3, krgcolname="KRG")

    df4 = pd.DataFrame(
        columns=["SG", "KRG", "KROG", "PCOG"],
        data=[[0, 0, 1, 0], [1, 1.0000000001, 0, 0]],
    )
    gasoil = GasOil(h=0.1)
    with pytest.raises(ValueError):
        # Should say krg is above 1.0
        print(df4)
        gasoil.add_fromtable(df4, krgcolname="KRG")


def test_gascondensate():
    """Test how sgro works when data is tabulated. Sgro is to be used
    for gas condensate modelling. sgro is tricky because only 0 and sgcr
    are valid values for sgro when constructing the objects."""

    # When sgro is nonzero, we are doing gas condensate modelling:
    gascond_orig = GasOil(sgro=0.2, sgcr=0.2, h=0.1)
    gascond_orig.add_corey_gas()
    gascond_orig.add_corey_oil(kroend=0.8)

    # Make a new object without assuming anything about sgro and sgcr:
    gascond_tabulated = GasOil(h=0.1)
    gascond_tabulated.add_fromtable(gascond_orig.table)
    check_table(gascond_tabulated.table)

    # sgro is estimated correctly in this case:
    assert np.isclose(gascond_tabulated.sgro, gascond_orig.sgro)

    # The object constructed from the table has an extra row at SG=0.1, because
    # we didn't tell it that sgcr was 0.2, otherwise they are numerically
    # equivalent:
    assert len(gascond_tabulated.table) == len(gascond_orig.table) + 1
    pd.testing.assert_frame_equal(
        gascond_orig.table.iloc[1:][["SG", "KRG", "KROG"]].reset_index(drop=True),
        gascond_tabulated.table.iloc[2:][["SG", "KRG", "KROG"]].reset_index(drop=True),
    )

    # Make a tricky gascondensate object which has a linear oil curve. The sgro
    # estimate will become 1.0, we should still be able to use this as a table.
    gascond_linear = GasOil(sgro=0.2, sgcr=0.2, h=0.1)
    gascond_linear.add_corey_gas()
    gascond_linear.add_corey_oil(kroend=0.8, kromax=1, nog=1)
    gascond_linear_tabulated = GasOil(h=0.1)
    gascond_linear_tabulated.add_fromtable(gascond_linear.table)

    check_table(gascond_linear_tabulated.table)

    # sgro could not be guessed here, and is reset to zero:
    assert np.isclose(gascond_linear_tabulated.sgro, 0.0)


def test_wo_singlecolumns():
    """Test that we can load single columns from individual dataframes"""
    krw = pd.DataFrame(columns=["SW", "KRW"], data=[[0.15, 0], [0.89, 1], [1, 1]])
    krow = pd.DataFrame(columns=["SW", "KROW"], data=[[0.15, 1], [0.89, 0], [1, 0]])
    pc1 = pd.DataFrame(columns=["SW", "PCOW"], data=[[0.15, 3], [0.89, 0.1], [1, 0]])
    wateroil = WaterOil(h=0.1, swl=0.15, sorw=1 - 0.89)
    wateroil.add_fromtable(krw)
    assert "KRW" in wateroil.table
    assert "KROW" not in wateroil.table
    wateroil.add_fromtable(krow)
    assert "KROW" in wateroil.table
    wateroil.add_fromtable(pc1)
    assert "PC" in wateroil.table

    # We want to allow a pc dataframe where sw starts from zero:
    # then should not preserve pc(sw=0)
    pc2 = pd.DataFrame(columns=["SW", "PCOW"], data=[[0, 3], [0.5, 0.1], [1, 0]])
    wateroil.add_fromtable(pc2)
    assert "PC" in wateroil.table
    assert wateroil.table["SW"].min() == 0.15
    assert wateroil.table["PC"].max() < 3

    # But we should refuse a pc dataframe not covering our sw range:
    wateroil = WaterOil(h=0.1, swl=0)
    with pytest.raises(ValueError):
        pc3 = pd.DataFrame(columns=["SW", "PCOW"], data=[[0.1, 3], [0.5, 0.1], [1, 0]])
        wateroil.add_fromtable(pc3)
    with pytest.raises(ValueError):
        pc3 = pd.DataFrame(columns=["SW", "PCOW"], data=[[0, 3], [0.5, 0.1]])
        wateroil.add_fromtable(pc3)

    # Disallow non-monotonous capillary pressure:
    pc4 = pd.DataFrame(columns=["SW", "PCOW"], data=[[0.15, 3], [0.89, 0.1], [1, 0.1]])
    with pytest.raises(ValueError):
        wateroil.add_fromtable(pc4)

    # But if capillary pressure is all zero, accept it:
    pc5 = pd.DataFrame(columns=["SW", "PCOW"], data=[[0, 0], [1, 0]])
    wateroil = WaterOil(h=0.1)
    wateroil.add_fromtable(pc5)
    assert wateroil.table["PC"].sum() == 0


def test_ieee_754():
    """Test difficult values from experienced bugs related to
    IEE754 representation errors"""
    wateroilgas = WaterOilGas(swl=0.18)
    # For this particular swl value, we get:
    # wateroilgas.gasoil.table.sg.max()
    # Out[18]: 0.8200000000000001  # = (1 - 0.18)

    # Can we then interpolate from a table that goes up to 0.82?
    sgof = pd.DataFrame(
        columns=["SG", "KRG", "KROG", "PCOG"], data=[[0, 0, 1, 0], [0.82, 1, 0, 0]]
    )
    # Note, replacing 0.82 in the table above with 1-0.18 *is not the same*
    wateroilgas.gasoil.add_fromtable(sgof)
    assert wateroilgas.gasoil.table["KRG"].max() == 1.0
    assert wateroilgas.gasoil.table["KROG"].min() == 0.0
    assert wateroilgas.gasoil.table["PC"].min() == 0.0
    assert wateroilgas.gasoil.table["PC"].max() == 0.0


def test_wo_invalidcurves():
    """Test what happens when we give in invalid data"""
    # Sw data not ordered:
    krw1 = pd.DataFrame(columns=["SW", "KRW"], data=[[0.15, 0], [0.1, 1], [1, 1]])
    wateroil = WaterOil(swl=krw1["SW"].min(), h=0.1)
    with pytest.raises(ValueError):
        # pchip-interpolator raises this error;
        # x coordinates are not in increasing order
        wateroil.add_fromtable(krw1, krwcolname="KRW")

    krw2 = pd.DataFrame(
        columns=["SW", "KRW"], data=[[0.15, 0], [0.4, 0.6], [0.6, 0.4], [1, 1]]
    )
    wateroil = WaterOil(swl=krw2["SW"].min(), h=0.1)
    with pytest.raises(ValueError):
        # Should get notified that krw is not monotonous
        wateroil.add_fromtable(krw2, krwcolname="KRW")
    krow2 = pd.DataFrame(
        columns=["SW", "KROW"], data=[[0.15, 1], [0.4, 0.4], [0.6, 0.6], [1, 0]]
    )
    wateroil = WaterOil(swl=krow2["SW"].min(), h=0.1)
    with pytest.raises(ValueError):
        # Should get notified that krow is not monotonous
        wateroil.add_fromtable(krow2, krowcolname="KROW")
    pc2 = pd.DataFrame(
        columns=["SW", "PC"], data=[[0.15, 1], [0.4, 0.4], [0.6, 0.6], [1, 0]]
    )
    wateroil = WaterOil(swl=pc2["SW"].min(), h=0.1)
    with pytest.raises(ValueError):
        # Should get notified that pc is not monotonous
        wateroil.add_fromtable(pc2, pccolname="PC")

    pc3 = pd.DataFrame(
        columns=["SW", "PC"],
        data=[[0, np.inf], [0.1, 1], [0.4, 0.4], [0.6, 0.2], [1, 0]],
    )
    wateroil = WaterOil(swl=pc3["SW"].min(), h=0.1)
    # Will get warning that infinite numbers are ignored.
    # In this case the extrapolation is quite bad.
    wateroil.add_fromtable(pc3, pccolname="PC")

    # But when we later set swl larger, then we should
    # not bother about the infinity:
    wateroil = WaterOil(swl=0.1, h=0.1)
    wateroil.add_fromtable(pc3, pccolname="PC")
    assert np.isclose(wateroil.table["PC"].max(), pc3.iloc[1:]["PC"].max())

    # Choosing endpoint slightly to the left of 0.1 incurs
    # extrapolation. A warning will be given
    wateroil = WaterOil(swl=0.05, h=0.1)
    wateroil.add_fromtable(pc3, pccolname="PC")
    # Inequality due to extrapolation:
    assert wateroil.table["PC"].max() > pc3.iloc[1:]["PC"].max()


def test_go_invalidcurves():
    """Test  fromtable on invalid gasoil data"""
    # Sw data not ordered:
    krg1 = pd.DataFrame(columns=["SG", "KRG"], data=[[0.15, 0], [0.1, 1], [1, 1]])
    gasoil = GasOil(h=0.1)
    with pytest.raises(ValueError):
        # pchip-interpolator raises this error;
        # x coordinates are not in increasing order
        gasoil.add_fromtable(krg1, krgcolname="KRG")

    krg2 = pd.DataFrame(
        columns=["SG", "KRG"], data=[[0.15, 0], [0.4, 0.6], [0.6, 0.4], [1, 1]]
    )
    gasoil = GasOil(h=0.1)
    with pytest.raises(ValueError):
        # Should get notified that krg is not monotonous
        gasoil.add_fromtable(krg2, krgcolname="KRG")
    krog2 = pd.DataFrame(
        columns=["SG", "KROG"], data=[[0.15, 1], [0.4, 0.4], [0.6, 0.6], [1, 0]]
    )
    gasoil = GasOil(h=0.1)
    with pytest.raises(ValueError):
        # Should get notified that krog is not monotonous
        gasoil.add_fromtable(krog2, krogcolname="KROG")
    pc2 = pd.DataFrame(
        columns=["SG", "PC"], data=[[0.15, 1], [0.4, 0.4], [0.6, 0.6], [1, 0]]
    )
    gasoil = GasOil(h=0.1)
    with pytest.raises(ValueError):
        # Should get notified that pc is not monotonous
        gasoil.add_fromtable(pc2, pccolname="PC")


def test_wo_fromtable_problems():
    """Test wateroil from tables with problematic data"""
    # With default object:
    df1 = pd.DataFrame(
        columns=["SW", "KRW", "KROW", "PCOW"],
        data=[[0.15, 0, 1, 3], [0.89, 1, 0, 0.1], [1, 1, 0, 0]],
    )
    # With default object:
    wateroil = WaterOil(h=0.1)
    with pytest.raises(ValueError):
        wateroil.add_fromtable(df1)
        # This results in krw and krow overshooting 0 and 1
    # Fix left endpoint:
    wateroil = WaterOil(h=0.1, swl=df1["SW"].min())
    wateroil.add_fromtable(df1)
    # The table is now valid, but we did not preserve the 0.89 point
    check_table(wateroil.table)

    # If we also tell the WaterOil object about sorw, we are guaranteed
    # to have it expclitly included:
    wateroil = WaterOil(h=0.1, swl=df1["SW"].min(), sorw=1 - 0.89)
    wateroil.add_fromtable(df1)
    check_table(wateroil.table)
    # For low enough h, this will however NOT matter.


@pytest.mark.parametrize(
    "dframe, swl, exception, message",
    [
        (
            pd.DataFrame(
                columns=["SW", "KRW", "KROW", "PCOW"],
                data=[[0.15, "fooo", 1, 3], [0.89, 1, 0, 0.1], [1, 1, 0, 0]],
            ),
            0.15,
            ValueError,
            "Failed to parse column KRW",
        ),
        (
            pd.DataFrame(
                columns=["SW", "KRW", "KROW", "PCOW"],
                data=[[0.15, 1.1, 1, 3], [0.89, 1, 0, 0.1], [1, 1, 0, 0]],
            ),
            0.15,
            ValueError,
            "KRW is above 1",
        ),
        (
            pd.DataFrame(
                columns=["SW", "KRW", "KROW", "PCOW"],
                data=[[0.15, -0.1, 1, 3], [0.89, 1, 0, 0.1], [1, 1, 0, 0]],
            ),
            0.15,
            ValueError,
            "KRW is below 0",
        ),
        (
            pd.DataFrame(
                columns=["SW", "KRW", "KROW", "PCOW"],
                data=[[0.15, 0.1, "foo", 3], [0.89, 1, 0, 0.1], [1, 1, 0, 0]],
            ),
            0.15,
            ValueError,
            "Failed to parse column KROW",
        ),
        (
            pd.DataFrame(
                columns=["SW", "KRW", "KROW", "PCOW"],
                data=[[0.15, 0.1, 1.1, 3], [0.89, 1, 0, 0.1], [1, 1, 0, 0]],
            ),
            0.15,
            ValueError,
            "KROW is above 1",
        ),
        (
            pd.DataFrame(
                columns=["SW", "KRW", "KROW", "PCOW"],
                data=[[0.15, 0.1, 1, 3], [0.89, 1, 0, 0.1], [1, 1, -0.001, 0]],
            ),
            0.15,
            ValueError,
            "KROW is below 0",
        ),
        (
            pd.DataFrame(
                columns=["SW", "KRW", "KROW", "PCOW"],
                data=[[0.15, 0.1, 1, 3], [0.89, 1, 0, 0.1], [1, 1, -0.001, 0]],
            ),
            0.10,
            ValueError,
            "Incompatible swl",
        ),
        (
            pd.DataFrame(
                columns=["SW", "KRW", "KROW", "PCOW"],
                data=[[0.15, 0.1, 1, 3], [0.89, 1, 0, 0.1], [1, 1, -0.001, 0]],
            ),
            0,
            ValueError,
            "Incompatible swl",
        ),
        (
            pd.DataFrame(
                columns=["SW", "KRW", "KROW", "PCOW"],
                data=[[0.15, 0.1, 1, 3], [0.89, 1, 0, 0.1], [1, 1, -0.001, 0]],
            ),
            0.16,
            ValueError,
            "Incompatible swl",
        ),
    ],
)
def test_wo_from_table_exceptions(dframe, swl, exception, message):
    wateroil = WaterOil(h=0.1, swl=swl)
    with pytest.raises(exception, match=message):
        wateroil.add_fromtable(dframe)


def test_fromtable_types():
    """Test loading from a table with incorrect types"""

    # This frame is valid, but the type was wrong. This
    # can happen if data is via CSV files, and some other rows
    # ruin the numerical interpretation of a column.
    df1 = pd.DataFrame(
        columns=["SW", "KRW", "KROW", "PC"],
        data=[["0", "0", "1", "0"], ["1", "1", "0", "0"]],
    )
    wateroil = WaterOil(h=0.1)
    wateroil.add_fromtable(
        df1, swcolname="SW", krwcolname="KRW", krowcolname="KROW", pccolname="PC"
    )
    assert "KRW" in wateroil.table.columns
    assert "KROW" in wateroil.table.columns
    assert "PC" in wateroil.table.columns
    check_table(wateroil.table)

    gasoil = GasOil(h=0.1)
    gasoil.add_fromtable(
        df1, sgcolname="SW", krgcolname="KRW", krogcolname="KROW", pccolname="PC"
    )
    assert "KRG" in gasoil.table.columns
    assert "KROG" in gasoil.table.columns
    assert "PC" in gasoil.table.columns
    check_table(gasoil.table)

    # But this should not make sense.
    df2 = pd.DataFrame(
        columns=["SW", "KRW", "KROW", "PC"],
        data=[["0", dict(foo="bar"), "1", "2"], ["1", "1", "0", "0"]],
    )
    wateroil = WaterOil(h=0.1)
    with pytest.raises((ValueError, TypeError)):
        wateroil.add_fromtable(
            df2, swcolname="SW", krwcolname="KRW", krowcolname="KROW", pccolname="PC"
        )
    gasoil = GasOil(h=0.1)
    with pytest.raises((ValueError, TypeError)):
        gasoil.add_fromtable(
            df2, sgcolname="SW", krgcolname="KRW", krogcolname="KROW", pccolname="PC"
        )


@settings(deadline=2000)
@given(st.floats(min_value=1e-5, max_value=1))
def test_wo_fromtable_h(h):
    """Test making curves from tabular data with random stepsize h"""
    df1 = pd.DataFrame(
        columns=["SW", "KRW", "KROW", "PCOW"],
        data=[[0.15, 0, 1, 3], [0.89, 1, 0, 0.1], [1, 1, 0, 0]],
    )
    wateroil = WaterOil(h=h, swl=0.15, sorw=1 - 0.89)
    wateroil.add_fromtable(df1)
    check_table(wateroil.table)
