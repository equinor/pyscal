"""Test the PyscalList module"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyscal import (
    GasOil,
    GasWater,
    PyscalFactory,
    PyscalList,
    SCALrecommendation,
    WaterOil,
    WaterOilGas,
)
from pyscal.utils.testing import sat_table_str_ok

try:
    import ecl2df

    HAVE_ECL2DF = True
except ImportError:
    HAVE_ECL2DF = False


def test_pyscallist_basic():
    """Test that the class acts like a list"""
    p_list = PyscalList()
    assert isinstance(p_list, PyscalList)
    assert not p_list

    p_list.append(None)
    assert not p_list
    p_list.append([])
    assert not p_list

    with pytest.raises(ValueError):
        p_list.append(1)
    with pytest.raises(ValueError):
        p_list.append("foo")

    p_list.append(WaterOil())
    assert len(p_list) == 1
    assert isinstance(p_list[1], WaterOil)
    with pytest.raises(IndexError):
        # pylint: disable=W0104
        p_list[0]
    with pytest.raises(IndexError):
        # pylint: disable=W0104
        p_list[2]
    with pytest.raises(ValueError):
        p_list.append(GasOil())
    assert len(p_list) == 1

    p_list.append(WaterOil())
    assert len(p_list) == 2

    p_list.append([WaterOil()])
    assert len(p_list) == 3

    with pytest.raises(ValueError, match="Not a pyscal object"):
        PyscalList([{}])

    init_from_list = PyscalList([WaterOil(), WaterOil()])
    assert len(init_from_list) == 2

    init_from_itself = PyscalList(PyscalList([WaterOil(), WaterOil()]))
    assert len(init_from_itself) == 2


def test_load_scalrec():
    """Load a SATNUM range from xlsx"""
    testdir = Path(__file__).absolute().parent

    scalrec_data = PyscalFactory.load_relperm_df(
        testdir / "data/scal-pc-input-example.xlsx"
    )

    # Also check that we can read the old excel format
    scalrec_data_legacy_xls = PyscalFactory.load_relperm_df(
        testdir / "data/scal-pc-input-example.xls"
    )
    pd.testing.assert_frame_equal(scalrec_data, scalrec_data_legacy_xls)

    scalrec_list = PyscalFactory.create_scal_recommendation_list(scalrec_data)
    wog_list = scalrec_list.interpolate(-0.3)

    with pytest.raises((AssertionError, ValueError)):
        scalrec_list.interpolate(-1.1)
    assert wog_list.pyscaltype == WaterOilGas

    with pytest.raises(TypeError):
        scalrec_list.build_eclipse_data(family=1)
    with pytest.raises(TypeError):
        scalrec_list.build_eclipse_data(family=2)
    with pytest.raises(TypeError):
        scalrec_list.SWOF()

    scalrec_list.interpolate([-1, 1, 0])
    with pytest.raises((AssertionError, ValueError)):
        scalrec_list.interpolate([-1, 0, 1.1])

    with pytest.raises(
        TypeError, match="Can only interpolate PyscalList of type SCALrecommendation"
    ):
        wog_list.interpolate(-0.5)

    scalrec_list.interpolate(-1, 1)
    scalrec_list.interpolate(-1, [0, -1, 1])

    with pytest.raises(
        ValueError, match="Too few interpolation parameters given for WaterOil"
    ):
        scalrec_list.interpolate([-1, 1])
    with pytest.raises(
        ValueError, match="Too few interpolation parameters given for GasOil"
    ):
        scalrec_list.interpolate(1, [-1, 1])
    with pytest.raises(
        ValueError, match="Too many interpolation parameters given for WaterOil"
    ):
        scalrec_list.interpolate([-1, 1, 0, 0])
    with pytest.raises(
        ValueError, match="Too many interpolation parameters given for GasOil"
    ):
        scalrec_list.interpolate(1, [-1, 1, 0, 0])

    # Test slicing the scalrec to base, this is relevant for API usage.
    base_data = scalrec_data[scalrec_data["CASE"] == "base"].drop("CASE", axis=1)
    PyscalFactory.load_relperm_df(base_data)  # Ensure no errors.
    pyscal_list = PyscalFactory.create_pyscal_list(base_data)
    assert "LET" in pyscal_list.build_eclipse_data(family=1)


def test_df():
    """Test dataframe dumps"""
    testdir = Path(__file__).absolute().parent

    scalrec_data = PyscalFactory.load_relperm_df(
        testdir / "data/scal-pc-input-example.xlsx"
    )
    scalrec_list = PyscalFactory.create_scal_recommendation_list(scalrec_data)
    wog_list = scalrec_list.interpolate(-0.3)

    # Test dataframe dumps:
    dframe = scalrec_list.df()
    assert "SG" in dframe
    assert "KRG" in dframe
    assert "KROG" in dframe
    assert "PCOW" in dframe
    if "PCOG" in dframe:
        # Allow PCOG to be included later.
        assert dframe["PCOG"].sum() == 0
    assert "KRW" in dframe
    assert "KROW" in dframe
    assert "SATNUM" in dframe
    assert dframe["SATNUM"].min() == 1
    assert len(dframe["SATNUM"].unique()) == len(scalrec_list)
    assert set(dframe["CASE"]) == set(["pess", "base", "opt"])
    assert dframe["SATNUM"].max() == len(scalrec_list)
    if HAVE_ECL2DF:
        # Test using ecl2df to do the include file printing. First we need to
        # massage the dataframe into what ecl2df can handle:
        base_df_swof = (
            dframe.set_index("CASE")
            .loc["base"][["SW", "KRW", "KROW", "PCOW", "SATNUM"]]
            .assign(KEYWORD="SWOF")
            .dropna()
            .reset_index(drop=True)
        )
        ecl_inc = ecl2df.satfunc.df2ecl(base_df_swof)
        dframe_from_inc = ecl2df.satfunc.df(ecl_inc)
        pd.testing.assert_frame_equal(base_df_swof, dframe_from_inc)

        # Test also SGOF
        base_df_sgof = (
            dframe.set_index("CASE")
            .loc["base"][["SG", "KRG", "KROG", "SATNUM"]]
            .assign(KEYWORD="SGOF", PCOG=0.0)
            .dropna()
            .reset_index(drop=True)
        )
        ecl_inc = ecl2df.satfunc.df2ecl(base_df_sgof)
        dframe_from_inc = ecl2df.satfunc.df(ecl_inc)
        pd.testing.assert_frame_equal(base_df_sgof, dframe_from_inc, check_like=True)

    # WaterOilGasList:
    dframe = wog_list.df()
    assert "SG" in dframe
    assert "KRG" in dframe
    assert "KROG" in dframe
    assert "PCOW" in dframe
    assert "PCOG" in dframe  # This gets included through interpolation
    assert "KRW" in dframe
    assert "KROW" in dframe
    assert "SATNUM" in dframe
    assert dframe["SATNUM"].min() == 1
    assert len(dframe["SATNUM"].unique()) == len(wog_list)
    assert "CASE" not in dframe
    assert dframe["SATNUM"].max() == len(wog_list)

    # WaterOil list
    input_dframe = pd.DataFrame(columns=["SATNUM", "Nw", "Now"], data=[[1, 2, 2]])
    relperm_data = PyscalFactory.load_relperm_df(input_dframe)
    wo_list = PyscalFactory.create_pyscal_list(relperm_data, h=0.2)
    dframe = wo_list.df()
    assert "SW" in dframe
    assert "KRW" in dframe
    assert "KROW" in dframe
    assert "PCOW" not in dframe  # to be interpreted as zero
    assert "SATNUM" in dframe
    assert len(dframe.columns) == 4
    assert not dframe.empty

    # GasOil list
    input_dframe = pd.DataFrame(columns=["SATNUM", "Ng", "Nog"], data=[[1, 2, 2]])
    relperm_data = PyscalFactory.load_relperm_df(input_dframe)
    go_list = PyscalFactory.create_pyscal_list(relperm_data, h=0.2)
    dframe = go_list.df()
    assert "SG" in dframe
    assert "KRG" in dframe
    assert "KROG" in dframe
    assert "PCOG" not in dframe  # to be interpreted as zero
    assert "SATNUM" in dframe
    assert len(dframe.columns) == 4
    assert not dframe.empty

    gasoil_family1 = go_list.build_eclipse_data(family=1)
    assert "SGOF" in gasoil_family1

    with pytest.raises(ValueError, match="SLGOF not meaningful for GasOil"):
        go_list.build_eclipse_data(family=1, slgof=True)


def test_load_scalrec_tags():
    """Test tag handling for a SCAL recommendation with SATNUM range"""
    testdir = Path(__file__).absolute().parent

    scalrec_data = PyscalFactory.load_relperm_df(
        testdir / "data/scal-pc-input-example.xlsx"
    )
    scalrec_list = PyscalFactory.create_scal_recommendation_list(scalrec_data)

    wog_list = scalrec_list.interpolate(-1)

    swof = wog_list.SWOF()
    assert swof.count("SCAL recommendation interpolation to -1") == 3
    assert swof.count("SATNUM 1") == 1
    assert swof.count("SATNUM") == 3

    sof3 = wog_list.SOF3()
    assert sof3.count("SCAL recommendation interpolation to -1") == 3
    assert sof3.count("SATNUM 1") == 1
    assert sof3.count("SATNUM") == 3

    assert (
        scalrec_list.interpolate(1)
        .SWOF()
        .count("SCAL recommendation interpolation to 1\n")
        == 3
    )
    assert (
        scalrec_list.interpolate(0)
        .SWOF()
        .count("SCAL recommendation interpolation to 0\n")
        == 3
    )
    assert (
        scalrec_list.interpolate(-0.444)
        .SWOF()
        .count("SCAL recommendation interpolation to -0.444\n")
        == 3
    )

    # Give each row in the SCAL recommendation its own tag (low, base, high
    # explicitly in each tag, someone will do that)
    scalrec_data["TAG"] = [
        "SAT1 low",
        "SAT1 base",
        "SAT1 high",
        "SAT2 pess",
        "SAT2 base",
        "SAT2 opt",
        "SAT3 pessimistic",
        "SAT3 base case",
        "SAT3 optimistic",
    ]
    scalrec_list2 = PyscalFactory.create_scal_recommendation_list(scalrec_data)
    swof = scalrec_list2.interpolate(-0.5, h=0.2).SWOF()
    assert swof.count("SCAL recommendation") == 3
    for tag in scalrec_data["TAG"]:
        assert swof.count(tag) == 1


def test_deprecated_dump_to_file(tmpdir):
    """Test dumping Eclipse include data to file. This
    functionality is deprecated in pyscallist"""
    testdir = Path(__file__).absolute().parent

    relperm_data = PyscalFactory.load_relperm_df(
        testdir / "data/relperm-input-example.xlsx"
    )
    pyscal_list = PyscalFactory.create_pyscal_list(relperm_data)

    fam1 = pyscal_list.build_eclipse_data(family=1)
    sat_table_str_ok(fam1)

    fam2 = pyscal_list.build_eclipse_data(family=2)
    sat_table_str_ok(fam2)

    # Empty PyscalLists should return empty strings:
    assert PyscalList().build_eclipse_data(family=1) == ""
    assert PyscalList().build_eclipse_data(family=2) == ""

    tmpdir.chdir()
    with pytest.warns(DeprecationWarning):
        pyscal_list.dump_family_1(filename="outputdir/output-fam1.inc")
    assert "SWOF" in Path("outputdir/output-fam1.inc").read_text(encoding="utf8")

    with pytest.warns(DeprecationWarning):
        pyscal_list.dump_family_2(filename="anotherdir/output-fam2.inc")
    assert "SOF3" in Path("anotherdir/output-fam2.inc").read_text(encoding="utf8")

    with pytest.warns(DeprecationWarning):
        pyscal_list.SWOF(write_to_filename="swof.inc")
    assert "SWOF" in Path("swof.inc").read_text(encoding="utf8")


def test_capillary_pressure():
    """Test that we recognize capillary pressure parametrizations"""
    dframe = pd.DataFrame(
        columns=[
            "SATNUM",
            "nw",
            "now",
            "swl",
            "a",
            "b",
            "PORO_reF",
            "PERM_ref",
            "drho",
        ],
        data=[[1, 1, 1, 0.05, 3.6, -3.5, 0.25, 15, 150]],
    )
    pyscal_list = PyscalFactory.create_pyscal_list(
        PyscalFactory.load_relperm_df(dframe)
    )
    swof = pyscal_list.build_eclipse_data(family=1)
    assert "Simplified J-function" in swof
    assert "petrophysical" not in swof

    dframe = pd.DataFrame(
        columns=[
            "SATNUM",
            "nw",
            "now",
            "swl",
            "a_petro",
            "b_petro",
            "PORO_reF",
            "PERM_ref",
            "drho",
        ],
        data=[[1, 1, 1, 0.05, 3.6, -3.5, 0.25, 15, 150]],
    )
    pyscal_list = PyscalFactory.create_pyscal_list(
        PyscalFactory.load_relperm_df(dframe)
    )
    swof = pyscal_list.build_eclipse_data(family=1)
    assert "Simplified J-function" in swof
    assert "petrophysical" in swof

    dframe = pd.DataFrame(
        columns=[
            "SATNUM",
            "nw",
            "now",
            "swl",
            "a",
            "b",
            "PORO",
            "PERM",
            "sigma_COStau",
        ],
        data=[[1, 1, 1, 0.05, 3.6, -3.5, 0.25, 15, 30]],
    )
    pyscal_list = PyscalFactory.create_pyscal_list(
        PyscalFactory.load_relperm_df(dframe)
    )
    swof = pyscal_list.build_eclipse_data(family=1)
    assert "normalized J-function" in swof
    assert "sigma_costau" in swof
    assert "petrophysical" not in swof


def test_swl_from_height():
    """Test that can initialize swl from capillary pressure height"""
    df_columns = [
        "SATNUM",
        "nw",
        "now",
        "swl",
        "swlheight",
        "swirr",
        "a",
        "b",
        "PORO_reF",
        "PERM_ref",
        "drho",
    ]
    dframe = pd.DataFrame(
        columns=df_columns,
        data=[[1, 1, 1, np.nan, 300, 0.00, 3.6, -3.5, 0.25, 15, 150]],
    )
    pyscal_list = PyscalFactory.create_pyscal_list(
        PyscalFactory.load_relperm_df(dframe)
    )

    # Mix swlheight init and direct swl-init:
    assert np.isclose(pyscal_list[1].swl, 0.157461)
    dframe = pd.DataFrame(
        columns=df_columns,
        data=[
            [1, 1, 1, np.nan, 300, 0.00, 3.6, -3.5, 0.25, 15, 150],
            [2, 1, 1, 0.3, np.nan, 0.00, 3.6, -3.5, 0.25, 15, 150],
        ],
    )
    pyscal_list = PyscalFactory.create_pyscal_list(
        PyscalFactory.load_relperm_df(dframe)
    )
    assert np.isclose(pyscal_list[1].swl, 0.157461)
    assert np.isclose(pyscal_list[2].swl, 0.3)

    # Ambiguous, swl and swlheight both supplied:
    dframe = pd.DataFrame(
        columns=df_columns,
        data=[[1, 1, 1, 0.3, 300, 0.00, 3.6, -3.5, 0.25, 15, 150]],
    )
    with pytest.raises(ValueError):
        PyscalFactory.create_pyscal_list(PyscalFactory.load_relperm_df(dframe))

    # WaterOilGas (gasoil is also dependant on the computed swl)
    df_wog_columns = [
        "SATNUM",
        "nw",
        "now",
        "ng",
        "nog",
        "swlheight",
        "swirr",
        "a",
        "b",
        "PORO_reF",
        "PERM_ref",
        "drho",
    ]
    dframe = pd.DataFrame(
        columns=df_wog_columns,
        data=[[1, 1, 1, 2, 2, 300, 0.00, 3.6, -3.5, 0.25, 15, 150]],
    )
    pyscal_list = PyscalFactory.create_pyscal_list(
        PyscalFactory.load_relperm_df(dframe)
    )
    assert np.isclose(pyscal_list[1].wateroil.swl, 0.157461)
    assert np.isclose(pyscal_list[1].gasoil.swl, 0.157461)

    # Test for GasWater:
    df_gw_columns = [
        "SATNUM",
        "nw",
        "ng",
        "swlheight",
        "swirr",
        "a",
        "b",
        "PORO_REF",
        "PERM_REF",
        "drho",
    ]
    dframe = pd.DataFrame(
        columns=df_gw_columns,
        data=[[1, 2, 3, 300, 0.00, 3.6, -3.5, 0.25, 15, 150]],
    )
    pyscal_list = PyscalFactory.create_pyscal_list(
        PyscalFactory.load_relperm_df(dframe)
    )
    assert np.isclose(pyscal_list[1].wateroil.swl, 0.157461)
    assert np.isclose(pyscal_list[1].swl, 0.157461)


def test_twophase_scal_wateroil():
    """Test interpolation for a two-phase setup water-oil"""
    # Triggers bug in pyscal <= 0.5.0
    dframe = pd.DataFrame(
        columns=["SATNUM", "CASE", "NW", "NOW", "TAG"],
        data=[
            [1, "pess", 1, 1, "thetag"],
            [1, "base", 2, 2, "thetag"],
            [1, "opt", 3, 3, "thetag"],
        ],
    )
    pyscal_list = PyscalFactory.create_scal_recommendation_list(
        PyscalFactory.load_relperm_df(dframe)
    )
    pyscal_list = pyscal_list.interpolate(-0.5)

    # This works, but provides an error message for Gas.
    # The correct usage would be to call .SWOF(), but the pyscal
    # client calls 1 on two-phase problems.
    swof = pyscal_list.build_eclipse_data(family=1)
    assert "thetag" in swof
    assert "SWOF" in swof
    assert "SGOF" not in swof


def test_twophase_scal_gasoil():
    """Test interpolation for a two-phase setup gas-oil"""
    dframe = pd.DataFrame(
        columns=["SATNUM", "CASE", "NG", "NOG", "TAG"],
        data=[
            [1, "pess", 1, 1, "thetag"],
            [1, "base", 2, 2, "thetag"],
            [1, "opt", 3, 3, "thetag"],
        ],
    )
    pyscal_list = PyscalFactory.create_scal_recommendation_list(
        PyscalFactory.load_relperm_df(dframe)
    )
    pyscal_list = pyscal_list.interpolate(-0.5)

    sgof = pyscal_list.build_eclipse_data(family=1)
    assert "thetag" in sgof
    assert "SGOF" in sgof
    assert "SWOF" not in sgof


def test_gaswater():
    """Test list of gas-water objects"""
    dframe = pd.DataFrame(
        columns=["SATNUM", "NW", "NG", "TAG"],
        data=[[1, 2, 2, "thetag"], [2, 3, 3, "othertag"]],
    )
    pyscal_list = PyscalFactory.create_pyscal_list(
        PyscalFactory.load_relperm_df(dframe), h=0.1
    )
    assert pyscal_list.pyscaltype == GasWater
    dump = pyscal_list.build_eclipse_data(family=2)
    assert "SATNUM 2" in dump
    assert "SATNUM 1" in dump
    assert "SATNUM 3" not in dump
    assert "SWFN" in dump
    assert "SGFN" in dump
    assert "othertag" in dump
    assert "thetag" in dump

    with pytest.raises(ValueError, match="Family 1 output not possible for GasWater"):
        pyscal_list.build_eclipse_data(family=1)

    sgfn = pyscal_list.SGFN()
    assert "SGFN" in sgfn
    assert "krw = krg" in sgfn
    swfn = pyscal_list.SWFN()
    assert "SWFN" in swfn
    assert "krw = krg" in swfn

    with pytest.raises(
        AttributeError, match="'GasWater' object has no attribute 'SWOF'"
    ):
        pyscal_list.SWOF()


def test_gaswater_scal():
    """Test list of gas-water objects in scal recommendation"""
    dframe = pd.DataFrame(
        columns=["SATNUM", "CASE", "NW", "NG", "TAG"],
        data=[
            [1, "pess", 2, 2, "thetag"],
            [1, "base", 3, 3, "thetag"],
            [1, "opt", 4, 4, "thetag"],
        ],
    )
    pyscal_list = PyscalFactory.create_scal_recommendation_list(
        PyscalFactory.load_relperm_df(dframe), h=0.1
    )
    assert pyscal_list.pyscaltype == SCALrecommendation

    dump = pyscal_list.interpolate(-1).build_eclipse_data(family=2)
    assert "SWFN" in dump
    assert "SGFN" in dump


def test_explicit_df():
    """Test some dataframes, check the error messages given"""

    dframe = pd.DataFrame(columns=["satnum"], data=[[1], [2]])
    with pytest.raises(ValueError):
        # SATNUM column must be upper case, or should we allow lowercase??
        relperm_data = PyscalFactory.load_relperm_df(dframe)

    dframe = pd.DataFrame(columns=["SATNUM"], data=[[0], [1]])
    with pytest.raises(ValueError):
        # SATNUM must start at 1.
        relperm_data = PyscalFactory.load_relperm_df(dframe)

    dframe = pd.DataFrame(columns=["SATNUM"], data=[[1], ["foo"]])
    with pytest.raises(ValueError):
        # SATNUM must contain only integers
        relperm_data = PyscalFactory.load_relperm_df(dframe)

    dframe = pd.DataFrame(
        columns=["SATNUM", "nw", "now"], data=[[1.01, 1, 1], [2.01, 1, 1]]
    )
    # This one will pass, as these can be converted to ints
    relperm_data = PyscalFactory.load_relperm_df(dframe)

    dframe = pd.DataFrame(
        columns=["SATNUM", "nw", "now"], data=[[1.01, 1, 1], [1.3, 1, 1]]
    )
    with pytest.raises(ValueError):
        # complains about non-uniqueness in SATNUM
        relperm_data = PyscalFactory.load_relperm_df(dframe)

    dframe = pd.DataFrame(columns=["SATNUM"], data=[[1], [2]])
    with pytest.raises(ValueError):
        # Not enough data
        relperm_data = PyscalFactory.load_relperm_df(dframe)

    # Minimal amount of data:
    dframe = pd.DataFrame(columns=["SATNUM", "Nw", "Now"], data=[[1, 2, 2]])
    relperm_data = PyscalFactory.load_relperm_df(dframe)
    p_list = PyscalFactory.create_pyscal_list(relperm_data, h=0.2)
    p_list.build_eclipse_data(family=1)
    with pytest.raises(ValueError):
        p_list.build_eclipse_data(family=2)

    # Case insensitive for parameters
    dframe = pd.DataFrame(columns=["SATNUM", "nw", "now"], data=[[1, 2, 2]])
    relperm_data = PyscalFactory.load_relperm_df(dframe)
    p_list = PyscalFactory.create_pyscal_list(relperm_data, h=0.2)
    p_list.build_eclipse_data(family=1)

    # Minimal wateroilgas
    dframe = pd.DataFrame(
        columns=["SATNUM", "Nw", "Now", "ng", "nOG"], data=[[1, 2, 2, 2, 2]]
    )
    relperm_data = PyscalFactory.load_relperm_df(dframe)
    p_list = PyscalFactory.create_pyscal_list(relperm_data, h=0.2)
    relperm_str = p_list.build_eclipse_data(family=1)
    assert "SWOF" in relperm_str
    assert "SGOF" in relperm_str
    assert "SLGOF" not in relperm_str
    assert "SOF3" not in relperm_str

    # Minimal wateroilgas with pc
    dframe = pd.DataFrame(
        columns=[
            "SATNUM",
            "swl",
            "Nw",
            "Now",
            "ng",
            "nOG",
            "a",
            "b",
            "poro_ref",
            "perm_ref",
            "drho",
        ],
        data=[[1, 0.1, 2, 2, 2, 2, 1.5, -0.5, 0.1, 100, 300]],
    )
    relperm_data = PyscalFactory.load_relperm_df(dframe)
    p_list = PyscalFactory.create_pyscal_list(relperm_data, h=0.2)
    relperm_str = p_list.build_eclipse_data(family=1)
    assert "SWOF" in relperm_str
    assert "Simplified" in relperm_str  # Bad practice, testing for stuff in comments
    assert "a=1.5" in relperm_str


def test_comments_df():
    """Test that we support a tag column in the dataframe."""
    dframe = pd.DataFrame(
        columns=["SATNUM", "tag", "Nw", "Now", "ng", "nOG"],
        data=[[1, "thisisacomment", 2, 2, 2, 2]],
    )
    relperm_data = PyscalFactory.load_relperm_df(dframe)
    p_list = PyscalFactory.create_pyscal_list(relperm_data, h=0.2)
    relperm_str = p_list.build_eclipse_data(family=1)
    assert "thisisacomment" in relperm_str

    # tag vs comment in dataframe should not matter.
    dframe = pd.DataFrame(
        columns=["SATNUM", "comment", "Nw", "Now", "ng", "nOG"],
        data=[[1, "thisisacomment", 2, 2, 2, 2]],
    )
    relperm_data = PyscalFactory.load_relperm_df(dframe)
    p_list = PyscalFactory.create_pyscal_list(relperm_data, h=0.2)
    relperm_str = p_list.build_eclipse_data(family=1)
    assert relperm_str.count("thisisacomment") == 2

    # Check case insensitiveness:
    dframe = pd.DataFrame(
        columns=["SAtnUM", "coMMent", "Nw", "Now", "ng", "nOG"],
        data=[[1, "thisisacomment", 2, 2, 2, 2]],
    )
    relperm_data = PyscalFactory.load_relperm_df(dframe)
    p_list = PyscalFactory.create_pyscal_list(relperm_data, h=0.2)
    assert p_list.build_eclipse_data(family=1).count("thisisacomment") == 2

    # UTF-8 stuff:
    dframe = pd.DataFrame(
        columns=["SATNUM", "TAG", "Nw", "Now", "Ng", "Nog"],
        data=[[1, "æøå", 2, 2, 2, 2]],
    )
    relperm_data = PyscalFactory.load_relperm_df(dframe)
    p_list = PyscalFactory.create_pyscal_list(relperm_data, h=0.2)
    assert p_list.build_eclipse_data(family=1).count("æøå") == 2


def test_error_messages_pr_satnum():
    """When errors are somewhere in a big dataframe, we should
    provide hints to the user for where to look"""

    # A string instead of a number in SATNUM 2
    dframe = pd.DataFrame(
        columns=["SATNUM", "nw", "now"], data=[[1, 1, 1], [2, "foo", 1]]
    )
    with pytest.raises(ValueError, match="SATNUM 2"):
        PyscalFactory.create_pyscal_list(dframe, h=0.2)

    dframe = pd.DataFrame(
        columns=["SATNUM", "nw", "now"], data=[[1, 1, 1], [2, np.nan, 1]]
    )
    with pytest.raises(ValueError, match="SATNUM 2"):
        PyscalFactory.create_pyscal_list(dframe, h=0.2)

    # Mixed up order:
    dframe = pd.DataFrame(
        columns=["SATNUM", "nw", "now"], data=[[2, np.nan, 1], [1, 1, 1]]
    )
    with pytest.raises(ValueError, match="SATNUM 2"):
        PyscalFactory.create_pyscal_list(dframe, h=0.2)

    # Gasoil list
    dframe = pd.DataFrame(
        columns=["SATNUM", "ng", "nog"], data=[[1, 1, 1], [2, np.nan, 1]]
    )
    with pytest.raises(ValueError, match="SATNUM 2"):
        PyscalFactory.create_pyscal_list(dframe, h=0.2)

    # Gaswater list:
    dframe = pd.DataFrame(
        columns=["SATNUM", "ng", "nw"], data=[[1, 1, 1], [2, np.nan, 1]]
    )
    with pytest.raises(ValueError, match="SATNUM 2"):
        PyscalFactory.create_pyscal_list(dframe, h=0.2)

    # Wateroilgas list:
    dframe = pd.DataFrame(
        columns=["SATNUM", "nw", "now", "ng", "nog"],
        data=[[1, 1, 1, 1, 1], [2, np.nan, 1, 2, 3]],
    )
    with pytest.raises(ValueError, match="SATNUM 2"):
        PyscalFactory.create_pyscal_list(dframe, h=0.2)

    # SCAL rec list:
    dframe = pd.DataFrame(
        columns=["SATNUM", "CASE", "nw", "now", "ng", "nog"],
        data=[
            [1, "low", 1, 1, 1, 1],
            [1, "base", 1, 1, 1, 1],
            [1, "high", 1, 1, 1, 1],
            [2, "low", 1, 1, 2, 3],
            [2, "base", 2, 1, 2, 3],
            [2, "high", 3, 1, 2, 3],
        ],
    )
    # Perturb all entries in the dataframe:
    for rowidx in list(range(0, 6)):
        for colidx in list(range(2, 6)):
            dframe_perturbed = dframe.copy()
            dframe_perturbed.iloc[rowidx, colidx] = np.nan
            satnum = dframe.iloc[rowidx, 0]
            case = dframe.iloc[rowidx, 1]
            # The error should hint both to SATNUM and to low/base/high
            with pytest.raises(ValueError, match=f"SATNUM {satnum}"):
                PyscalFactory.create_scal_recommendation_list(dframe_perturbed, h=0.2)
            with pytest.raises(ValueError, match=f"Problem with {case}"):
                PyscalFactory.create_scal_recommendation_list(dframe_perturbed, h=0.2)


def test_fast():
    """Test fast mode for SCALrecmmendation"""
    testdir = Path(__file__).absolute().parent

    scalrec_data = PyscalFactory.load_relperm_df(
        testdir / "data/scal-pc-input-example.xlsx"
    )
    scalrec_list_fast = PyscalFactory.create_scal_recommendation_list(
        scalrec_data, fast=True
    )

    for item in scalrec_list_fast:
        assert item.fast
        assert item.low.fast
        assert item.base.fast
        assert item.high.fast

    wog_list_fast = scalrec_list_fast.interpolate(-0.5)
    for item in wog_list_fast:
        assert item.fast

    # WaterOilGas list
    dframe = pd.DataFrame(
        columns=["SATNUM", "nw", "now", "ng", "nog"],
        data=[
            [1, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [3, 2, 2, 2, 2],
        ],
    )
    relperm_data = PyscalFactory.load_relperm_df(dframe)
    p_list_fast = PyscalFactory.create_pyscal_list(relperm_data, h=0.2, fast=True)
    for item in p_list_fast:
        assert item.fast

    # GasOil list
    input_dframe = dframe[["SATNUM", "ng", "nog"]].copy()
    relperm_data = PyscalFactory.load_relperm_df(input_dframe)
    p_list_fast = PyscalFactory.create_pyscal_list(relperm_data, h=0.2, fast=True)
    for item in p_list_fast:
        assert item.fast

    # WaterOil list
    input_dframe = dframe[["SATNUM", "nw", "now"]].copy()
    relperm_data = PyscalFactory.load_relperm_df(input_dframe)
    p_list_fast = PyscalFactory.create_pyscal_list(relperm_data, h=0.2, fast=True)
    for item in p_list_fast:
        assert item.fast

    # GasWater list
    input_dframe = dframe[["SATNUM", "nw", "ng"]].copy()
    relperm_data = PyscalFactory.load_relperm_df(input_dframe)
    p_list_fast = PyscalFactory.create_pyscal_list(relperm_data, h=0.2, fast=True)
    for item in p_list_fast:
        assert item.fast

    # Testing with "fast" column in dataframe
    # Currently the choice is to only implement fast mode
    # as a global option. This column should do nothing now.
    # One could imagine it implemented
    # for individual SATNUM regions at a later stage
    dframe = pd.DataFrame(
        columns=["SATNUM", "nw", "now", "ng", "nog", "fast"],
        data=[
            [1, 2, 2, 2, 2, True],
            [2, 2, 2, 2, 2, False],
            [3, 2, 2, 2, 2, True],
        ],
    )
    relperm_data = PyscalFactory.load_relperm_df(dframe)
    assert "fast" in relperm_data
    p_list = PyscalFactory.create_pyscal_list(relperm_data, h=0.2)
    for item in p_list:
        assert not item.fast
