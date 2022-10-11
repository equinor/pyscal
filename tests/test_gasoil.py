"""Test module for GasOil objects"""
import io
import sys

import hypothesis.strategies as st
import matplotlib
import matplotlib.pyplot
import numpy as np
import pandas as pd
import pytest
from hypothesis import given

from pyscal import GasOil
from pyscal.constants import MAX_EXPONENT, SWINTEGERS
from pyscal.utils.relperm import truncate_zeroness
from pyscal.utils.testing import (
    check_linear_sections,
    check_table,
    float_df_checker,
    sat_table_str_ok,
)


def test_gasoil_init():
    """Test features in the constructor"""
    gasoil = GasOil()
    assert isinstance(gasoil, GasOil)
    assert gasoil.swirr == 0.0
    assert gasoil.swl == 0.0
    assert gasoil.krgendanchor == ""  # Because sorg is zero

    gasoil = GasOil(swl=0.1)
    assert gasoil.swirr == 0.0
    assert gasoil.swl == 0.1

    gasoil = GasOil(swirr=0.1)
    assert gasoil.swirr == 0.1
    assert gasoil.swl == 0.1  # This one is zero by default, but will follow swirr.
    assert gasoil.sorg == 0.0
    assert gasoil.sgcr == 0.0

    gasoil = GasOil(tag="foobar")
    assert gasoil.tag == "foobar"

    # This will print a warning, but will be the same as ""
    gasoil = GasOil(krgendanchor="bogus")
    assert isinstance(gasoil, GasOil)
    assert gasoil.krgendanchor == ""

    gasoil = GasOil(krgendanchor=None)
    assert isinstance(gasoil, GasOil)
    assert gasoil.krgendanchor == ""

    # Test with h=1
    gasoil = GasOil(h=1)
    gasoil.add_corey_gas()
    gasoil.add_corey_oil()
    assert np.isclose(gasoil.crosspoint(), 0.5)
    assert len(gasoil.table) == 2

    gasoil = GasOil(swl=0.1, h=1)
    gasoil.add_corey_gas()
    gasoil.add_corey_oil()
    assert len(gasoil.table) == 2
    assert np.isclose(gasoil.crosspoint(), 0.45)
    assert np.isclose(gasoil.table["SG"].min(), 0)
    assert np.isclose(gasoil.table["SG"].max(), 0.9)

    # Test too small h:
    gasoil = GasOil(swl=0.1, h=0.00000000000000000001)
    # (a warning is printed that h is truncated)
    assert gasoil.h == 1 / SWINTEGERS


def test_conserve_sgcr(mocker):
    """sgcr in the table can be hard to conserve"""
    low_sgcrvalue = 0.00001  # 1/10 of 1/SWINTEGERS
    # Verify the default behaviour of truncation:
    assert truncate_zeroness(low_sgcrvalue) == 0
    mocker.patch(
        # Remove the effect of zeroness truncation in order to provoke
        # the safeguarding measures for sgcr.
        "pyscal.gasoil.truncate_zeroness",
        return_value=low_sgcrvalue,
    )
    gasoil = GasOil(swl=0, sgcr=low_sgcrvalue, h=0.2)
    assert gasoil.sgcr == low_sgcrvalue  # To check that we have not truncated it away
    assert gasoil.table["SG"][0] == 0
    assert gasoil.table["SG"][1] == low_sgcrvalue

    gasoil.add_corey_gas()
    gasoil.add_corey_oil()
    assert gasoil.selfcheck()
    assert "SGOF" in gasoil.SGOF()


def test_errors():
    """Test some error situations for the constructor"""
    with pytest.raises(ValueError, match="No saturation range left"):
        GasOil(swl=0.3, sorg=0.8)
    with pytest.raises(ValueError, match="No saturation range left"):
        GasOil(swl=0.3, sgro=0.88, sgcr=0.88)
    with pytest.raises(ValueError, match="No saturation range left"):
        GasOil(swl=0.3, sgcr=0.88)

    # This only raises error if krgendanchor is non-default:
    GasOil(swl=0.3, sgcr=0.4, sorg=0.4, krgendanchor="")
    with pytest.raises(ValueError, match="No saturation range left"):
        GasOil(swl=0.3, sgcr=0.4, sorg=0.4)

    with pytest.raises(ValueError, match="No saturation range left"):
        GasOil(swl=0.3, sgcr=0.8, sorg=0.4, krgendanchor="")

    with pytest.raises(ValueError, match="No saturation range left"):
        GasOil(swl=0.3, sgcr=0.88, krgendanchor="")
    with pytest.raises(ValueError, match="sgro must be zero or equal to sgcr"):
        GasOil(swl=0.3, sgcr=0.1, sgro=0.2)

    with pytest.raises(ValueError, match="sgro must be zero or equal to sgcr"):
        GasOil(swl=0.4, sorg=0.4, sgro=0.4)


@pytest.mark.skipif(sys.platform == "win32", reason="Plotting not stable on windows")
def test_plotting(mocker):
    """Test that plotting code pass through (nothing displayed)"""
    mocker.patch("matplotlib.pyplot.show", return_value=None)
    gasoil = GasOil(swl=0.1, h=0.1)
    gasoil.add_corey_gas()
    gasoil.add_corey_oil()
    gasoil.plotkrgkrog(mpl_ax=matplotlib.pyplot.subplots()[1])
    gasoil.plotkrgkrog(mpl_ax=None)
    gasoil.plotkrgkrog(logyscale=True, mpl_ax=None)


@given(st.text())
def test_gasoil_tag(tag):
    """Test tagging of GasOil objects,
    that we are not able to produce something that
    can crash Eclipse"""
    gasoil = GasOil(h=0.5, tag=tag)
    gasoil.add_corey_gas()
    gasoil.add_corey_oil()
    sat_table_str_ok(gasoil.SGOF())
    sat_table_str_ok(gasoil.SGFN())


@given(
    st.floats(min_value=0, max_value=0.15),  # swl
    st.floats(min_value=0, max_value=0.3),  # sgcr
    st.floats(min_value=0, max_value=0.05),  # sorg
    st.floats(min_value=0.0001, max_value=0.2),  # h
    st.text(),
)
def test_gasoil_normalization(swl, sgcr, sorg, h, tag):
    """Check that normalization (sgn and son) is correct
    for all possible saturation endpoints"""
    gasoil = GasOil(
        swirr=0.0, swl=swl, sgcr=sgcr, sorg=sorg, h=h, krgendanchor="sorg", tag=tag
    )
    assert not gasoil.table.empty
    assert not gasoil.table.isnull().values.any()

    # Check that son is 1 at sg=0
    assert float_df_checker(gasoil.table, "SG", 0, "SON", 1)

    # Check that son is 0 at sorg with this krgendanchor.
    # It is important to use the endpoints from the returned gasoil
    # object, as they can be modified in case they are too close to zero.
    assert float_df_checker(gasoil.table, "SG", 1 - gasoil.sorg - gasoil.swl, "SON", 0)

    # Check that sgn is 0 at sgcr
    assert float_df_checker(gasoil.table, "SG", gasoil.sgcr, "SGN", 0)

    # Check that sgn is 1 at sorg
    assert float_df_checker(gasoil.table, "SG", 1 - gasoil.sorg - gasoil.swl, "SGN", 1)

    # Redo with different krgendanchor
    gasoil = GasOil(
        swirr=0.0, swl=swl, sgcr=sgcr, sorg=sorg, h=h, krgendanchor="", tag=tag
    )
    assert float_df_checker(gasoil.table, "SG", 1 - gasoil.swl, "SGN", 1)
    assert float_df_checker(gasoil.table, "SG", gasoil.sgcr, "SGN", 0)


@given(
    st.floats(min_value=0, max_value=0.3),  # swl
    st.floats(min_value=0, max_value=0.3),  # sgcr
    st.floats(min_value=0, max_value=0.4),  # sorg (sgn collapses when >0.4)
    st.booleans(),  # sgrononzero
    st.floats(min_value=0.1, max_value=1),  # kroend
    st.floats(min_value=0.1, max_value=1),  # kromax
    st.floats(min_value=0.1, max_value=1),  # krgend
    st.floats(min_value=0.2, max_value=1),  # krgmax
    st.floats(min_value=0.001, max_value=0.5),  # h
    st.booleans(),  # fast mode
)
def test_gasoil_krendmax(
    swl, sgcr, sorg, sgrononzero, kroend, kromax, krgend, krgmax, h, fast
):
    """Test that relperm curves are valid in all numerical corner cases."""
    if sgrononzero:
        sgro = sgcr
    else:
        sgro = 0
    try:
        assert 1 - sorg - swl - sgcr > 1 / SWINTEGERS, "No saturation range left"
        gasoil = GasOil(
            swl=swl, sgcr=sgcr, sorg=sorg, sgro=sgro, h=h, tag="", fast=fast
        )
    except AssertionError:
        # We end here when hypothesis sets up impossible/non-interesting scenarios
        return
    krgend = min(krgend, krgmax)
    kroend = min(kroend, kromax)
    gasoil.add_corey_oil(kroend=kroend, kromax=kromax)
    gasoil.add_corey_gas(krgend=krgend, krgmax=krgmax)
    check_table(gasoil.table)
    check_linear_sections(gasoil)
    assert gasoil.selfcheck()
    sat_table_str_ok(gasoil.SGOF())
    check_endpoints(gasoil, krgend, krgmax, kroend, kromax)
    assert 0 < gasoil.crosspoint() < 1

    # Redo with krgendanchor not defaulted
    gasoil = GasOil(swl=swl, sgcr=sgcr, sorg=sorg, h=h, krgendanchor="", tag="")
    gasoil.add_corey_oil(kroend=kroend)
    gasoil.add_corey_gas(krgend=krgend, krgmax=krgmax)
    check_table(gasoil.table)
    check_linear_sections(gasoil)
    assert gasoil.selfcheck()
    check_endpoints(gasoil, krgend, krgmax, kroend, kromax)
    assert 0 < gasoil.crosspoint() < 1

    # Redo with LET:
    gasoil = GasOil(swl=swl, sgcr=sgcr, sorg=sorg, h=h, tag="")
    gasoil.add_LET_oil(t=1.1, kroend=kroend, kromax=kromax)
    gasoil.add_LET_gas(krgend=krgend, krgmax=krgmax)
    check_table(gasoil.table)
    check_linear_sections(gasoil)
    assert gasoil.selfcheck()
    check_endpoints(gasoil, krgend, krgmax, kroend, kromax)
    assert 0 < gasoil.crosspoint() < 1


def check_endpoints(gasoil, krgend, krgmax, kroend, kromax):
    """Discrete tests that endpoints get numerically correct"""
    swtol = 1 / SWINTEGERS

    # Oil curve, from sg = 0 to sg = 1:
    if gasoil.sgro > 0:
        # Gas-condensate: sgcr = sgro > 0
        assert float_df_checker(gasoil.table, "SG", 0, "KROG", kromax)
        assert float_df_checker(gasoil.table, "SON", 1.0, "KROG", kroend)
        assert np.isclose(gasoil.table["KROG"].max(), kromax)
    else:
        assert float_df_checker(gasoil.table, "SG", 0, "KROG", kroend)
        assert np.isclose(gasoil.table["KROG"].max(), kroend)

    # son=0 @ 1 - sorg - swl or 1 - swl) should be zero:
    assert float_df_checker(gasoil.table, "SON", 0.0, "KROG", 0)
    # sgn=1 @ 1 - swl:
    assert float_df_checker(gasoil.table, "SGN", 1.0, "KROG", 0)
    assert np.isclose(gasoil.table["KROG"].min(), 0.0)

    # Gas curve, from sg=0 to sg=1:
    assert float_df_checker(gasoil.table, "SG", 0.0, "KRG", 0)
    assert float_df_checker(gasoil.table, "SGN", 0.0, "KRG", 0)
    assert float_df_checker(gasoil.table, "SG", gasoil.sgcr, "KRG", 0)

    # If krgendanchor == "sorg" then krgmax is irrelevant.
    if gasoil.sorg > swtol and gasoil.sorg > gasoil.h and gasoil.krgendanchor == "sorg":
        assert float_df_checker(gasoil.table, "SGN", 1.0, "KRG", krgend)
        assert np.isclose(gasoil.table["KRG"].max(), krgmax)
    if gasoil.krgendanchor != "sorg":
        assert np.isclose(gasoil.table["KRG"].max(), krgend)
    assert np.isclose(gasoil.table["KRG"].min(), 0.0)


def test_sgro_vs_sgcr():
    """Test that a sgro that is close enough to sgcr does not cause problems"""
    with pytest.raises(ValueError, match="sgro must be zero or equal to sgcr"):
        GasOil(sgcr=0.1, sgro=0.1 + 1e-7, h=0.01)
    with pytest.raises(ValueError, match="sgro must be zero or equal to sgcr"):
        GasOil(sgcr=0.1, sgro=0.1 - 1e-7, h=0.01)

    # This is close enough to sgcr:
    gasoil = GasOil(sgcr=0.1, sgro=0.1 + 1e-8, h=0.01)
    gasoil.add_corey_oil(nog=2, kroend=0.5, kromax=1)

    # KROG should not be affected at sg=0.1, even if sgro is "larger":
    assert float_df_checker(gasoil.table, "SG", 0.0, "KROG", 1.0)
    assert float_df_checker(gasoil.table, "SG", 0.1, "KROG", 0.5)
    assert gasoil.table["SG"][0] == 0
    assert gasoil.table["SG"][1] == 0.1


def test_gasoil_krgendanchor():
    """Test behaviour of the krgendanchor"""
    gasoil = GasOil(krgendanchor="sorg", sorg=0.2, h=0.1)
    assert gasoil.sorg
    gasoil.add_corey_gas(ng=1)
    gasoil.add_corey_oil(nog=1)

    # kg should be 1.0 at 1 - sorg due to krgendanchor == "sorg":
    assert (
        gasoil.table[np.isclose(gasoil.table["SG"], 1 - gasoil.sorg)]["KRG"].values[0]
        == 1.0
    )
    assert gasoil.table[np.isclose(gasoil.table["SG"], 1.0)]["KRG"].values[0] == 1.0

    gasoil = GasOil(krgendanchor="", sorg=0.2, h=0.1)
    assert gasoil.sorg
    gasoil.add_corey_gas(ng=1)
    gasoil.add_corey_oil(nog=1)

    # kg should be < 1 at 1 - sorg due to krgendanchor being ""
    assert (
        gasoil.table[np.isclose(gasoil.table["SG"], 1 - gasoil.sorg)]["KRG"].values[0]
        < 1.0
    )
    assert gasoil.table[np.isclose(gasoil.table["SG"], 1.0)]["KRG"].values[0] == 1.0
    assert gasoil.selfcheck()
    assert gasoil.crosspoint() > 0

    # Test once more for LET curves:
    gasoil = GasOil(krgendanchor="sorg", sorg=0.2, h=0.1)
    assert gasoil.sorg
    gasoil.add_LET_gas(1, 1, 1.1)
    gasoil.add_LET_oil(1, 1, 1.1)
    check_linear_sections(gasoil)
    assert 0 < gasoil.crosspoint() < 1

    # kg should be 1.0 at 1 - sorg due to krgendanchor == "sorg":
    assert (
        gasoil.table[np.isclose(gasoil.table["SG"], 1 - gasoil.sorg)]["KRG"].values[0]
        == 1.0
    )
    assert gasoil.table[np.isclose(gasoil.table["SG"], 1.0)]["KRG"].values[0] == 1.0

    gasoil = GasOil(krgendanchor="", sorg=0.2, h=0.1)
    assert gasoil.sorg
    gasoil.add_LET_gas(1, 1, 1.1)
    gasoil.add_LET_oil(1, 1, 1.1)
    check_linear_sections(gasoil)
    assert gasoil.selfcheck()

    # kg should be < 1 at 1 - sorg due to krgendanchor being ""
    assert (
        gasoil.table[np.isclose(gasoil.table["SG"], 1 - gasoil.sorg)]["KRG"].values[0]
        < 1.0
    )
    assert gasoil.table[np.isclose(gasoil.table["SG"], 1.0)]["KRG"].values[0] == 1.0


def test_nexus():
    """Test the Nexus export"""
    gasoil = GasOil(h=0.01, swl=0.1, sgcr=0.3, sorg=0.3)
    gasoil.add_corey_oil(nog=10, kroend=0.5)
    gasoil.add_corey_gas(ng=10, krgend=0.5)
    nexus_lines = gasoil.GOTABLE().splitlines()
    non_comments = [
        line for line in nexus_lines if not line.startswith("!") or not line
    ]
    assert non_comments[0] == "GOTABLE"
    assert non_comments[1] == "SG KRG KROG PC"
    df = pd.read_table(
        io.StringIO("\n".join(non_comments[2:])),
        engine="python",
        sep=r"\s+",
        header=None,
    )
    # pylint: disable=no-member  # false positive on Pandas dataframe
    assert (df.values <= 1.0).all()
    assert (df.values >= 0.0).all()


def test_linearsegments():
    """Made for testing the linear segments during
    the resolution of issue #163"""
    gasoil = GasOil(h=0.01, swl=0.1, sgcr=0.3, sorg=0.3)
    gasoil.add_corey_oil(nog=10, kroend=0.5)
    gasoil.add_corey_gas(ng=10, krgend=0.5)
    check_table(gasoil.table)
    check_linear_sections(gasoil)


def test_kroend():
    """Manual testing of kromax and kroend behaviour"""
    gasoil = GasOil(swirr=0.01, sgcr=0.01, h=0.01, swl=0.1, sorg=0.05)
    gasoil.add_LET_gas()
    gasoil.add_LET_oil(2, 2, 2.1)
    assert gasoil.table["KROG"].max() == 1
    gasoil.add_LET_oil(2, 2, 2.1, kroend=0.5)
    check_linear_sections(gasoil)
    assert gasoil.table["KROG"].max() == 0.5

    assert 0 < gasoil.crosspoint() < 1

    gasoil.add_corey_oil(2)
    assert gasoil.table["KROG"].max() == 1
    gasoil.add_corey_oil(nog=2, kroend=0.5)
    assert gasoil.table["KROG"].max() == 0.5


@given(
    st.floats(min_value=1e-4, max_value=MAX_EXPONENT),  # ng
    st.floats(min_value=1e-4, max_value=MAX_EXPONENT),  # nog
)
def test_gasoil_corey1(ng, nog):
    """Test the Corey formulation for gasoil"""
    gasoil = GasOil()
    try:
        gasoil.add_corey_oil(nog=nog)
        gasoil.add_corey_gas(ng=ng)
    except AssertionError:
        # This happens for "invalid" input
        return

    assert "KROG" in gasoil.table
    assert "KRG" in gasoil.table
    assert isinstance(gasoil.krgcomment, str)
    check_table(gasoil.table)
    sgofstr = gasoil.SGOF()
    assert len(sgofstr) > 100
    sat_table_str_ok(sgofstr)

    gasoil.update_sgcomment_and_sorg()
    check_table(gasoil.table)
    check_linear_sections(gasoil)
    sgofstr = gasoil.SGOF()
    assert len(sgofstr) > 100
    sat_table_str_ok(sgofstr)


def test_comments():
    """Test that the outputters include endpoints in comments"""
    gasoil = GasOil(h=0.3)
    gasoil.add_corey_gas()
    gasoil.add_corey_oil()
    sgfn = gasoil.SGFN()
    assert "--" in sgfn
    assert "pyscal: " in sgfn  # part of version string
    assert "swirr=0" in sgfn
    assert "sgcr=0" in sgfn
    assert "swl=0" in sgfn
    assert "sorg=0" in sgfn
    assert "ng=2" in sgfn
    assert "krgend=1" in sgfn
    assert "Corey" in sgfn
    assert "krg = krog @ sg=0.5" in sgfn
    assert "Zero capillary pressure" in sgfn
    assert "SG" in sgfn
    assert "KRG" in sgfn
    assert "PC" in sgfn

    sgof = gasoil.SGOF()
    assert "--" in sgof
    assert "pyscal: " in sgof  # part of version string
    assert "swirr=0" in sgof
    assert "sgcr=0" in sgof
    assert "swl=0" in sgof
    assert "sorg=0" in sgof
    assert "ng=2" in sgof
    assert "nog=2" in sgof
    assert "krgend=1" in sgof
    assert "Corey" in sgof
    assert "krg = krog @ sg=0.5" in sgof
    assert "Zero capillary pressure" in sgof
    assert "SG" in sgof
    assert "KRG" in sgof
    assert "KROG" in sgof
    assert "PC" in sgof


@given(st.floats(), st.floats(), st.floats(), st.floats(), st.floats())
def test_gasoil_let1(l, e, t, krgend, krgmax):
    """Test the LET formulation, take 1"""
    gasoil = GasOil()
    try:
        gasoil.add_LET_oil(l, e, t, krgend)
        gasoil.add_LET_gas(l, e, t, krgend, krgmax)
    except AssertionError:
        # This happens for negative values f.ex.
        return
    assert "KROG" in gasoil.table
    assert "KRG" in gasoil.table
    assert isinstance(gasoil.krgcomment, str)
    check_table(gasoil.table)
    check_linear_sections(gasoil)
    sgofstr = gasoil.SGOF()
    assert len(sgofstr) > 100
    sat_table_str_ok(sgofstr)


def test_sgfn():
    """Test that we can call SGFN without oil relperm defined"""
    gasoil = GasOil()
    gasoil.add_corey_gas()
    sgfn_str = gasoil.SGFN()
    assert "SGFN" in sgfn_str
    assert len(sgfn_str) > 15


def test_fast():
    """Test the fast option"""
    # First without fast-mode:
    gasoil = GasOil(h=0.1)
    gasoil.add_corey_gas()
    gasoil.add_corey_oil()
    # This crosspoint computation is only present for fast=False:
    assert "-- krg = krog @ sg=0.5" in gasoil.SGOF()

    # Provoke non-strict-monotone krow:
    gasoil.table.loc[0:2, "KROG"] = [1.00, 0.81, 0.81]
    # (this is valid in non-imbibition, but pyscal will correct it for all
    # curves)
    assert "0.1000000 0.0100000 0.8100000 0.0000000" in gasoil.SGOF()
    assert "0.2000000 0.0400000 0.8099999 0.0000000" in gasoil.SGOF()
    #   monotonicity correction:   ^^^^^^

    # Now redo with fast option:
    gasoil = GasOil(h=0.1, fast=True)
    gasoil.add_corey_gas()
    gasoil.add_corey_oil()
    # This crosspoint computation is only present for fast=False:
    assert "-- krg = krog" not in gasoil.SGOF()

    # Provoke non-strict-monotone krow, in fast-mode
    # this slips through:
    gasoil.table.loc[0:2, "KROG"] = [1.00, 0.81, 0.81]
    assert "0.1000000 0.0100000 0.8100000 0.0000000" in gasoil.SGOF()
    assert "0.2000000 0.0400000 0.8100000 0.0000000" in gasoil.SGOF()
    # not corrected:               ^^^^^^

    gasoil.table.loc[0:2, "KRG"] = [0.00, 0.01, 0.01]
    assert "0.1000000 0.0100000" in gasoil.SGFN()
    assert "0.2000000 0.0100000" in gasoil.SGFN()


def test_roundoff():
    """Test robustness to monotonicity issues arising from
    representation errors

    https://docs.python.org/3/tutorial/floatingpoint.html#representation-error

    The dataframe injected in this function has occured in the wild, and
    caused fatal errors in Eclipse100. The error lies in
    pd.dataframe.to_csv(float_format=".7f") which does truncation of floating points
    instead of rounding (intentional). Since we have a strict dependency on
    monotonicity properties for Eclipse to work, the data must be rounded
    before being sent to to_csv(). This is being done in the .SGOF() and SWOF() as
    it is a representation issue, not a numerical issues in the objects themselves.
    """

    gasoil = GasOil()
    # Inject a custom dataframe that has occured in the wild,
    # and given monotonicity issues in GasOil.SGOF().
    gasoil.table = pd.DataFrame(
        columns=["SG", "KRG", "KROG", "PC"],
        data=[
            [0.02, 0, 0.19524045000000001, 0],
            [0.040000000000000001, 0, 0.19524044999999998, 0],
            [0.059999999999999998, 0, 0.19524045000000004, 0],
            [0.080000000000000002, 0, 0.19524045000000001, 0],
            [0.10000000000000001, 0, 0.19524045000000001, 0],
            [0.16, 0, 0.19524045000000001, 0],
            [0.17999999999999999, 0, 0.19524045000000001, 0],
            [0.19999999999999998, 0, 0.19524044999999998, 0],
            [0.22, 0, 0.19524045000000001, 0],
            [1, 1, 0, 0],
        ],
    )
    gasoil.table["SGN"] = gasoil.table["SG"]
    gasoil.table["SON"] = 1 - gasoil.table["SG"]
    # If this value (as string) occurs, then we are victim of floating point truncation
    # in float_format=".7f":
    assert "0.1952404" not in gasoil.SGOF()
    assert "0.1952405" in gasoil.SGOF()
    check_table(gasoil.table)  # This function allows this monotonicity hiccup.


@pytest.mark.parametrize(
    "columnname, errorvalues",
    [
        ("SG", [1, 0]),
        ("SG", [0, 2]),
        ("KRG", [1, 0]),
        ("KRG", [0, 2]),
        ("KRG", [1, 2]),
        ("KROG", [0, 1]),
        ("KROG", [2, 0]),
        ("PC", [np.inf, 0]),
        ("PC", [np.nan, 0]),
        ("PC", [1, 2]),
        ("PC", [0, 1]),
    ],
)
def test_selfcheck(columnname, errorvalues):
    """Test the selfcheck feature of a GasOil object"""
    gasoil = GasOil(h=1)
    gasoil.add_corey_gas()
    gasoil.add_corey_oil()
    assert gasoil.selfcheck()

    # Punch the internal table directly to trigger error:
    gasoil.table[columnname] = errorvalues
    assert not gasoil.selfcheck()
    assert gasoil.SGOF() == ""
    if not columnname == "KROG":
        assert gasoil.SGFN() == ""
