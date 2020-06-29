"""Test module for pyscal.utils"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np

import pytest

from hypothesis import given, settings
import hypothesis.strategies as st


from pyscal import utils, WaterOil, GasOil
from common import check_table, float_df_checker, sat_table_str_ok


def test_df2str():
    """Test handling of roundoff issues when printing dataframes

    See also test_gasoil.py::test_roundoff()
    """
    # Easy cases:
    assert utils.df2str(pd.DataFrame(data=[0.1]), digits=1).strip() == "0.1"
    assert utils.df2str(pd.DataFrame(data=[0.1]), digits=3).strip() == "0.100"
    assert (
        utils.df2str(pd.DataFrame(data=[0.1]), digits=3, roundlevel=3).strip()
        == "0.100"
    )
    assert (
        utils.df2str(pd.DataFrame(data=[0.1]), digits=3, roundlevel=4).strip()
        == "0.100"
    )
    assert (
        utils.df2str(pd.DataFrame(data=[0.1]), digits=3, roundlevel=5).strip()
        == "0.100"
    )
    assert (
        utils.df2str(pd.DataFrame(data=[0.01]), digits=3, roundlevel=2).strip()
        == "0.010"
    )

    # Here roundlevel will ruin the result:
    assert (
        utils.df2str(pd.DataFrame(data=[0.01]), digits=3, roundlevel=1).strip()
        == "0.000"
    )

    # Tricky ones:
    # This one should be rounded down:
    assert utils.df2str(pd.DataFrame(data=[0.0034999]), digits=3).strip() == "0.003"
    # But if we are on the 9999 side due to representation error, the
    # number probably represents 0.0035 so it should be rounded up
    assert (
        utils.df2str(
            pd.DataFrame(data=[0.003499999999998]), digits=3, roundlevel=5
        ).strip()
        == "0.004"
    )
    # If we round to more digits than we have in IEE754, we end up truncating:
    assert (
        utils.df2str(
            pd.DataFrame(data=[0.003499999999998]), digits=3, roundlevel=20
        ).strip()
        == "0.003"  # Wrong
    )
    # If we round straight to out output, we are not getting the chance to correct for
    # the representation error:
    assert (
        utils.df2str(
            pd.DataFrame(data=[0.003499999999998]), digits=3, roundlevel=3
        ).strip()
        == "0.003"  # Wrong
    )
    # So roundlevel > digits
    assert (
        utils.df2str(
            pd.DataFrame(data=[0.003499999999998]), digits=3, roundlevel=4
        ).strip()
        == "0.004"
    )
    # But digits < roundlevel < 15 works:
    assert (
        utils.df2str(
            pd.DataFrame(data=[0.003499999999998]), digits=3, roundlevel=14
        ).strip()
        == "0.004"
    )
    assert (
        utils.df2str(
            pd.DataFrame(data=[0.003499999999998]), digits=3, roundlevel=15
        ).strip()
        == "0.003"  # Wrong
    )

    # Double rounding is a potential issue, as:
    assert round(0.0034445, 5) == 0.00344
    assert round(round(0.0034445, 6), 5) == 0.00345  # Wrong
    # So if pd.to_csv would later round instead of truncate, we could be victim
    # of this, having roundlevel >  digits + 1 would avoid that:
    assert round(round(0.0034445, 7), 5) == 0.00344
    # (this is the rationale for roundlevel > digits + 1)


def test_df2str_monotone():
    """Test the monotonocity enforcement in df2str()"""

    # Don't touch all-zero columns
    assert (
        utils.df2str(pd.DataFrame(data=[0, 0, 0]), digits=2, monotone_column=0)
        == "0\n0\n0\n"
    )
    # A constant nonzero column, makes no sense as capillary pressure
    # but still we ensure it runs in eclipse:
    assert (
        utils.df2str(pd.DataFrame(data=[1, 1, 1]), digits=2, monotone_column=0)
        == "1.00\n0.99\n0.98\n"
    )
    assert (
        utils.df2str(
            pd.DataFrame(data=[1, 1, 1]),
            digits=2,
            monotone_column=0,
            monotone_direction=-1,
        )
        == "1.00\n0.99\n0.98\n"
    )
    assert (
        utils.df2str(
            pd.DataFrame(data=[1, 1, 1]),
            digits=2,
            monotone_column=0,
            monotone_direction="dec",
        )
        == "1.00\n0.99\n0.98\n"
    )
    assert (
        utils.df2str(
            pd.DataFrame(data=[1, 1, 1]),
            digits=2,
            monotone_column=0,
            monotone_direction=1,
        )
        == "1.00\n1.01\n1.02\n"
    )
    assert (
        utils.df2str(
            pd.DataFrame(data=[1, 1, 1]),
            digits=2,
            monotone_column=0,
            monotone_direction="inc",
        )
        == "1.00\n1.01\n1.02\n"
    )
    with pytest.raises(ValueError):
        assert (
            utils.df2str(
                pd.DataFrame(data=[1, 1, 1]),
                digits=2,
                monotone_column=0,
                monotone_direction="foo",
            )
            == "1.00\n1.01\n1.02\n"
        )

    assert (
        utils.df2str(pd.DataFrame(data=[1, 1, 1]), digits=7, monotone_column=0)
        == "1.0000000\n0.9999999\n0.9999998\n"
    )

    # Actual data that has occured:
    dframe = pd.DataFrame(
        data=[0.0000027, 0.0000026, 0.0000024, 0.0000024, 0.0000017], columns=["pc"]
    )
    assert (
        utils.df2str(dframe, monotone_column="pc")
        == "0.0000027\n0.0000026\n0.0000024\n0.0000023\n0.0000017\n"
    )


def test_diffjumppoint():
    """Test estimator for the jump in first derivative for some manually set up cases.

    This code is also extensively tested throuth test_addfromtable"""

    dframe = pd.DataFrame(columns=["x", "y"], data=[[0, 0], [0.3, 0.2], [1, 1]])

    assert utils.estimate_diffjumppoint(dframe, side="right") == 0.3
    assert utils.estimate_diffjumppoint(dframe, side="left") == 0.3

    dframe = pd.DataFrame(columns=["x", "y"], data=[[0, 0], [1, 1]])
    # We don't really care what gets printed from this, just don't crash..
    assert 0 <= utils.estimate_diffjumppoint(dframe, side="right") <= 1
    assert 0 <= utils.estimate_diffjumppoint(dframe, side="left") <= 1

    dframe = pd.DataFrame(
        columns=["x", "y"],
        data=[
            [0, 0],
            [0.1, 0.1],
            [0.2, 0.2],  # Linear until here
            [0.3, 0.4],  # Nonlinear region
            [0.4, 0.45],  # Nonlinear region
            [0.7, 0.7],  # Linear from here
            [0.8, 0.8],
            [1, 1],
        ],
    )
    assert utils.estimate_diffjumppoint(dframe, side="left") == 0.2
    assert utils.estimate_diffjumppoint(dframe, side="right") == 0.7

    dframe = pd.DataFrame(
        columns=["x", "y"],
        data=[
            [0, 0],
            [0.1, 0.0],
            [0.2, 0.0],  # Linear until here
            [0.3, 0.4],  # Nonlinear region
            [0.9, 1],  # Start linear region again
            [1, 1],
        ],
    )
    assert utils.estimate_diffjumppoint(dframe, side="left") == 0.2
    assert utils.estimate_diffjumppoint(dframe, side="right") == 0.9


@settings(deadline=1000)
@given(
    st.floats(min_value=0, max_value=0.1),  # swl
    st.floats(min_value=0, max_value=0.0),  # dswcr
    st.floats(min_value=0, max_value=0.1),  # dswlhigh
    st.floats(min_value=0, max_value=0.3),  # sorw
    st.floats(min_value=0.1, max_value=5),  # nw1
    st.floats(min_value=0.1, max_value=1),  # krwend1
    st.floats(min_value=0.1, max_value=5),  # now1
    st.floats(min_value=0.1, max_value=1),  # kroend1
    st.floats(min_value=0.1, max_value=5),  # nw2
    st.floats(min_value=0.1, max_value=1),  # krwend2
    st.floats(min_value=0.1, max_value=5),  # now2
    st.floats(min_value=0.1, max_value=1),  # kroend2
)
def test_normalize_nonlinpart_wo_hypo(
    swl, dswcr, dswlhigh, sorw, nw1, krwend1, now1, kroend1, nw2, krwend2, now2, kroend2
):
    """Test the normalization code in utils.

    In particular the fill_value argument to scipy has been tuned to
    fulfill this code"""
    wo_low = WaterOil(swl=swl, swcr=swl + dswcr, sorw=sorw)
    wo_high = WaterOil(
        swl=swl + dswlhigh, swcr=swl + dswlhigh + dswcr, sorw=max(sorw - 0.01, 0)
    )
    wo_low.add_corey_water(nw=nw1, krwend=krwend1)
    wo_high.add_corey_water(nw=nw2, krwend=krwend2)
    wo_low.add_corey_oil(now=now1, kroend=kroend1)
    wo_high.add_corey_oil(now=now2, kroend=kroend2)

    krwn1, kron1 = utils.normalize_nonlinpart_wo(wo_low)
    assert np.isclose(krwn1(0), 0)
    assert np.isclose(krwn1(1), krwend1)
    assert np.isclose(kron1(0), 0)
    assert np.isclose(kron1(1), kroend1)

    krwn2, kron2 = utils.normalize_nonlinpart_wo(wo_high)
    assert np.isclose(krwn2(0), 0)
    assert np.isclose(krwn2(1), krwend2)
    assert np.isclose(kron2(0), 0)
    assert np.isclose(kron2(1), kroend2)


@given(
    st.floats(min_value=0, max_value=0.3),  # swirr
    st.floats(min_value=0.01, max_value=0.3),  # dswl
)
def test_normalize_pc(swirr, dswl):
    """Test that we can normalize a pc curve"""
    wateroil = WaterOil(swirr=swirr, swl=swirr + dswl)
    wateroil.add_simple_J()
    pc_max = wateroil.table["pc"].max()
    pc_min = wateroil.table["pc"].min()

    pc_fn = utils.normalize_pc(wateroil)
    assert np.isclose(pc_fn(0), pc_max)
    assert np.isclose(pc_fn(1), pc_min)


def test_normalize_emptypc():
    """Test that we can normalize both
    when pc is missing, and when it is all zero"""
    wateroil = WaterOil()
    pc_fn = utils.normalize_pc(wateroil)
    assert np.isclose(pc_fn(0), 0)
    assert np.isclose(pc_fn(1), 0)

    wateroil = WaterOil(swl=0.01)
    wateroil.add_simple_J(g=0)
    pc_fn = utils.normalize_pc(wateroil)
    assert np.isclose(pc_fn(0), 0)
    assert np.isclose(pc_fn(1), 0)


def test_normalize_nonlinpart_wo():
    """Manual tests for utils.normalize_nonlinpart_wo"""
    wateroil = WaterOil(swl=0.1, swcr=0.12, sorw=0.05, h=0.05)
    wateroil.add_corey_water(nw=2.1, krwend=0.9)
    wateroil.add_corey_oil(now=3, kroend=0.8)
    krwn, kron = utils.normalize_nonlinpart_wo(wateroil)

    assert np.isclose(krwn(0), 0)
    assert np.isclose(krwn(1), 0.9)

    # kron is normalized on son
    assert np.isclose(kron(0), 0)
    assert np.isclose(kron(1), 0.8)

    # Test with tricky endpoints
    h = 0.01
    wateroil = WaterOil(swl=h, swcr=h, sorw=h, h=h)
    wateroil.add_corey_water(nw=2.1, krwend=0.9)
    wateroil.add_corey_oil(now=3, kroend=0.8)
    krwn, kron = utils.normalize_nonlinpart_wo(wateroil)
    assert np.isclose(krwn(0), 0.0)
    assert np.isclose(krwn(1), 0.9)
    assert np.isclose(kron(0), 0)
    assert np.isclose(kron(1), 0.8)

    # Test again with zero endpoints:
    wateroil = WaterOil(swl=0, swcr=0, sorw=0, h=0.01)
    wateroil.add_corey_water(nw=2.1, krwend=0.9)
    wateroil.add_corey_oil(now=3, kroend=0.8)
    krwn, kron = utils.normalize_nonlinpart_wo(wateroil)
    assert np.isclose(krwn(0), 0.0)
    assert np.isclose(krwn(1), 0.9)
    assert np.isclose(kron(0), 0)
    assert np.isclose(kron(1), 0.8)

    # Test when endpoints are messed up:
    wateroil = WaterOil(swl=0.1, swcr=0.2, sorw=0.1, h=0.1)
    wateroil.add_corey_water(nw=2.1, krwend=0.6)
    wateroil.add_corey_oil(now=3, kroend=0.8)
    wateroil.swl = 0
    wateroil.swcr = 0
    wateroil.sorw = 0
    krwn, kron = utils.normalize_nonlinpart_wo(wateroil)
    # These go well still, since we are at zero
    assert np.isclose(krwn(0), 0.0)
    assert np.isclose(kron(0), 0)
    # These do not match when endpoints are wrong
    assert not np.isclose(krwn(1), 0.6)
    # This is ok, as kro is anchored at swl
    assert np.isclose(kron(1), 0.8)

    # So fix endpoints!
    wateroil.swl = wateroil.table["sw"].min()
    wateroil.swcr = wateroil.estimate_swcr()
    wateroil.sorw = wateroil.estimate_sorw()
    # Try again
    krwn, kron = utils.normalize_nonlinpart_wo(wateroil)
    assert np.isclose(krwn(0), 0.0)
    assert np.isclose(kron(0), 0)
    assert np.isclose(krwn(1), 0.6)
    assert np.isclose(kron(1), 0.8)


def test_comment_formatter():
    """Test the comment formatter

    This is also tested through hypothesis.text()
    in test_wateroil and test_gasoil, there is it tested
    through the use of tag formatting
    """
    assert utils.comment_formatter(None) == "-- \n"
    assert utils.comment_formatter("\n") == "-- \n"
    assert utils.comment_formatter("\r\n") == "-- \n"
    assert utils.comment_formatter("\r") == "-- \n"
    assert utils.comment_formatter("foo") == "-- foo\n"
    assert utils.comment_formatter("foo", prefix="gaa") == "gaafoo\n"
    assert utils.comment_formatter("foo\nbar") == "-- foo\n-- bar\n"


def test_tag_preservation():
    """Test that we can preserve tags/comments through interpolation"""
    wo_low = WaterOil(swl=0.1)
    wo_high = WaterOil(swl=0.2)
    wo_low.add_corey_water(nw=2)
    wo_high.add_corey_water(nw=3)
    wo_low.add_corey_oil(now=2)
    wo_high.add_corey_oil(now=3)
    interpolant1 = utils.interpolate_wo(wo_low, wo_high, parameter=0.1, h=0.2)
    assert "Interpolated to 0.1" in interpolant1.tag
    sat_table_str_ok(interpolant1.SWOF())

    wo_high.tag = "FOOBAR"
    interpolant2 = utils.interpolate_wo(wo_low, wo_high, parameter=0.1, h=0.2)
    assert "Interpolated to 0.1" in interpolant2.tag
    assert "between" in interpolant2.tag
    assert wo_high.tag in interpolant2.tag
    sat_table_str_ok(interpolant2.SWOF())
    # wo_low.tag was empty deliberately here.

    # When wo_log and wo_high has the same tag:
    wo_low.tag = "FOOBAR"
    interpolant3 = utils.interpolate_wo(wo_low, wo_high, parameter=0.1, h=0.2)
    assert "Interpolated to 0.1" in interpolant3.tag
    assert "between" not in interpolant3.tag
    assert wo_high.tag in interpolant3.tag
    sat_table_str_ok(interpolant3.SWOF())

    # Explicit tag:
    interpolant4 = utils.interpolate_wo(
        wo_low, wo_high, parameter=0.1, h=0.2, tag="Explicit tag"
    )
    assert interpolant4.tag == "Explicit tag"

    # Tag with newline
    interpolant6 = utils.interpolate_wo(
        wo_low, wo_high, parameter=0.1, h=0.2, tag="Explicit tag\non two lines"
    )
    assert "Explicit tag" in interpolant6.tag
    print(interpolant6.SWOF())
    sat_table_str_ok(interpolant6.SWOF())

    # Empty tag:
    interpolant5 = utils.interpolate_wo(wo_low, wo_high, parameter=0.1, h=0.2, tag="")
    assert interpolant5.tag == ""

    # Also sample check for GasOil (calls the same code)
    go_low = GasOil()
    go_high = GasOil()
    go_low.add_corey_gas(ng=2)
    go_high.add_corey_gas(ng=3)
    go_low.add_corey_oil(nog=2)
    go_high.add_corey_oil(nog=3)
    interpolant1 = utils.interpolate_go(go_low, go_high, parameter=0.1, h=0.2)
    assert "Interpolated to 0.1" in interpolant1.tag
    sat_table_str_ok(interpolant1.SGOF())


@settings(max_examples=40, deadline=5000)
@given(
    st.floats(min_value=0.01, max_value=0.1),  # swl
    st.floats(min_value=0, max_value=0.0),  # dswcr
    st.floats(min_value=0, max_value=0.1),  # dswlhigh
    st.floats(min_value=0, max_value=0.3),  # sorw
    st.floats(min_value=1, max_value=5),  # nw_l
    st.floats(min_value=1, max_value=5),  # nw_h
    st.floats(min_value=1, max_value=5),  # now_l
    st.floats(min_value=1, max_value=5),  # now_h
    st.floats(min_value=0.1, max_value=1),  # krwend_l
    st.floats(min_value=0.1, max_value=1),  # krwend_h
    st.floats(min_value=0.1, max_value=1),  # kroend_l
    st.floats(min_value=0.1, max_value=1),  # kroend_h
)
def test_interpolate_wo(
    swl,
    dswcr,
    dswlhigh,
    sorw,
    nw_l,
    nw_h,
    now_l,
    now_h,
    krwend_l,
    krwend_h,
    kroend_l,
    kroend_h,
):
    """
    Generate two random WaterOil curves, interpolate between them
    and check that the difference between each interpolant is small,
    this essentially checks that we can go continously between the
    two functions.
    """
    wo_low = WaterOil(swl=swl, swcr=swl + dswcr, sorw=sorw)
    wo_high = WaterOil(
        swl=swl + dswlhigh, swcr=swl + dswlhigh + dswcr, sorw=max(sorw - 0.01, 0)
    )
    wo_low.add_corey_water(nw=nw_l, krwend=krwend_l)
    wo_high.add_corey_water(nw=nw_h, krwend=krwend_h)
    wo_low.add_corey_oil(now=now_l, kroend=kroend_l)
    wo_high.add_corey_oil(now=now_h, kroend=kroend_h)
    ips = []
    ip_dist = 0.05
    for tparam in np.arange(0, 1 + ip_dist, ip_dist):
        wo_ip = utils.interpolate_wo(wo_low, wo_high, tparam)
        check_table(wo_ip.table)
        assert wo_ip.tag
        ips.append(wo_ip)
        assert 0 < wo_ip.crosspoint() < 1

    # Distances between low and interpolants:
    dists = [
        (wo_low.table - interp.table)[["krw", "krow"]].sum().sum() for interp in ips
    ]
    assert np.isclose(dists[0], 0)

    # Distance between high and the last interpolant
    assert (wo_high.table - ips[-1].table)[["krw", "krow"]].sum().sum() < 0.01

    # Distances between low and interpolants:
    dists = [
        (wo_low.table - interp.table)[["krw", "krow"]].sum().sum() for interp in ips
    ]
    print(
        "Interpolation, mean: {}, min: {}, max: {}, std: {} ip-par-dist: {}".format(
            np.mean(dists), min(dists), max(dists), np.std(np.diff(dists[1:])), ip_dist
        )
    )
    assert np.isclose(dists[0], 0)  # Reproducing wo_low
    # All curves that are close in parameter t, should be close in sum().sum().
    # That means that diff of the distances should be similar,
    # that is the std.dev of the distances is low:
    ip_dist_std = np.std(np.diff(dists[1:]))  # This number depends on 'h' and 't' range
    # (avoiding the first which reproduces go_low
    if ip_dist_std > 1.0:  # Found by trial and error
        print("ip_dist_std: {}".format(ip_dist_std))
        print(dists)
        from matplotlib import pyplot as plt

        _, mpl_ax = plt.subplots()
        wo_low.plotkrwkrow(mpl_ax=mpl_ax, color="red")
        wo_high.plotkrwkrow(mpl_ax=mpl_ax, color="blue")
        for interp in ips:
            interp.plotkrwkrow(mpl_ax=mpl_ax, color="green")
        plt.show()
        assert False

    # If this fails, there is a combination of parameters that might
    # trigger a discontinuity in the interpolants, which we don't want.


@settings(max_examples=40, deadline=5000)
@given(
    st.floats(min_value=0.01, max_value=0.1),  # swl
    st.floats(min_value=0, max_value=0.0),  # dswcr
    st.floats(min_value=0, max_value=0.1),  # dswlhigh
    st.floats(min_value=0, max_value=0.3),  # sorw
    st.floats(min_value=0.1, max_value=2),  # a_l
    st.floats(min_value=0.1, max_value=2),  # a_h
    st.floats(min_value=-2, max_value=-0.1),  # b_l
    st.floats(min_value=-2, max_value=-0.5),  # b_l
)
def test_interpolate_wo_pc(swl, dswcr, dswlhigh, sorw, a_l, a_h, b_l, b_h):
    """
    Generate two random WaterOil curves, interpolate pc between them
    and check that the difference between each interpolant is small,
    this essentially checks that we can go continously between the
    two functions.
    """
    wo_low = WaterOil(swl=swl, swcr=swl + dswcr, sorw=sorw)
    wo_high = WaterOil(
        swl=swl + dswlhigh, swcr=swl + dswlhigh + dswcr, sorw=max(sorw - 0.01, 0)
    )
    wo_low.add_corey_water()
    wo_high.add_corey_water()
    wo_low.add_corey_oil()
    wo_high.add_corey_oil()
    wo_low.add_simple_J(a=a_l, b=b_l)
    wo_high.add_simple_J(a=a_h, b=b_h)
    ips = []
    ip_dist = 0.05
    for t in np.arange(0, 1 + ip_dist, ip_dist):
        wo_ip = utils.interpolate_wo(wo_low, wo_high, t)
        check_table(wo_ip.table)
        ips.append(wo_ip)
        assert 0 < wo_ip.crosspoint() < 1

    # Distances between low and interpolants:
    dists = [(wo_low.table - interp.table)[["pc"]].sum().sum() for interp in ips]
    assert np.isclose(dists[0], 0)

    # Distance between high and the last interpolant
    assert (wo_high.table - ips[-1].table)[["pc"]].sum().sum() < 0.01

    # Distances between low and interpolants:
    dists = [(wo_low.table - interp.table)[["pc"]].sum().sum() for interp in ips]
    print(
        "Interpolation, mean: {}, min: {}, max: {}, std: {} ip-par-dist: {}".format(
            np.mean(dists), min(dists), max(dists), np.std(np.diff(dists[1:])), ip_dist
        )
    )
    assert np.isclose(dists[0], 0)  # Reproducing wo_low
    # All curves that are close in parameter t, should be close in sum().sum().
    # That means that diff of the distances should be similar,
    # that is the std.dev of the distances is low:
    ip_dist_std = np.std(np.diff(dists[1:]))  # This number depends on 'h' and 't' range
    # (avoiding the first which reproduces go_low
    if ip_dist_std > 1.0:  # Found by trial and error
        print("ip_dist_std: {}".format(ip_dist_std))
        print(dists)
        from matplotlib import pyplot as plt

        _, mpl_ax = plt.subplots()
        wo_low.plotpc(mpl_ax=mpl_ax, color="red", logyscale=True)
        wo_high.plotpc(mpl_ax=mpl_ax, color="blue", logyscale=True)
        for interp in ips:
            interp.plotpc(mpl_ax=mpl_ax, color="green", logyscale=True)
        plt.show()
        assert False

    # If this fails, there is a combination of parameters that might
    # trigger a discontinuity in the interpolants, which we don't want.


@settings(max_examples=40, deadline=1000)
@given(
    st.floats(min_value=0, max_value=0.1),  # swl
    st.floats(min_value=0, max_value=0.1),  # sgcr
    st.floats(min_value=0, max_value=0.1),  # dswlhigh
    st.floats(min_value=0, max_value=0.1),  # dsgcr
    st.floats(min_value=0, max_value=0.3),  # sorg
    st.floats(min_value=0, max_value=0.1),  # dsorg
    st.floats(min_value=0.1, max_value=5),  # ng1
    st.floats(min_value=0.01, max_value=1),  # krgend1
    st.floats(min_value=0.1, max_value=5),  # nog1
    st.floats(min_value=0.01, max_value=1),  # kroend1
    st.floats(min_value=0.1, max_value=5),  # ng2
    st.floats(min_value=0.01, max_value=1),  # krgend2
    st.floats(min_value=0.1, max_value=5),  # nog2
    st.floats(min_value=0.01, max_value=1),  # kroend2
)
def test_normalize_nonlinpart_go_hypo(
    swl,
    sgcr,
    dswlhigh,
    dsgcr,
    sorg,
    dsorg,
    ng1,
    krgend1,
    nog1,
    kroend1,
    ng2,
    krgend2,
    nog2,
    kroend2,
):
    """Test the normalization code in utils.

    In particular the fill_value argument to scipy has been tuned to
    fulfill this code"""
    go_low = GasOil(swl=swl, sgcr=sgcr, sorg=sorg)
    go_high = GasOil(swl=swl + dswlhigh, sgcr=sgcr + dsgcr, sorg=max(sorg - dsorg, 0))
    go_low.add_corey_gas(ng=ng1, krgend=krgend1)
    go_high.add_corey_gas(ng=ng2, krgend=krgend2)
    go_low.add_corey_oil(nog=nog1, kroend=kroend1)
    go_high.add_corey_oil(nog=nog2, kroend=kroend2)

    krgn1, kron1 = utils.normalize_nonlinpart_go(go_low)
    assert np.isclose(krgn1(0), 0)
    assert np.isclose(krgn1(1), krgend1)
    assert np.isclose(kron1(0), 0)
    assert np.isclose(kron1(1), kroend1)

    krgn2, kron2 = utils.normalize_nonlinpart_go(go_high)
    assert np.isclose(krgn2(0), 0)
    assert np.isclose(krgn2(1), krgend2)
    assert np.isclose(kron2(0), 0)
    assert np.isclose(kron2(1), kroend2)


def test_normalize_nonlinpart_go():
    """Manual tests for utils.normalize_nonlinpart_go"""
    gasoil = GasOil(swl=0.1, sgcr=0.12, sorg=0.05, h=0.05)
    gasoil.add_corey_gas(ng=2.1, krgend=0.9)
    gasoil.add_corey_oil(nog=3, kroend=0.8)
    krgn, kron = utils.normalize_nonlinpart_go(gasoil)

    assert np.isclose(krgn(0), 0)
    assert np.isclose(krgn(1), 0.9)

    # kron is normalized on son
    assert np.isclose(kron(0), 0)
    assert np.isclose(kron(1), 0.8)

    # Test with tricky endpoints
    h = 0.01
    gasoil = GasOil(swl=h, sgcr=h, sorg=h, h=h)
    gasoil.add_corey_gas(ng=2.1, krgend=0.9)
    gasoil.add_corey_oil(nog=3, kroend=0.8)
    krgn, kron = utils.normalize_nonlinpart_go(gasoil)
    assert np.isclose(krgn(0), 0.0)
    assert np.isclose(krgn(1), 0.9)
    assert np.isclose(kron(0), 0)
    assert np.isclose(kron(1), 0.8)

    # Test again with zero endpoints:
    gasoil = GasOil(swl=0, sgcr=0, sorg=0, h=0.01)
    gasoil.add_corey_gas(ng=2.1, krgend=0.9)
    gasoil.add_corey_oil(nog=3, kroend=0.8)
    krgn, kron = utils.normalize_nonlinpart_go(gasoil)
    assert np.isclose(krgn(0), 0.0)
    assert np.isclose(krgn(1), 0.9)
    assert np.isclose(kron(0), 0)
    assert np.isclose(kron(1), 0.8)

    # Test when endpoints are messed up (cleared)
    gasoil = GasOil(swl=0.1, sgcr=0.2, sorg=0.1, h=0.1)
    gasoil.add_corey_gas(ng=2.1, krgend=0.6)
    gasoil.add_corey_oil(nog=3, kroend=0.8)
    gasoil.swl = 0
    gasoil.sgcr = 0
    gasoil.sorg = 0
    krgn, kron = utils.normalize_nonlinpart_go(gasoil)
    # These go well still, since we are at zero
    assert np.isclose(krgn(0), 0.0)
    assert np.isclose(kron(0), 0)
    assert np.isclose(kron(1), 0.8)
    # These do not match when endpoints are wrong
    assert not np.isclose(krgn(1), 0.6)

    # So fix endpoints!
    gasoil.swl = 1 - gasoil.table["sg"].max()
    gasoil.sgcr = gasoil.estimate_sgcr()
    gasoil.sorg = gasoil.estimate_sorg()
    # Try again
    krgn, kron = utils.normalize_nonlinpart_go(gasoil)
    assert np.isclose(krgn(0), 0.0)
    assert np.isclose(kron(0), 0)
    assert np.isclose(krgn(1), 0.6)
    assert np.isclose(kron(1), 0.8)


def test_ip_wo_kroend():
    """Test behaviour of kroend under interpolation"""
    wo_low = WaterOil(swl=0, swcr=0.1, sorw=0.2)
    wo_low.add_corey_water(nw=2, krwend=0.5, krwmax=0.7)
    wo_low.add_corey_oil(now=2, kroend=0.6)

    wo_high = WaterOil(swl=0.02, swcr=0.05, sorw=0.1)
    wo_high.add_corey_water(nw=2, krwend=0.5, krwmax=0.72)
    wo_high.add_corey_oil(now=2, kroend=0.7)

    # Interpolate to midpoint between the curves above
    wo_ip = utils.interpolate_wo(wo_low, wo_high, 0.5)

    # kroend at mean swl:
    assert float_df_checker(wo_ip.table, "sw", 0.01, "krow", (0.6 + 0.7) / 2.0)

    assert float_df_checker(wo_ip.table, "sw", 1, "krw", 0.71)
    assert float_df_checker(wo_ip.table, "sw", 1 - 0.15, "krw", 0.5)


def test_ip_go_kroend():
    """Test behaviour of kroend under interpolation"""
    go_low = GasOil(swl=0, sgcr=0.1, sorg=0.2)
    go_low.add_corey_gas(ng=2, krgend=0.5, krgmax=0.7)
    go_low.add_corey_oil(nog=2, kroend=0.6)

    go_high = GasOil(swl=0.02, sgcr=0.05, sorg=0.1)
    go_high.add_corey_gas(ng=2, krgend=0.5, krgmax=0.72)
    go_high.add_corey_oil(nog=2, kroend=0.7)

    # Interpolate to midpoint between the curves above
    go_ip = utils.interpolate_go(go_low, go_high, 0.5)

    # kroend at sg zero:
    assert float_df_checker(go_ip.table, "sg", 0.0, "krog", (0.6 + 0.7) / 2.0)

    assert np.isclose(go_ip.swl, 0.01)
    assert np.isclose(go_ip.sorg, 0.15)

    # krgmax at 1 - swl:
    assert float_df_checker(go_ip.table, "sg", 1 - go_ip.swl, "krg", 0.71)
    # krgend at 1 - swl - sorg
    assert float_df_checker(go_ip.table, "sg", 1 - go_ip.swl - go_ip.sorg, "krg", 0.5)

    # If krgendanchor is None, then krgmax should be irrelevant
    go_low = GasOil(swl=0, sgcr=0.1, sorg=0.2, krgendanchor="")
    go_low.add_corey_gas(ng=5, krgend=0.5, krgmax=0.7)
    # krgmax will trigger warning message, intended, as the 0.7
    # value here will mean nothing.
    go_low.add_corey_oil(nog=2, kroend=0.6)

    go_high = GasOil(swl=0.02, sgcr=0.05, sorg=0.1, krgendanchor="")
    go_high.add_corey_gas(ng=4, krgend=0.5, krgmax=0.72)
    # krgmax will trigger warning message, intended.
    go_high.add_corey_oil(nog=2, kroend=0.7)

    # Interpolate to midpoint between the curves above
    go_ip = utils.interpolate_go(go_low, go_high, 0.5, h=0.1)

    assert float_df_checker(go_ip.table, "sg", 0.0, "krog", (0.6 + 0.7) / 2.0)

    # Activate these line to see a bug, interpolation_go
    # does not honor krgendanchorA:
    # from matplotlib import pyplot as plt
    # _, mpl_ax = plt.subplots()
    # go_low.plotkrgkrog(mpl_ax=mpl_ax, color="red")
    # go_high.plotkrgkrog(mpl_ax=mpl_ax, color="blue")
    # go_ip.plotkrgkrog(mpl_ax=mpl_ax, color="green")
    # plt.show()

    # krgmax is irrelevant, krgend is used here:
    assert float_df_checker(go_ip.table, "sg", 1 - 0.01, "krg", 0.5)

    # Do we get into trouble if krgendanchor is different in low and high?
    go_low = GasOil(swl=0, sgcr=0.1, sorg=0.2, krgendanchor="sorg")
    go_low.add_corey_gas(ng=2, krgend=0.5, krgmax=0.7)
    go_low.add_corey_oil(nog=2, kroend=0.6)

    go_high = GasOil(swl=0.02, sgcr=0.05, sorg=0.1, krgendanchor="")
    go_high.add_corey_gas(ng=2, krgend=0.5)
    go_high.add_corey_oil(nog=2, kroend=0.7)

    # Interpolate to midpoint between the curves above
    go_ip = utils.interpolate_go(go_low, go_high, 0.5)

    assert float_df_checker(go_ip.table, "sg", 0.0, "krog", (0.6 + 0.7) / 2.0)

    # max(krg) is here avg of krgmax and krgend from the differnt tables:
    assert float_df_checker(go_ip.table, "sg", 1 - 0.01, "krg", 0.6)

    # krgend at 1 - swl - sorg, non-trivial expression, so a numerical
    # value is used here in the test:
    assert float_df_checker(go_ip.table, "sg", 1 - 0.01 - 0.15, "krg", 0.4491271)

    # krog-zero at 1 - swl - sorg:
    assert float_df_checker(go_ip.table, "sg", 1 - 0.01 - 0.15, "krog", 0)


@settings(max_examples=50, deadline=5000)
@given(
    st.floats(min_value=0, max_value=0.1),  # swl
    st.floats(min_value=0, max_value=0.1),  # sgcr
    st.floats(min_value=0, max_value=0.1),  # dsgcr
    st.floats(min_value=0, max_value=0.1),  # dswlhigh
    st.floats(min_value=0, max_value=0.3),  # sorg
    st.floats(min_value=0, max_value=0.1),  # dsorg
    st.floats(min_value=1, max_value=5),  # ng_l
    st.floats(min_value=1, max_value=5),  # ng_h
    st.floats(min_value=1, max_value=5),  # nog_l
    st.floats(min_value=1, max_value=5),  # nog_h
    st.floats(min_value=0.1, max_value=1),  # krgend_l
    st.floats(min_value=0.1, max_value=1),  # krgend_h
    st.floats(min_value=0.1, max_value=1),  # kroend_l
    st.floats(min_value=0.1, max_value=1),  # kroend_h
)
def test_interpolate_go(
    swl,
    sgcr,
    dsgcr,
    dswlhigh,
    sorg,
    dsorg,
    ng_l,
    ng_h,
    nog_l,
    nog_h,
    krgend_l,
    krgend_h,
    kroend_l,
    kroend_h,
):
    """Test many possible combinations of interpolation between two
    Corey gasoil curves, looking for numerical corner cases"""
    go_low = GasOil(swl=swl, sgcr=sgcr, sorg=sorg)
    go_high = GasOil(swl=swl + dswlhigh, sgcr=sgcr + dsgcr, sorg=max(sorg - dsorg, 0))
    go_low.add_corey_gas(ng=ng_l, krgend=krgend_l)
    go_high.add_corey_gas(ng=ng_h, krgend=krgend_h)
    go_low.add_corey_oil(nog=nog_l, kroend=kroend_l)
    go_high.add_corey_oil(nog=nog_h, kroend=kroend_h)
    ips = []
    ip_dist = 0.05
    for t in np.arange(0, 1 + ip_dist, ip_dist):
        go_ip = utils.interpolate_go(go_low, go_high, t)
        check_table(go_ip.table)
        ips.append(go_ip)
        assert 0 < go_ip.crosspoint() < 1

    # Distances between low and interpolants:
    dists = [
        (go_low.table - interp.table)[["krg", "krog"]].sum().sum() for interp in ips
    ]
    print(
        "Interpolation, mean: {}, min: {}, max: {}, std: {} ip-par-dist: {}".format(
            np.mean(dists), min(dists), max(dists), np.std(np.diff(dists[1:])), ip_dist
        )
    )
    assert np.isclose(dists[0], 0)  # Reproducing go_low
    # All curves that are close in parameter t, should be close in sum().sum().
    # That means that diff of the distances should be similar,
    # that is the std.dev of the distances is low:
    ip_dist_std = np.std(
        np.diff(dists[1:])
    )  # This number depends on 'h' and 't' range, and
    # by how different the low and high is.
    # (avoiding the first which reproduces go_low
    if ip_dist_std > 1.0:  # number found from trial and error.
        print("ip_dist_std: {}".format(ip_dist_std))
        print(dists)
        from matplotlib import pyplot as plt

        _, mpl_ax = plt.subplots()
        go_low.plotkrgkrog(mpl_ax=mpl_ax, color="red")
        go_high.plotkrgkrog(mpl_ax=mpl_ax, color="blue")
        for interp in ips:
            interp.plotkrgkrog(mpl_ax=mpl_ax, color="green")
        plt.show()
        assert False

    # If this fails, there is a combination of parameters that might
    # trigger a discontinuity in the interpolants, which we don't want.
