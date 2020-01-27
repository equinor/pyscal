"""Test module for pyscal.utils"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np

from hypothesis import given, settings
import hypothesis.strategies as st


from pyscal import utils, WaterOil, GasOil
from test_wateroil import check_table


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
    assert not np.isclose(kron(1), 0.8)

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
    # These do not match when endpoints are wrong
    assert not np.isclose(krgn(1), 0.6)
    assert not np.isclose(kron(1), 0.8)

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
