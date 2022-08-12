"""Test module for relperm interpolation support code"""

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from matplotlib import pyplot as plt

from pyscal import GasOil, WaterOil
from pyscal.constants import EPSILON as epsilon
from pyscal.utils.interpolation import (
    interpolate_go,
    interpolate_wo,
    normalize_nonlinpart_go,
    normalize_nonlinpart_wo,
    normalize_pc,
)
from pyscal.utils.testing import (
    check_table,
    float_df_checker,
    sat_table_str_ok,
    slow_hypothesis,
)


@slow_hypothesis
@given(
    st.floats(min_value=0, max_value=0.1),  # swl
    st.floats(min_value=0, max_value=0.0),  # dswcr
    st.floats(min_value=0, max_value=0.1),  # dswlhigh
    st.floats(min_value=0, max_value=0.3),  # sorw
    st.floats(min_value=0, max_value=0.2),  # dsocrsorw
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
    swl,
    dswcr,
    dswlhigh,
    sorw,
    dsocrsorw,
    nw1,
    krwend1,
    now1,
    kroend1,
    nw2,
    krwend2,
    now2,
    kroend2,
):
    # pylint: disable=too-many-arguments,too-many-locals
    """Test the normalization code in utils.

    In particular the fill_value argument to scipy has been tuned to
    fulfill this code"""
    wo_low = WaterOil(swl=swl, swcr=swl + dswcr, sorw=sorw, socr=sorw + dsocrsorw)
    sorw_high = max(sorw - 0.01, 0)
    wo_high = WaterOil(
        swl=swl + dswlhigh,
        swcr=swl + dswlhigh + dswcr,
        sorw=sorw_high,
        socr=sorw_high + dsocrsorw,
    )
    wo_low.add_corey_water(nw=nw1, krwend=krwend1)
    wo_high.add_corey_water(nw=nw2, krwend=krwend2)
    wo_low.add_corey_oil(now=now1, kroend=kroend1)
    wo_high.add_corey_oil(now=now2, kroend=kroend2)

    krwn1, kron1 = normalize_nonlinpart_wo(wo_low)
    assert np.isclose(krwn1(0), 0)
    assert np.isclose(krwn1(1), krwend1)
    assert np.isclose(kron1(0), 0)
    assert np.isclose(kron1(1), kroend1)

    krwn2, kron2 = normalize_nonlinpart_wo(wo_high)
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
    pc_max = wateroil.table["PC"].max()
    pc_min = wateroil.table["PC"].min()

    pc_fn = normalize_pc(wateroil)
    assert np.isclose(pc_fn(0), pc_max)
    assert np.isclose(pc_fn(1), pc_min)


def test_normalize_pc_wrongobject():
    """Test error message when wrong object is provided"""
    with pytest.raises(ValueError, match="Only WaterOil or GasOil"):
        normalize_pc({})


def test_normalize_emptypc():
    """Test that we can normalize both
    when pc is missing, and when it is all zero"""
    wateroil = WaterOil()
    pc_fn = normalize_pc(wateroil)
    assert np.isclose(pc_fn(0), 0)
    assert np.isclose(pc_fn(1), 0)

    wateroil = WaterOil(swl=0.01)
    wateroil.add_simple_J(g=0)
    pc_fn = normalize_pc(wateroil)
    assert np.isclose(pc_fn(0), 0)
    assert np.isclose(pc_fn(1), 0)


def test_normalize_nonlinpart_wo():
    """Manual tests for normalize_nonlinpart_wo"""
    wateroil = WaterOil(swl=0.1, swcr=0.12, sorw=0.05, h=0.05)
    wateroil.add_corey_water(nw=2.1, krwend=0.9)
    wateroil.add_corey_oil(now=3, kroend=0.8)
    krwn, kron = normalize_nonlinpart_wo(wateroil)

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
    krwn, kron = normalize_nonlinpart_wo(wateroil)
    assert np.isclose(krwn(0), 0.0)
    assert np.isclose(krwn(1), 0.9)
    assert np.isclose(kron(0), 0)
    assert np.isclose(kron(1), 0.8)

    # Test again with zero endpoints:
    wateroil = WaterOil(swl=0, swcr=0, sorw=0, h=0.01)
    wateroil.add_corey_water(nw=2.1, krwend=0.9)
    wateroil.add_corey_oil(now=3, kroend=0.8)
    krwn, kron = normalize_nonlinpart_wo(wateroil)
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
    krwn, kron = normalize_nonlinpart_wo(wateroil)
    # These go well still, since we are at zero
    assert np.isclose(krwn(0), 0.0)
    assert np.isclose(kron(0), 0)
    # These do not match when endpoints are wrong
    assert not np.isclose(krwn(1), 0.6)
    # This is ok, as kro is anchored at swl
    assert np.isclose(kron(1), 0.8)

    # So fix endpoints!
    wateroil.swl = wateroil.table["SW"].min()
    wateroil.swcr = wateroil.estimate_swcr()
    wateroil.sorw = wateroil.estimate_sorw()
    # Try again
    krwn, kron = normalize_nonlinpart_wo(wateroil)
    assert np.isclose(krwn(0), 0.0)
    assert np.isclose(kron(0), 0)
    assert np.isclose(krwn(1), 0.6)
    assert np.isclose(kron(1), 0.8)


def test_tag_preservation():
    """Test that we can preserve tags/comments through interpolation"""
    wo_low = WaterOil(swl=0.1)
    wo_high = WaterOil(swl=0.2)
    wo_low.add_corey_water(nw=2)
    wo_high.add_corey_water(nw=3)
    wo_low.add_corey_oil(now=2)
    wo_high.add_corey_oil(now=3)
    interpolant1 = interpolate_wo(wo_low, wo_high, parameter=0.1, h=0.2)
    assert "Interpolated to 0.1" in interpolant1.tag
    sat_table_str_ok(interpolant1.SWOF())

    wo_high.tag = "FOOBAR"
    interpolant2 = interpolate_wo(wo_low, wo_high, parameter=0.1, h=0.2)
    assert "Interpolated to 0.1" in interpolant2.tag
    assert "between" in interpolant2.tag
    assert wo_high.tag in interpolant2.tag
    sat_table_str_ok(interpolant2.SWOF())
    # wo_low.tag was empty deliberately here.

    # When wo_log and wo_high has the same tag:
    wo_low.tag = "FOOBAR"
    interpolant3 = interpolate_wo(wo_low, wo_high, parameter=0.1, h=0.2)
    assert "Interpolated to 0.1" in interpolant3.tag
    assert "between" not in interpolant3.tag
    assert wo_high.tag in interpolant3.tag
    sat_table_str_ok(interpolant3.SWOF())

    # Explicit tag:
    interpolant4 = interpolate_wo(
        wo_low, wo_high, parameter=0.1, h=0.2, tag="Explicit tag"
    )
    assert interpolant4.tag == "Explicit tag"

    # Tag with newline
    interpolant6 = interpolate_wo(
        wo_low, wo_high, parameter=0.1, h=0.2, tag="Explicit tag\non two lines"
    )
    assert "Explicit tag" in interpolant6.tag
    print(interpolant6.SWOF())
    sat_table_str_ok(interpolant6.SWOF())

    # Empty tag:
    interpolant5 = interpolate_wo(wo_low, wo_high, parameter=0.1, h=0.2, tag="")
    assert interpolant5.tag == ""

    # Also sample check for GasOil (calls the same code)
    go_low = GasOil()
    go_high = GasOil()
    go_low.add_corey_gas(ng=2)
    go_high.add_corey_gas(ng=3)
    go_low.add_corey_oil(nog=2)
    go_high.add_corey_oil(nog=3)
    interpolant1 = interpolate_go(go_low, go_high, parameter=0.1, h=0.2)
    assert "Interpolated to 0.1" in interpolant1.tag
    sat_table_str_ok(interpolant1.SGOF())


@slow_hypothesis
@given(
    st.floats(min_value=0.01, max_value=0.1),  # swl
    st.floats(min_value=0, max_value=0.0),  # dswcr
    st.floats(min_value=0, max_value=0.1),  # dswlhigh
    st.floats(min_value=0, max_value=0.3),  # sorw
    st.floats(min_value=0, max_value=0.3),  # dsocrsorw
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
    dsocrsorw,
    nw_l,
    nw_h,
    now_l,
    now_h,
    krwend_l,
    krwend_h,
    kroend_l,
    kroend_h,
):
    # pylint: disable=too-many-arguments,too-many-locals
    """
    Generate two random WaterOil curves, interpolate between them
    and check that the difference between each interpolant is small,
    this essentially checks that we can go continously between the
    two functions.
    """
    wo_low = WaterOil(swl=swl, swcr=swl + dswcr, sorw=sorw, socr=sorw + dsocrsorw)
    sorw_high = max(sorw - 0.01, 0)
    wo_high = WaterOil(
        swl=swl + dswlhigh,
        swcr=swl + dswlhigh + dswcr,
        sorw=max(sorw - 0.01, 0),
        socr=sorw_high + dsocrsorw,
    )
    wo_low.add_corey_water(nw=nw_l, krwend=krwend_l)
    wo_high.add_corey_water(nw=nw_h, krwend=krwend_h)
    wo_low.add_corey_oil(now=now_l, kroend=kroend_l)
    wo_high.add_corey_oil(now=now_h, kroend=kroend_h)
    ips = []
    ip_dist = 0.05
    for tparam in np.arange(0, 1 + ip_dist, ip_dist):
        wo_ip = interpolate_wo(wo_low, wo_high, tparam)
        check_table(wo_ip.table)
        assert wo_ip.tag
        ips.append(wo_ip)
        assert 0 < wo_ip.crosspoint() < 1

    # Distances between low and interpolants:
    dists = [
        (wo_low.table - interp.table)[["KRW", "KROW"]].sum().sum() for interp in ips
    ]

    # wo_low should be reproduced exactly:
    assert np.isclose(dists[0], 0)

    # Distance between high and the last interpolant
    assert (wo_high.table - ips[-1].table)[["KRW", "KROW"]].sum().sum() < 0.01

    # Distances between low and interpolants:
    dists = [
        (wo_low.table - interp.table)[["KRW", "KROW"]].sum().sum() for interp in ips
    ]
    print(
        f"Interpolation, mean: {np.mean(dists)}, min: {min(dists)}, "
        f"max: {max(dists)}, std: {np.std(np.diff(dists[1:]))} ip-par-dist: {ip_dist}"
    )
    # All curves that are close in parameter t, should be close in sum().sum().
    # That means that diff of the distances should be similar,
    # that is the std.dev of the distances is low:
    ip_dist_std = np.std(np.diff(dists[1:]))  # This number depends on 'h' and 't' range
    # (avoiding the first which reproduces go_low
    if ip_dist_std > 1.0:  # Found by trial and error
        print(f"ip_dist_std: {ip_dist_std}")
        print(dists)

        _, mpl_ax = plt.subplots()
        wo_low.plotkrwkrow(mpl_ax=mpl_ax, color="red")
        wo_high.plotkrwkrow(mpl_ax=mpl_ax, color="blue")
        for interp in ips:
            interp.plotkrwkrow(mpl_ax=mpl_ax, color="green")
        plt.show()
        assert False

    # If this fails, there is a combination of parameters that might
    # trigger a discontinuity in the interpolants, which we don't want.


@slow_hypothesis
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
    # pylint: disable=too-many-locals
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
        wo_ip = interpolate_wo(wo_low, wo_high, t)
        check_table(wo_ip.table)
        ips.append(wo_ip)
        assert 0 < wo_ip.crosspoint() < 1

    # Distances between low and interpolants:
    dists = [(wo_low.table - interp.table)[["PC"]].sum().sum() for interp in ips]
    assert np.isclose(dists[0], 0)

    # Distance between high and the last interpolant
    assert (wo_high.table - ips[-1].table)[["PC"]].sum().sum() < 0.01

    # Distances between low and interpolants:
    dists = [(wo_low.table - interp.table)[["PC"]].sum().sum() for interp in ips]
    print(
        f"Interpolation, mean: {np.mean(dists)}, min: {min(dists)}, "
        f"max: {max(dists)}, std: {np.std(np.diff(dists[1:]))} ip-par-dist: {ip_dist}"
    )
    assert np.isclose(dists[0], 0)  # Reproducing wo_low
    # All curves that are close in parameter t, should be close in sum().sum().
    # That means that diff of the distances should be similar,
    # that is the std.dev of the distances is low:
    ip_dist_std = np.std(np.diff(dists[1:]))  # This number depends on 'h' and 't' range
    # (avoiding the first which reproduces go_low
    if ip_dist_std > 1.0:  # Found by trial and error
        print(f"ip_dist_std: {ip_dist_std}")
        print(dists)
        _, mpl_ax = plt.subplots()
        wo_low.plotpc(mpl_ax=mpl_ax, color="red", logyscale=True)
        wo_high.plotpc(mpl_ax=mpl_ax, color="blue", logyscale=True)
        for interp in ips:
            interp.plotpc(mpl_ax=mpl_ax, color="green", logyscale=True)
        plt.show()
        assert False

    # If this fails, there is a combination of parameters that might
    # trigger a discontinuity in the interpolants, which we don't want.


@slow_hypothesis
@given(
    st.floats(min_value=0, max_value=0.1),  # swl
    st.floats(min_value=0, max_value=0.1),  # sgcr
    st.floats(min_value=0, max_value=0.1),  # dswlhigh
    st.floats(min_value=0, max_value=0.1),  # dsgcr
    st.floats(min_value=0, max_value=0.3),  # sorg
    st.floats(min_value=0, max_value=0.1),  # dsorg
    st.booleans(),  # sgrononzero
    st.floats(min_value=0.1, max_value=5),  # ng1
    st.floats(min_value=0.01, max_value=1),  # krgend1
    st.floats(min_value=0.1, max_value=5),  # nog1
    st.floats(min_value=0.01, max_value=1),  # kroend1
    st.floats(min_value=0.01, max_value=1),  # kromax1
    st.floats(min_value=0.1, max_value=5),  # ng2
    st.floats(min_value=0.01, max_value=1),  # krgend2
    st.floats(min_value=0.1, max_value=5),  # nog2
    st.floats(min_value=0.01, max_value=1),  # kroend2
    st.floats(min_value=0.01, max_value=1),  # kromax2
)
def test_normalize_nonlinpart_go_hypo(
    swl,
    sgcr,
    dswlhigh,
    dsgcr,
    sorg,
    dsorg,
    sgrononzero,
    ng1,
    krgend1,
    nog1,
    kroend1,
    kromax1,
    ng2,
    krgend2,
    nog2,
    kroend2,
    kromax2,
):
    # pylint: disable=too-many-arguments,too-many-locals
    """Test the normalization code in utils.

    In particular the fill_value argument to scipy has been tuned to
    fulfill this code"""
    kroend1 = min(kroend1, kromax1)
    kroend2 = min(kroend2, kromax2)

    if sgrononzero:
        sgro_low = sgcr
        sgro_high = sgcr + dsgcr
    else:
        sgro_low = 0.0
        sgro_high = 0.0

    go_low = GasOil(swl=swl, sgcr=sgcr, sorg=sorg, sgro=sgro_low)
    go_high = GasOil(
        swl=swl + dswlhigh,
        sgcr=sgcr + dsgcr,
        sgro=sgro_high,
        sorg=max(sorg - dsorg, 0),
    )
    go_low.add_corey_gas(ng=ng1, krgend=krgend1)
    go_high.add_corey_gas(ng=ng2, krgend=krgend2)
    go_low.add_corey_oil(nog=nog1, kroend=kroend1, kromax=kromax1)
    go_high.add_corey_oil(nog=nog2, kroend=kroend2, kromax=kromax2)

    krgn1, kron1 = normalize_nonlinpart_go(go_low)
    assert np.isclose(krgn1(0), 0)
    assert np.isclose(krgn1(1), krgend1)
    assert np.isclose(kron1(0), 0)
    assert np.isclose(kron1(1), kroend1)

    krgn2, kron2 = normalize_nonlinpart_go(go_high)
    assert np.isclose(krgn2(0), 0)
    assert np.isclose(krgn2(1), krgend2)
    assert np.isclose(kron2(0), 0)
    assert np.isclose(kron2(1), kroend2)


def test_normalize_nonlinpart_go():
    """Manual tests for normalize_nonlinpart_go"""
    gasoil = GasOil(swl=0.1, sgcr=0.12, sgro=0.12, sorg=0.05, h=0.05)
    gasoil.add_corey_gas(ng=2.1, krgend=0.9)
    gasoil.add_corey_oil(nog=3, kroend=0.75, kromax=0.8)
    krgn, kron = normalize_nonlinpart_go(gasoil)

    assert np.isclose(krgn(0), 0)
    assert np.isclose(krgn(1), 0.9)

    # kron is normalized on son
    assert np.isclose(kron(0), 0)
    assert np.isclose(kron(1), 0.75)

    # Test with tricky endpoints
    h = 0.01
    gasoil = GasOil(swl=h, sgcr=h, sorg=h, h=h)
    gasoil.add_corey_gas(ng=2.1, krgend=0.9)
    gasoil.add_corey_oil(nog=3, kroend=0.8)
    krgn, kron = normalize_nonlinpart_go(gasoil)
    assert np.isclose(krgn(0), 0.0)
    assert np.isclose(krgn(1), 0.9)
    assert np.isclose(kron(0), 0)
    assert np.isclose(kron(1), 0.8)

    # Test again with zero endpoints:
    gasoil = GasOil(swl=0, sgcr=0, sorg=0, h=0.01)
    gasoil.add_corey_gas(ng=2.1, krgend=0.9)
    gasoil.add_corey_oil(nog=3, kroend=0.8)
    krgn, kron = normalize_nonlinpart_go(gasoil)
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
    krgn, kron = normalize_nonlinpart_go(gasoil)
    # These go well still, since we are at zero
    assert np.isclose(krgn(0), 0.0)
    assert np.isclose(kron(0), 0)
    assert np.isclose(kron(1), 0.8)
    # These do not match when endpoints are wrong
    assert not np.isclose(krgn(1), 0.6)

    # So fix endpoints!
    gasoil.swl = 1 - gasoil.table["SG"].max()
    gasoil.sgcr = gasoil.estimate_sgcr()
    gasoil.sorg = gasoil.estimate_sorg()
    # Try again
    krgn, kron = normalize_nonlinpart_go(gasoil)
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
    wo_ip = interpolate_wo(wo_low, wo_high, 0.5)

    # kroend at mean swl:
    assert float_df_checker(wo_ip.table, "SW", 0.01, "KROW", (0.6 + 0.7) / 2.0)

    assert float_df_checker(wo_ip.table, "SW", 1, "KRW", 0.71)
    assert float_df_checker(wo_ip.table, "SW", 1 - 0.15, "KRW", 0.5)


def test_ip_go_kroendmax():
    """Test behaviour of kroend/kromax under interpolation, gas condensate modelling"""
    go_low = GasOil(swl=0, sgro=0.1, sgcr=0.1)
    go_high = GasOil(swl=0, sgro=0)
    go_low.add_corey_gas()
    go_low.add_corey_oil(nog=2, kroend=0.5, kromax=1)
    go_high.add_corey_gas()
    go_high.add_corey_oil(nog=2, kroend=1)

    # Interpolate to midpoint between the curves above
    go_ip = interpolate_go(go_low, go_high, 0.5)

    # kro(sg=0) is 1 for all interpolants:
    assert float_df_checker(go_ip.table, "SG", 0.0, "KROG", 1.0)

    # kro(sg=mean(sgro)) = mean kroeend
    assert float_df_checker(go_ip.table, "SG", (0 + 0.1) / 2.0, "KROG", 0.75)

    assert np.isclose(go_ip.estimate_sgro(), (0 + 0.1) / 2.0)


def test_ip_go_kroend():
    """Test behaviour of kroend under interpolation"""
    go_low = GasOil(swl=0, sgcr=0.1, sorg=0.2)
    go_low.add_corey_gas(ng=2, krgend=0.5, krgmax=0.7)
    go_low.add_corey_oil(nog=2, kroend=0.6)

    go_high = GasOil(swl=0.02, sgcr=0.05, sorg=0.1)
    go_high.add_corey_gas(ng=2, krgend=0.5, krgmax=0.72)
    go_high.add_corey_oil(nog=2, kroend=0.7)

    # Interpolate to midpoint between the curves above
    go_ip = interpolate_go(go_low, go_high, 0.5)

    # kroend at sg zero:
    assert float_df_checker(go_ip.table, "SG", 0.0, "KROG", (0.6 + 0.7) / 2.0)

    assert np.isclose(go_ip.swl, 0.01)
    assert np.isclose(go_ip.sorg, 0.15)

    # krgmax at 1 - swl:
    assert float_df_checker(go_ip.table, "SG", 1 - go_ip.swl, "KRG", 0.71)
    # krgend at 1 - swl - sorg
    assert float_df_checker(go_ip.table, "SG", 1 - go_ip.swl - go_ip.sorg, "KRG", 0.5)

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
    go_ip = interpolate_go(go_low, go_high, 0.5, h=0.1)

    assert float_df_checker(go_ip.table, "SG", 0.0, "KROG", (0.6 + 0.7) / 2.0)

    # Activate these line to see a bug, interpolation_go
    # does not honor krgendanchorA:
    # _, mpl_ax = plt.subplots()
    # go_low.plotkrgkrog(mpl_ax=mpl_ax, color="red")
    # go_high.plotkrgkrog(mpl_ax=mpl_ax, color="blue")
    # go_ip.plotkrgkrog(mpl_ax=mpl_ax, color="green")
    # plt.show()

    # krgmax is irrelevant, krgend is used here:
    assert float_df_checker(go_ip.table, "SG", 1 - 0.01, "KRG", 0.5)

    # Also check that estimated new sgcr is between the inputs:
    assert 0.05 <= go_ip.estimate_sgcr() <= 0.1

    # Do we get into trouble if krgendanchor is different in low and high?
    go_low = GasOil(swl=0, sgcr=0.1, sorg=0.2, krgendanchor="sorg")
    go_low.add_corey_gas(ng=2, krgend=0.5, krgmax=0.7)
    go_low.add_corey_oil(nog=2, kroend=0.6)

    go_high = GasOil(swl=0.02, sgcr=0.05, sorg=0.1, krgendanchor="")
    go_high.add_corey_gas(ng=2, krgend=0.5)
    go_high.add_corey_oil(nog=2, kroend=0.7)

    # Interpolate to midpoint between the curves above
    go_ip = interpolate_go(go_low, go_high, 0.5)

    assert float_df_checker(go_ip.table, "SG", 0.0, "KROG", (0.6 + 0.7) / 2.0)

    # max(krg) is here avg of krgmax and krgend from the differnt tables:
    assert float_df_checker(go_ip.table, "SG", 1 - 0.01, "KRG", 0.6)

    # krgend at 1 - swl - sorg, non-trivial expression, so a numerical
    # value is used here in the test:
    assert float_df_checker(go_ip.table, "SG", 1 - 0.01 - 0.15, "KRG", 0.4491271)

    # krog-zero at 1 - swl - sorg:
    assert float_df_checker(go_ip.table, "SG", 1 - 0.01 - 0.15, "KROG", 0)


@slow_hypothesis
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
    # pylint: disable=too-many-arguments,too-many-locals
    """Test many possible combinations of interpolation between two
    Corey gasoil curves, looking for numerical corner cases"""
    h = 0.01
    go_low = GasOil(swl=swl, sgcr=sgcr, sorg=sorg, h=h)
    go_high = GasOil(
        swl=swl + dswlhigh, sgcr=sgcr + dsgcr, sorg=max(sorg - dsorg, 0), h=h
    )
    go_low.add_corey_gas(ng=ng_l, krgend=krgend_l)
    go_high.add_corey_gas(ng=ng_h, krgend=krgend_h)
    go_low.add_corey_oil(nog=nog_l, kroend=kroend_l)
    go_high.add_corey_oil(nog=nog_h, kroend=kroend_h)
    ips = []
    ip_dist = 0.05
    for t in np.arange(0, 1 + ip_dist, ip_dist):
        go_ip = interpolate_go(go_low, go_high, t)
        check_table(go_ip.table)
        ips.append(go_ip)
        assert 0 < go_ip.crosspoint() < 1

        # sgcr is non-trivial, if exponents are high, an effective sgcr might
        # be larger than the value used to initialize the curve. Try to be
        # permissive enough here. This can even cause the interpolant to be
        # outside the low-high envelope, but it is the way it is supposed to
        # be when sgcr is interpolated separately.
        sgcr_low = min(
            go_low.sgcr, go_low.estimate_sgcr(), go_high.sgcr, go_high.estimate_sgcr()
        )
        sgcr_high = max(
            go_low.sgcr, go_low.estimate_sgcr(), go_high.sgcr, go_high.estimate_sgcr()
        )

        sgcr_ip = go_ip.estimate_sgcr()

        sgcr_lower_bound_ok = sgcr_low - h - epsilon < sgcr_ip
        sgcr_upper_bound_ok = sgcr_ip < sgcr_high + h + epsilon

        assert sgcr_lower_bound_ok
        assert sgcr_upper_bound_ok

    # Distances between low and interpolants:
    dists = [
        (go_low.table - interp.table)[["KRG", "KROG"]].sum().sum() for interp in ips
    ]
    print(
        f"Interpolation, mean: {np.mean(dists)}, min: {min(dists)}, "
        f"max: {max(dists)}, std: {np.std(np.diff(dists[1:]))} ip-par-dist: {ip_dist}"
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
        print(f"ip_dist_std: {ip_dist_std}")
        print(dists)
        _, mpl_ax = plt.subplots()
        go_low.plotkrgkrog(mpl_ax=mpl_ax, color="red")
        go_high.plotkrgkrog(mpl_ax=mpl_ax, color="blue")
        for interp in ips:
            interp.plotkrgkrog(mpl_ax=mpl_ax, color="green")
        plt.show()
        assert False

    # If this fails, there is a combination of parameters that might
    # trigger a discontinuity in the interpolants, which we don't want.


def test_interpolations_go_fromtable():
    """Test based on bug exposed in pyscal 0.6.1, where sgcr
    was underestimated in interpolations following add_fromtable().
    """
    base = pd.DataFrame(
        columns=["SG", "KRG", "KROG"],
        data=[
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.2, 0.0, 1.0],  # sgcr
            [0.3, 0.1, 0.9],
            [0.8, 0.8, 0.0],  # sorg
            [0.9, 0.9, 0.0],
            [1.0, 1.0, 0.0],
        ],
    )
    opt = pd.DataFrame(
        columns=["SG", "KRG", "KROG"],
        data=[
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.3, 0.0, 1.0],
            [0.4, 0.1, 0.2],  # sgcr
            [0.9, 0.9, 0.0],  # sorg
            [0.95, 0.95, 0.0],
            [1.0, 1.0, 0.0],
        ],
    )
    go_base = GasOil(h=0.01)
    go_base.add_fromtable(base)
    assert np.isclose(go_base.estimate_sgcr(), 0.2)
    assert np.isclose(go_base.estimate_sorg(), 0.2)
    go_opt = GasOil(h=0.01)
    go_opt.add_fromtable(opt)
    assert np.isclose(go_opt.estimate_sgcr(), 0.3)
    assert np.isclose(go_opt.estimate_sorg(), 0.1)

    go_ip = interpolate_go(go_base, go_opt, 0.5, h=0.01)
    assert np.isclose(go_ip.estimate_sgcr(), 0.25)
    assert np.isclose(go_ip.estimate_sorg(), 0.15)


def test_interpolations_wo_fromtable():
    """Analog test as test_interpolations_go_fromtable().

    Pyscal 0.6.1 and earlier fails this test sorw.
    """
    base = pd.DataFrame(
        columns=["SW", "KRW", "KROW"],
        data=[
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.2, 0.0, 1.0],  # swcr
            [0.3, 0.1, 0.9],
            [0.8, 0.8, 0.0],  # sorw
            [0.9, 0.9, 0.0],
            [1.0, 1.0, 0.0],
        ],
    )
    opt = pd.DataFrame(
        columns=["SW", "KRW", "KROW"],
        data=[
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.3, 0.0, 1.0],
            [0.4, 0.1, 0.2],  # swcr
            [0.9, 0.9, 0.0],  # sorw
            [0.95, 0.95, 0.0],
            [1.0, 1.0, 0.0],
        ],
    )
    wo_base = WaterOil(h=0.01)
    wo_base.add_fromtable(base)
    assert np.isclose(wo_base.estimate_swcr(), 0.2)
    assert np.isclose(wo_base.estimate_sorw(), 0.2)
    wo_opt = WaterOil(h=0.01)
    wo_opt.add_fromtable(opt)
    assert np.isclose(wo_opt.estimate_swcr(), 0.3)
    assert np.isclose(wo_opt.estimate_sorw(), 0.1)

    wo_ip = interpolate_wo(wo_base, wo_opt, 0.5, h=0.01)
    assert np.isclose(wo_ip.estimate_swcr(), 0.25)
    assert np.isclose(wo_ip.estimate_sorw(), 0.15)


def test_interpolations_wo_fromtable_socr():
    """Test that socr is correctly handled through
    both add_fromtable() and interpolation."""
    base = pd.DataFrame(
        columns=["SW", "KRW", "KROW"],
        data=[
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.2, 0.0, 1.0],  # swcr
            [0.3, 0.1, 0.9],
            [0.7, 0.3, 0.0],  # socr
            [0.8, 0.8, 0.0],  # sorw
            [0.9, 0.9, 0.0],
            [1.0, 1.0, 0.0],
        ],
    )
    opt = pd.DataFrame(
        columns=["SW", "KRW", "KROW"],
        data=[
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.3, 0.0, 1.0],
            [0.4, 0.1, 0.2],  # swcr
            [0.9, 0.9, 0.0],  # sorw == socr
            [0.95, 0.95, 0.0],
            [1.0, 1.0, 0.0],
        ],
    )
    wo_base = WaterOil(h=0.01)
    wo_base.add_fromtable(base)
    assert np.isclose(wo_base.estimate_swcr(), 0.2)
    assert np.isclose(wo_base.estimate_sorw(), 0.2)
    assert np.isclose(wo_base.estimate_socr(), 0.3)
    wo_opt = WaterOil(h=0.01)
    wo_opt.add_fromtable(opt)
    assert np.isclose(wo_opt.estimate_swcr(), 0.3)
    assert np.isclose(wo_opt.estimate_sorw(), 0.1)
    assert np.isclose(wo_opt.estimate_socr(), 0.1)

    wo_ip = interpolate_wo(wo_base, wo_opt, 0.5, h=0.01)
    assert np.isclose(wo_ip.estimate_swcr(), 0.25)
    assert np.isclose(wo_ip.estimate_sorw(), 0.15)
    assert np.isclose(wo_ip.estimate_socr(), 0.2)


@given(
    st.floats(min_value=0, max_value=0.1),  # sgro_low
    st.floats(min_value=0, max_value=0.1),  # sgro_high
)
def test_gascond_interpolation(sgro_low, sgro_high):
    """sgro is required to be either 0 or sgcr, and interpolations
    will crash when this is not the case. This test validates that
    we can let sgro and sgcr go to zero, and that we always are able
    to interpolate without crashes."""
    go_low = GasOil(sgro=sgro_low, sgcr=sgro_low)
    go_high = GasOil(sgro=sgro_high, sgcr=sgro_high)
    go_low.add_corey_gas()
    go_low.add_corey_oil()
    go_high.add_corey_gas()
    go_high.add_corey_oil()

    go_ip = interpolate_go(go_low, go_high, parameter=0.5)
    check_table(go_ip.table)


def test_gasoil_gascond_fails():
    """Interpolation between an object for gas condendensate (sgro > 0)
    and an object with sgro = 0 (regular gas-oil) should fail"""
    gascond = GasOil(sgro=0.1, sgcr=0.1)
    gasoil = GasOil(sgro=0.0, sgcr=0.1)

    gasoil.add_corey_gas()
    gasoil.add_corey_oil()
    gascond.add_corey_gas()
    gascond.add_corey_oil()

    # interpolation parameter at zero is no interpolation, just lookup,
    # so it works fine:
    check_table(interpolate_go(gasoil, gascond, parameter=0.0).table)
    check_table(interpolate_go(gasoil, gascond, parameter=epsilon).table)
    check_table(interpolate_go(gasoil, gascond, parameter=10 * epsilon).table)

    with pytest.raises(ValueError, match="Interpolated sgro"):
        interpolate_go(gasoil, gascond, parameter=100 * epsilon)

    with pytest.raises(ValueError, match="Interpolated sgro"):
        interpolate_go(gasoil, gascond, parameter=0.5)

    check_table(interpolate_go(gasoil, gascond, parameter=1.0 - epsilon).table)
    check_table(interpolate_go(gasoil, gascond, parameter=1.0).table)


def test_endpointpitfalls_in_interpolation_socr():
    """Demonstration of behaviour and pitfalls around endpoints (socr in particular)
    when initializing objects through discretization (add_fromtable()) and how it will
    affect interpolation.
    """
    wo_low = WaterOil(swl=0.1, sorw=0.12)
    wo_low.add_LET_water(l=2, e=2, t=1)
    wo_low.add_LET_oil(l=5, e=5, t=1)

    assert wo_low.socr == wo_low.sorw  # By initialization

    # The KROW values approach zero faster than KRW approaches 1,
    # which makes the tabulated values indicate socr > sorw.
    # This is correct behaviour given finite floating point accuracy.
    assert wo_low.estimate_socr() > wo_low.estimate_sorw()  # difference is around 0.01
    assert np.isclose(wo_low.estimate_sorw(), 0.12)  # correct estimate

    # A not-so-extreme curve-set
    wo_high = WaterOil(swl=0.1, sorw=0.12)
    wo_high.add_LET_water(l=1, e=2, t=1)
    wo_high.add_LET_oil(l=1, e=2, t=1)
    assert wo_low.socr == wo_low.sorw
    # No surprises:
    assert np.isclose(wo_high.estimate_socr(), wo_high.estimate_sorw())

    # If the original wo_low object is then reinitialized through its own
    # discretization (e.g. via a SWOF file on disk and then back again, or via
    # its table property and then add_fromtable()), the original knowledge that
    # socr was defaulted and thus equal to sorw is lost and the best pyscal can
    # do is to use the estimators
    wo_low_dataframe = wo_low.table.copy()
    wo_low_fromtable = WaterOil(wo_low_dataframe["SW"].min())
    wo_low_fromtable.add_fromtable(wo_low_dataframe)
    # The properties of the WaterOil object are now the estimated values.
    assert wo_low_fromtable.socr > wo_low_fromtable.sorw

    # Interpolation from the original low object or the discretized low object
    # will be slightly different:
    wo_ip_original = interpolate_wo(wo_low, wo_high, 0.5)
    wo_ip_throughdiscretization = interpolate_wo(wo_low_fromtable, wo_high, 0.5)
    # The right end of the curves will have slightly lower values due to
    # the interpreteded socr for wo_ip_throughdiscretization:
    assert (
        wo_ip_original.table["KROW"].tail(20).sum()
        > wo_ip_throughdiscretization.table["KROW"].tail(20).sum()
    )
    # But it can be circumvented by explicitly stating the endpoints when
    # using add_fromtable()
    wo_low_fromtable_fixed = WaterOil(wo_low_dataframe["SW"].min())
    wo_low_fromtable_fixed.add_fromtable(wo_low_dataframe, sorw=0.12, socr=0.12)
    wo_ip_throughdiscretization_fixed = interpolate_wo(
        wo_low_fromtable_fixed, wo_high, 0.5
    )
    assert np.isclose(
        wo_ip_original.table["KROW"].tail(20).sum(),
        wo_ip_throughdiscretization_fixed.table["KROW"].tail(20).sum(),
    )

    # Plotting the minor discrepancy (zoom in to see the differences)
    # _, mpl_ax = pyplot.subplots()
    # wo_ip_original.plotkrwkrow(mpl_ax, color="green", alpha=0.6)
    # wo_ip_throughdiscretization.plotkrwkrow(mpl_ax, color="red", alpha=0.6)
    # pyplot.show()
