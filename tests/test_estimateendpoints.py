"""Test module endpoints computed by WaterOil"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from hypothesis import given
import hypothesis.strategies as st

from pyscal import WaterOil, GasOil
from pyscal.constants import EPSILON as epsilon


def test_swcr():
    """Test that we can locate swcr with a certain accuracy"""
    wateroil = WaterOil(swcr=0.3, swl=0.1, sorw=0.1, h=0.05)
    wateroil.add_corey_oil(now=2, kroend=0.8)
    wateroil.add_corey_water(nw=2, krwend=0.6)
    est_sorw = wateroil.estimate_sorw()
    est_swcr = wateroil.estimate_swcr()

    assert np.isclose(est_sorw, 0.1)
    assert np.isclose(est_swcr, 0.3)


def test_swcrsorw():
    """Test estimate_sorw/swcr for some manually set up cases"""
    # swcr, sorw, h:
    testtuples = [
        (0, 0.3, 0.1),
        (0.01, 0.2, 0.1),
        (0.3, 0.1, 0.1),
        (0.0, 0.0, 0.1),
        (0.00001, 0.9, 0.1),
        (0.0001, 0.99, 0.1),
    ]

    for testtuple in testtuples:
        real_swcr = testtuple[0]
        real_sorw = testtuple[1]
        h = testtuple[2]
        wateroil = WaterOil(swcr=real_swcr, sorw=real_sorw, h=h)
        wateroil.add_corey_oil(now=2, kroend=0.8)
        wateroil.add_corey_water(nw=2, krwend=0.9)
        est_sorw = wateroil.estimate_sorw()
        mis = abs(est_sorw - real_sorw)
        print("Testing sorw={}, swcr={} on h={}".format(real_sorw, real_swcr, h))
        if mis > 0.01:
            print("Missed, estimated was {}".format(est_sorw))
        assert mis < h + epsilon  # Can't guarantee better than h.
        est_swcr = wateroil.estimate_swcr()
        mis_swcr = abs(est_swcr - real_swcr)
        if mis_swcr > 0.0:
            print("Missed swcr, estimate was {}".format(est_swcr))
        assert mis_swcr < h + epsilon


def test_sorg():
    """Test estimate_sorg for some manually set up cases"""
    # sorg, sgcr, h, swl:
    testtuples = [
        (0.3, 0.01, 0.1, 0.1),
        (0.2, 0, 0.05, 0.0),
        (0.1, 0.3, 0.01, 0.5),
        (0.0, 0, 0.1, 0.000001),
        (0.9, 0.000001, 0.1, 0),
        (0.4, 0, 0.1, 0.2),
    ]

    for testtuple in testtuples:
        real_sorg = testtuple[0]
        real_sgcr = testtuple[1]
        h = testtuple[2]
        swl = testtuple[3]
        gasoil = GasOil(sgcr=0.03, sorg=real_sorg, h=h, swl=swl)
        gasoil.add_corey_oil(nog=2)
        gasoil.add_corey_gas(ng=2, krgend=0.9)
        print("Testing sorg={} on h={}, swl={}".format(real_sorg, h, swl))
        est_sorg = gasoil.estimate_sorg()
        mis = abs(est_sorg - real_sorg)
        if mis > 0.01:
            print("Missed, estimated was {}".format(est_sorg))
        assert mis < h + epsilon  # Can't guarantee better than h.

        # If krgendanchor is not sorg (default), then krg cannot be used
        # and the GasOil object will resort to using krog. Should work
        # when now=2 but might not always work for all kinds of LET parameters.
        gasoil = GasOil(sorg=real_sorg, h=h, swl=swl, krgendanchor="")
        gasoil.add_corey_oil(nog=2)
        gasoil.add_corey_gas(ng=2, krgend=0.8)
        est_sorg = gasoil.estimate_sorg()
        print("Estimated to {}".format(est_sorg))
        mis = abs(est_sorg - real_sorg)
        assert mis < h + epsilon

        # Test sgcr:
        gasoil = GasOil(
            sorg=real_sorg, sgcr=real_sgcr, h=h, swl=swl, krgendanchor="sorg"
        )
        gasoil.add_corey_oil(nog=2, kroend=0.8)
        gasoil.add_corey_gas(ng=2, krgend=0.8)
        est_sgcr = gasoil.estimate_sgcr()
        mis = abs(est_sgcr - real_sgcr)
        assert mis < h + epsilon


def test_linear_corey():
    """Test how the estimate_sorw works when Corey is linear"""
    # Piecewise linear, this we can detect:
    wateroil = WaterOil(swl=0, h=0.01, sorw=0.3)
    wateroil.add_corey_water(nw=1, krwend=0.8)
    est_sorw = wateroil.estimate_sorw()
    assert np.isclose(est_sorw, 0.3)

    # Piecewise linear, this we can detect:
    wateroil = WaterOil(swl=0, h=0.01, sorw=0.5)
    wateroil.add_corey_water(nw=1, krwend=1)
    est_sorw = wateroil.estimate_sorw()
    assert np.isclose(est_sorw, 0.5)

    # This one is impossible to deduct sorw from, because
    # the entire krw curve is linear.
    wateroil = WaterOil(swl=0, h=0.01, sorw=0.5)
    wateroil.add_corey_water(nw=1, krwend=0.5)
    wateroil.add_corey_oil(now=2)
    est_sorw = wateroil.estimate_sorw()
    # The code will return sw = swl + h by design:
    # Unless information is in the krow table, anything better is impossible
    assert np.isclose(est_sorw, 1 - 0.01)


@given(
    st.floats(min_value=0.001, max_value=0.2),
    st.floats(min_value=0, max_value=0.5),
    st.floats(min_value=0.5, max_value=3),
)
def test_sorw_hypo(h, sorw, nw):
    """Test estimate_sorw extensively"""
    wateroil = WaterOil(sorw=sorw, h=h)
    wateroil.add_corey_oil(now=2)
    wateroil.add_corey_water(nw=nw)
    est_sorw = wateroil.estimate_sorw()
    est_error = abs(sorw - est_sorw)
    assert not np.isnan(est_sorw)
    error_requirement = h + 10000 * epsilon

    if abs(1 - nw) < 0.1:
        # When the Corey curve is almost linear, we don't
        # expect sorw estimation to be robust
        error_requirement = 3 * h
    if est_error > error_requirement:
        print("h = {}, sorw = {}".format(h, sorw))
        print("estimated sorw = {}".format(est_sorw))
        print(wateroil.table)

    # We don't bother to tune the code better than this:
    assert est_error <= error_requirement
