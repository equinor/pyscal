"""Test module for the saturation ranges in WaterOil objects"""

import hypothesis.strategies as st
from hypothesis import given

from pyscal import WaterOil
from pyscal.constants import EPSILON, SWINTEGERS
from pyscal.utils.testing import check_table, float_df_checker


@given(
    st.floats(),
    st.floats(),
    st.floats(),
    st.floats(),
    st.floats(),
    st.floats(min_value=-0.1, max_value=2),
    st.text(),
)
def test_wateroil_random(swirr, swl, swcr, sorw, socr, h, tag):
    """Shoot wildly with arguments, the code should throw ValueError
    or AssertionError when input is invalid, but we don't want other crashes"""
    try:
        WaterOil(swirr=swirr, swl=swl, swcr=swcr, sorw=sorw, socr=socr, h=h, tag=tag)
    except (ValueError, AssertionError):
        pass


@given(
    st.floats(min_value=0, max_value=0.1),
    st.floats(min_value=0, max_value=0.15),
    st.floats(min_value=0, max_value=0.3),
    st.floats(min_value=0, max_value=0.3),
    st.floats(min_value=0, max_value=0.05),
    st.floats(min_value=0.01, max_value=0.2),
    st.text(),
)
def test_wateroil_normalization(swirr, swl, swcr, sorw, socr_add, h, tag):
    """Shoot with more realistic values and test normalized saturations"""
    wateroil = WaterOil(
        swirr=swirr, swl=swl, swcr=swcr, sorw=sorw, socr=sorw + socr_add, h=h, tag=tag
    )
    assert not wateroil.table.empty
    assert not wateroil.table.isnull().values.any()

    # Check that son is 1 at swl:
    assert float_df_checker(wateroil.table, "SW", wateroil.swl, "SON", 1)

    # Check that son is 0 at socr:
    if wateroil.socr > h:
        assert float_df_checker(wateroil.table, "SW", 1 - wateroil.socr, "SON", 0)

    # Check that swn is 0 at swcr:
    assert float_df_checker(wateroil.table, "SW", wateroil.swcr, "SWN", 0)
    # Check that swn is 1 at 1 - sorw
    if wateroil.sorw > 1 / SWINTEGERS:
        assert float_df_checker(wateroil.table, "SW", 1 - wateroil.sorw, "SWN", 1)

    # Check that swnpc is 0 at swirr and 1 at 1:
    if wateroil.swirr >= wateroil.swl + h:
        assert float_df_checker(wateroil.table, "SW", wateroil.swirr, "SWNPC", 0)
    else:
        # Let this go, when swirr is too close to swl. We
        # are not guaranteed to have sw=swirr present
        pass

    assert float_df_checker(wateroil.table, "SW", 1.0, "SWNPC", 1)


@given(st.floats(min_value=0, max_value=1 - EPSILON))
def test_wateroil_swir(swirr):
    """Check that the saturation values are valid for all swirr"""
    wateroil = WaterOil(swirr=swirr)
    check_table(wateroil.table)


@given(st.floats(min_value=0, max_value=1 - EPSILON))
def test_wateroil_swl(swl):
    """Check that the saturation values are valid for all swl"""
    wateroil = WaterOil(swl=swl)
    check_table(wateroil.table)


@given(st.floats(min_value=0, max_value=1 - EPSILON))
def test_wateroil_swcr(swcr):
    """Check that the saturation values are valid for all swcr"""
    wateroil = WaterOil(swcr=swcr)
    check_table(wateroil.table)


@given(st.floats(min_value=0, max_value=1 - EPSILON))
def test_wateroil_sorw(sorw):
    """Check that the saturation values are valid for all sorw"""
    wateroil = WaterOil(sorw=sorw)
    check_table(wateroil.table)


@given(st.floats(min_value=0, max_value=1 - EPSILON))
def test_wateroil_socr(socr):
    """Check that the saturation values are valid for all socr"""
    wateroil = WaterOil(socr=socr)
    check_table(wateroil.table)


@given(st.floats(min_value=0, max_value=1), st.floats(min_value=0, max_value=1))
def test_wateroil_dual(param1, param2):
    """Test combination of 2 floats as parameters"""
    try:
        wateroil = WaterOil(swl=param1, sorw=param2)
        check_table(wateroil.table)
        # Will fail when swl > 1 - sorw
    except AssertionError:
        pass

    try:
        wateroil = WaterOil(swl=param1, swirr=param2)
        check_table(wateroil.table)
    except AssertionError:
        pass

    try:
        wateroil = WaterOil(swcr=param1, sorw=param2)
        check_table(wateroil.table)
    except AssertionError:
        pass

    try:
        wateroil = WaterOil(swirr=param1, sorw=param2)
        check_table(wateroil.table)
    except AssertionError:
        pass
