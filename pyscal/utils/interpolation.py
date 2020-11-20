"""Utility function for pyscal
"""

import logging

import pandas as pd
from scipy.interpolate import interp1d

import pyscal
from ..constants import SWINTEGERS


logger = logging.getLogger(__name__)


def normalize_nonlinpart_wo(curve):
    """Make krw and krow functions that evaluate only on the
    (potentially) nonlinear part of the relperm curves, and with
    a normalized argument (0,1) on that interval.

    For a WaterOil krw curve, the nonlinear part is from swcr to sorw.
    swcr is mapped to zero, and 1 - sorw is mapped to 1. Then there is
    an assumed linear part from sorw to 1 which we ignore here.

    For a WaterOil krow curve, the nonlinear part
    is from 1 - sorw (mapped to zero) to swl (mapped to 1).

    These endpoints must be known the the WaterOil object coming in (the object
    can determine them using functions 'estimate_sorw()' and 'estimate_swcr()'

    If the entire curve is linear, it will not matter for this function, because
    this function only deals with the presumably known endpoints.

    Arguments:
        curve (WaterOil): incoming oilwater curve set (krw and krow)

    Returns:
        tuple of lambda functions. The first will evaluate krw on
            the normalized Sw interval [0,1], the second will
            evaluate krow on the normalized So interval [0,1].
    """
    krw_interp = interp1d(
        curve.table["sw"],
        curve.table["krw"],
        kind="linear",
        bounds_error=False,
        fill_value=(0.0, curve.table["krw"].max()),
    )

    # The internal dataframe might contain normalized
    # saturation values, but we do not want to assume they
    # are there or even correct, therefore we effectively
    # recalculate them
    def sw_fn(swn):
        return curve.swcr + swn * (1.0 - curve.swcr - curve.sorw)

    def krw_fn(swn):
        return krw_interp(sw_fn(swn))

    kro_interp = interp1d(
        1.0 - curve.table["sw"],
        curve.table["krow"],
        kind="linear",
        bounds_error=False,
        fill_value=(0.0, curve.table["krow"].max()),
    )

    def so_fn(son):
        return curve.sorw + son * (1.0 - curve.sorw - curve.swl)

    def kro_fn(son):
        return kro_interp(so_fn(son))

    return (krw_fn, kro_fn)


def normalize_nonlinpart_go(curve):
    """Make krg and krog functions that evaluates only on the
    (potentially) nonlinear part of the relperm curves, and with
    a normalized argument (0,1) on that interval.

    For a GasOil krg curve, the nonlinear part
    is from sgcr to sorg. sgcr is mapped to sg=zero, and sg=1 - sorg - swl is mapped
    to 1. Then there is an assumed linear part from sorg to 1 which we ignore here.

    For a GasOil krow curve, the nonlinear part
    is from 1 - sorg (mapped to zero) to sg=0 (mapped to 1).

    These endpoints must be known the the GasOil object coming in (the object
    can determine them using functions 'estimate_sorg()' and 'estimate_sgcr()'

    If the entire curve is linear, it will not matter for this function, because
    this function only deals with the presumably known endpoints.

    Arguments:
        curve (GasOil): incoming gasoil curve set (krg and krog)

    Returns:
        tuple of functions. The first will evaluate krg on
            the normalized Sg interval [0,1], the second will
            evaluate krog on the normalized So interval [0,1].
    """
    krg_interp = interp1d(
        curve.table["sg"],
        curve.table["krg"],
        kind="linear",
        bounds_error=False,
        fill_value=(0.0, curve.table["krg"].max()),
    )

    # The internal dataframe might contain normalized
    # saturation values, but we do not want to assume they
    # are there or even correct, therefore we effectively
    # recalculate them
    def sg_fn(sgn):
        return curve.sgcr + sgn * (1.0 - curve.swl - curve.sgcr - curve.sorg)

    def krg_fn(sgn):
        return krg_interp(sg_fn(sgn))

    kro_interp = interp1d(
        1.0 - curve.table["sg"],
        curve.table["krog"],
        kind="linear",
        bounds_error=False,
        fill_value=(0.0, curve.table["krog"].max()),
    )

    def so_fn(son):
        return curve.swl + curve.sorg + son * (1.0 - curve.swl - curve.sorg)

    def kro_fn(son):
        return kro_interp(so_fn(son))

    return (krg_fn, kro_fn)


def normalize_pc(curve):
    """Normalize the capillary pressure curve.

    This is only normalized with respect to the
    smallest and largest saturation present in the table,
    not to the could-be-uncertain swirr
    that the object could contain, because we then have
    to make assumptions on the equations used to generate
    the data in the table.

    Args:
        curve (WaterOil or GasOil): An object with a table with a pc column

    Returns:
        a lambda function that will evaluate pc on
        the normalized interval [0,1]
    """
    if isinstance(curve, pyscal.WaterOil):
        sat_col = "sw"
    elif isinstance(curve, pyscal.GasOil):
        sat_col = "sg"
    else:
        raise ValueError("Only WaterOil or GasOil allowed as argument")

    if "pc" not in curve.table:
        # Return a dummy zero lambda
        return lambda sxn: 0

    min_pc = curve.table["pc"].min()
    max_pc = curve.table["pc"].max()
    min_sx = curve.table[sat_col].min()
    max_sx = curve.table[sat_col].max()

    pc_interp = interp1d(
        curve.table[sat_col],
        curve.table["pc"],
        kind="linear",
        bounds_error=False,
        fill_value=(max_pc, min_pc),  # This gives constant extrapolation outside [0, 1]
    )

    # Map from normalized value to real saturation domain:
    def sx_fn(sxn):
        return curve.table[sat_col].min() + sxn * (max_sx - min_sx)

    def pc_fn(sxn):
        return pc_interp(sx_fn(sxn))

    return pc_fn


def _interpolate_tags(low, high, parameter, tag):
    """Preserve tag/comment. Depending on context, the
    interpolation parameter may or may not make sense. In a SCALrecommendation
    interpolation, the new tag should be constructed in the caller of this function.
    because of the way the parameter value is handled.

    This function is used by interpolate_wo and interpolate_go

    Args:
        low (WaterOil or GasOil): low case in interpolation
        high (WaterOil or GasOil): high case
        parameter (float): between 0 and 1
        tag (str): If not none, this is directly returned.

    Returns:
        string, a "computed" tag if a tag is not directly supplied
    """
    if tag is None:
        if low.tag == high.tag:
            if low.tag:
                return "Interpolated to {} in {}".format(parameter, low.tag)
            # No tags defined.
            return "Interpolated to {}".format(parameter)
        return "Interpolated to {} between {} and {}".format(
            parameter, low.tag, high.tag
        )
    return tag


def interpolate_wo(wo_low, wo_high, parameter, h=0.01, tag=None):
    """Interpolates between two water-oil curves.

    The saturation endpoints for the curves must be known
    by the objects. They can be estimated by estimate_sorw() etc.
    or can be set manually for finer control.

    The interpolation algorithm is different left and right
    for saturation endpoints, and saturation endpoints are
    interpolated individually.

    Arguments:
        wo_low (WaterOil): a "low" case
        wo_high (WaterOil): a "high" case
        parameter (float): Between 0 and 1. 0 will return the low case, 1 will return
            the high case. Any number in between will return an interpolated curve
        h (float): Saturation step-size in interpolant. If defaulted, a value
            smaller than in the input curves are used, to preserve information.
        tag (string): Tag to associate to the constructed object. If None
            it will be automatically filled. Set to empty string to ensure no tag.
    Returns:
        A new oil-water curve

    """
    # Warning: Almost code duplication with corresponding _go function

    assert isinstance(wo_low, pyscal.WaterOil)
    assert isinstance(wo_high, pyscal.WaterOil)

    assert 0 <= parameter <= 1
    # Extrapolation is refused, but perhaps later implemented with truncation to (0,1)

    # Constructs functions that works on normalized saturation interval
    krw1, kro1 = normalize_nonlinpart_wo(wo_low)
    krw2, kro2 = normalize_nonlinpart_wo(wo_high)
    pc1 = normalize_pc(wo_low)
    pc2 = normalize_pc(wo_high)

    # Construct a function that can be applied to both relperm values
    # and endpoints
    def weighted_value(a, b):
        return a * (1.0 - parameter) + b * parameter

    # Interpolate saturation endpoints
    swl_new = weighted_value(wo_low.swl, wo_high.swl)
    swcr_new = weighted_value(wo_low.swcr, wo_high.swcr)
    sorw_new = weighted_value(wo_low.sorw, wo_high.sorw)

    # Interpolate kr at saturation endpoints
    krwmax_new = weighted_value(wo_low.table["krw"].max(), wo_high.table["krw"].max())
    krwend_new = weighted_value(krw1(1), krw2(1))
    kroend_new = weighted_value(kro1(1), kro2(1))

    # Construct the new WaterOil object, with interpolated
    # endpoints:
    wo_new = pyscal.WaterOil(swl=swl_new, swcr=swcr_new, sorw=sorw_new, h=h)

    # Add interpolated relperm data in nonlinear parts:
    wo_new.table["krw"] = weighted_value(
        krw1(wo_new.table["swn"]), krw2(wo_new.table["swn"])
    )
    wo_new.table["krow"] = weighted_value(
        kro1(wo_new.table["son"]), kro2(wo_new.table["son"])
    )

    wo_new.set_endpoints_linearpart_krw(krwend=krwend_new, krwmax=krwmax_new)
    wo_new.set_endpoints_linearpart_krow(kroend=kroend_new)

    # We need a new fit-for-purpose normalized swnpc, that ignores
    # the initial swnpc (swirr-influenced)
    wo_new.table["swn_pc_intp"] = (wo_new.table["sw"] - wo_new.table["sw"].min()) / (
        wo_new.table["sw"].max() - wo_new.table["sw"].min()
    )
    wo_new.table["pc"] = weighted_value(
        pc1(wo_new.table["swn_pc_intp"]), pc2(wo_new.table["swn_pc_intp"])
    )

    wo_new.tag = _interpolate_tags(wo_low, wo_high, parameter, tag)

    return wo_new


def interpolate_go(go_low, go_high, parameter, h=0.01, tag=None):
    """Interpolates between two gas-oil curves.

    The saturation endpoints for the curves must be known
    by the objects. They can be estimated by estimate_sorg() etc.
    or can be set manually for finer control.

    The interpolation algorithm is different left and right
    for saturation endpoints, and saturation endpoints are
    interpolated individually.

    Arguments:
        go_low (GasOil): a "low" case
        go_high (GasOil): a "high" case
        parameter (float): Between 0 and 1. 0 will return the low case, 1 will return
            the high case. Any number in between will return an interpolated curve
        h (float): Saturation step-size in interpolant. If defaulted, a value
            smaller than in the input curves are used, to preserve information.
        tag (string): Tag to associate to the constructed object. If None
            it will be automatically filled. Set to empty string to ensure no tag.
    Returns:
        A new gas-oil curve

    """
    # Warning: Almost code duplication with corresponding _wo function

    assert isinstance(go_low, pyscal.GasOil)
    assert isinstance(go_high, pyscal.GasOil)

    assert 0 <= parameter <= 1
    # Extrapolation is refused, but perhaps later implemented with truncation to (0,1)

    # Constructs functions that works on normalized saturation interval
    krg1, kro1 = normalize_nonlinpart_go(go_low)
    krg2, kro2 = normalize_nonlinpart_go(go_high)
    pc1 = normalize_pc(go_low)
    pc2 = normalize_pc(go_high)

    # Construct a lambda function that can be applied to both relperm values
    # and endpoints
    def weighted_value(a, b):
        return a * (1.0 - parameter) + b * parameter

    # Interpolate saturation endpoints
    swl_new = weighted_value(go_low.swl, go_high.swl)
    sgcr_new = weighted_value(go_low.sgcr, go_high.sgcr)
    sorg_new = weighted_value(go_low.sorg, go_high.sorg)

    # Interpolate kr at saturation endpoints
    krgmax_new = weighted_value(go_low.table["krg"].max(), go_high.table["krg"].max())
    krgend_new = weighted_value(krg1(1), krg2(1))
    kroend_new = weighted_value(kro1(1), kro2(1))

    # Construct the new GasOil object, with interpolated
    # endpoints:
    go_new = pyscal.GasOil(swl=swl_new, sgcr=sgcr_new, sorg=sorg_new, h=h)

    # Add interpolated relperm data in nonlinear parts:
    go_new.table["krg"] = weighted_value(
        krg1(go_new.table["sgn"]), krg2(go_new.table["sgn"])
    )
    go_new.table["krog"] = weighted_value(
        kro1(go_new.table["son"]), kro2(go_new.table["son"])
    )
    go_new.table["pc"] = weighted_value(
        pc1(go_new.table["sgn"]), pc2(go_new.table["sgn"])
    )

    # We need a new fit-for-purpose normalized sgnpc
    go_new.table["sgn_pc_intp"] = (go_new.table["sg"] - go_new.table["sg"].min()) / (
        go_new.table["sg"].max() - go_new.table["sg"].min()
    )
    go_new.table["pc"] = weighted_value(
        pc1(go_new.table["sgn_pc_intp"]), pc2(go_new.table["sgn_pc_intp"])
    )

    go_new.set_endpoints_linearpart_krog(kroend=kroend_new)

    # Here we should have honored krgendanchor. Check github issue.
    go_new.set_endpoints_linearpart_krg(krgend=krgend_new, krgmax=krgmax_new)

    go_new.tag = _interpolate_tags(go_low, go_high, parameter, tag)

    return go_new


def interpolator(
    tableobject, wo_low, wo_high, parameter, sat="sw", kr1="krw", kr2="krow", pc="pc"
):
    """Interpolates between two curves.

    DEPRECATED FUNCTION!

    The interpolation parameter is 0 through 1,
    irrespective of phases or low-base/base-high/low-high.

    Args:
        tabjeobject (WaterOil or GasOil): A partially setup object where
            relperm and pc columns are to be filled with numbers.
        wo_low (WaterOil or GasOil): "Low" case of interpolation (relates
            to interpolation parameter 0). Must be copies, as they
            will be modified.
        wo_high: Ditto, relates to interpolation parameter 1
        parameter (float): Between 0 and 1, what you want to interpolate to.
        sat (str): Name of the saturation column, typically 'sw' or 'sg'
        kr1 (str): Name of the first relperm column ('krw' or 'krg')
        kr2 (str): Name of the second relperm column ('krow' or 'krog')
        pc (str): Name of the capillary pressure column ('pc')

    Returns:
        None, but modifies the first argument.
    """
    logger.warning("utils.interpolator() is deprecated and will disappear")

    wo_low.table.rename(columns={kr1: kr1 + "_1"}, inplace=True)
    wo_high.table.rename(columns={kr1: kr1 + "_2"}, inplace=True)
    wo_low.table.rename(columns={kr2: kr2 + "_1"}, inplace=True)
    wo_high.table.rename(columns={kr2: kr2 + "_2"}, inplace=True)
    wo_low.table.rename(columns={pc: pc + "_1"}, inplace=True)
    wo_high.table.rename(columns={pc: pc + "_2"}, inplace=True)

    # Result data container:
    satresult = pd.DataFrame(data=tableobject.table[sat], columns=[sat])

    # Merge swresult with wo_low and wo_high, and interpolate all
    # columns in sw:
    intdf = (
        pd.concat([wo_low.table, wo_high.table, satresult], sort=True)
        .set_index(sat)
        .sort_index()
        .interpolate(method="slinear")
        .fillna(method="bfill")
        .fillna(method="ffill")
    )

    # Normalized saturations does not make sense for the
    # interpolant, remove:
    for col in ["swn", "son", "swnpc", "H", "J"]:
        if col in intdf.columns:
            del intdf[col]

    intdf[kr1] = intdf[kr1 + "_1"] * (1 - parameter) + intdf[kr1 + "_2"] * parameter
    intdf[kr2] = intdf[kr2 + "_1"] * (1 - parameter) + intdf[kr2 + "_2"] * parameter
    if pc + "_1" in wo_low.table.columns and pc + "_2" in wo_high.table.columns:
        intdf[pc] = intdf[pc + "_1"] * (1 - parameter) + intdf[pc + "_2"] * parameter
    else:
        intdf[pc] = 0

    # Slice out the resulting sw values and columns. Slicing on
    # floating point indices is not robust so we need to slice on an
    # integer version of the sw column
    tableobject.table["swint"] = list(
        map(int, list(map(round, tableobject.table[sat] * SWINTEGERS)))
    )
    intdf["swint"] = list(map(int, list(map(round, intdf.index.values * SWINTEGERS))))
    intdf = intdf.reset_index()
    intdf.drop_duplicates("swint", inplace=True)
    intdf.set_index("swint", inplace=True)
    intdf = intdf.loc[tableobject.table["swint"].values]
    intdf = intdf[[sat, kr1, kr2, pc]].reset_index()

    # intdf['swint'] = (intdf['sw'] * SWINTEGERS).astype(int)
    # intdf.drop_duplicates('swint', inplace=True)

    # Populate the WaterOil object
    tableobject.table[kr1] = intdf[kr1]
    tableobject.table[kr2] = intdf[kr2]
    tableobject.table[pc] = intdf[pc]
    tableobject.table.fillna(method="ffill", inplace=True)
