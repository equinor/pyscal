"""Utility function for pyscal"""

import logging
from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d

from pyscal import GasOil, WaterOil

logger = logging.getLogger(__name__)


def normalize_nonlinpart_wo(curve: WaterOil) -> Tuple[Callable, Callable]:
    """Make krw and krow functions that evaluate only on the
    (potentially) nonlinear part of the relperm curves, and with
    a normalized argument (0,1) on that interval.

    For a WaterOil krw curve, the nonlinear part is from swcr to sorw.
    swcr is mapped to zero, and 1 - sorw is mapped to 1. Then there is
    an assumed linear part from sorw to 1 which we ignore here.

    For a WaterOil krow curve, the nonlinear part
    is from 1 - sorw (mapped to zero) to swl (mapped to 1).

    These endpoints must be known the the WaterOil object coming in (the object
    can determine them using functions estimate_sorw(), estimate_swcr() and
    estimate_socr()

    If the entire curve is linear, it will not matter for this function, because
    this function only deals with the presumably known endpoints.

    Arguments:
        curve: incoming oilwater curve set (krw and krow)

    Returns:
        tuple of lambda functions. The first will evaluate krw on
        the normalized Sw interval [0,1], the second will
        evaluate krow on the normalized So interval [0,1].
    """
    krw_interp = interp1d(
        curve.table["SW"],
        curve.table["KRW"],
        kind="linear",
        bounds_error=False,
        fill_value=(0.0, curve.table["KRW"].max()),
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
        1.0 - curve.table["SW"],
        curve.table["KROW"],
        kind="linear",
        bounds_error=False,
        fill_value=(0.0, curve.table["KROW"].max()),
    )

    def so_fn(son):
        return curve.socr + son * (1.0 - curve.socr - curve.swl)

    def kro_fn(son):
        return kro_interp(so_fn(son))

    return (krw_fn, kro_fn)


def normalize_nonlinpart_go(curve: GasOil) -> Tuple[Callable, Callable]:
    """Make krg and krog functions that evaluates only on the
    (potentially) nonlinear part of the relperm curves, and with
    a normalized argument (0,1) on that interval.

    For a GasOil krg curve, the nonlinear part
    is from sgcr to sorg. sgcr is mapped to sg=zero, and sg=1 - sorg - swl is mapped
    to 1. Then there is an assumed linear part from sorg to 1 which we ignore here.

    For a GasOil krog curve, the nonlinear part
    is from 1 - sorg (mapped to zero) to sg=sgro (mapped to 1).

    These endpoints must be known the the GasOil object coming in (the object
    can determine them using functions 'estimate_sorg()', 'estimate_sgcr()' and
    'estimate_sgro()'

    If the entire curve is linear, it will not matter for this function, because
    this function only deals with the presumably known endpoints.

    Arguments:
        curve: incoming gasoil curve set (krg and krog)

    Returns:
        tuple of functions. The first will evaluate krg on
        the normalized Sg interval [0,1], the second will
        evaluate krog on the normalized So interval [0,1].
    """
    krg_interp = interp1d(
        curve.table["SG"],
        curve.table["KRG"],
        kind="linear",
        bounds_error=False,
        fill_value=(0.0, curve.table["KRG"].max()),
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
        1.0 - curve.table["SG"],
        curve.table["KROG"],
        kind="linear",
        bounds_error=False,
        fill_value=(0.0, curve.table["KROG"].max()),
    )

    def so_fn(son):
        return (
            curve.swl + curve.sorg + son * (1.0 - curve.swl - curve.sorg - curve.sgro)
        )

    def kro_fn(son):
        return kro_interp(so_fn(son))

    return (krg_fn, kro_fn)


def normalize_pc(curve: Union[WaterOil, GasOil]) -> Callable:
    """Normalize the capillary pressure curve.

    This is only normalized with respect to the
    smallest and largest saturation present in the table,
    not to the could-be-uncertain swirr
    that the object could contain, because we then have
    to make assumptions on the equations used to generate
    the data in the table.

    Args:
        curve: An object with a table with a pc column

    Returns:
        a lambda function that will evaluate pc on
        the normalized interval [0,1]
    """
    if isinstance(curve, WaterOil):
        sat_col = "SW"
    elif isinstance(curve, GasOil):
        sat_col = "SG"
    else:
        raise ValueError("Only WaterOil or GasOil allowed as argument")

    if "PC" not in curve.table:
        # Return a dummy zero lambda
        return lambda sxn: 0

    min_pc = curve.table["PC"].min()
    max_pc = curve.table["PC"].max()
    min_sx = curve.table[sat_col].min()
    max_sx = curve.table[sat_col].max()

    pc_interp = interp1d(
        curve.table[sat_col],
        curve.table["PC"],
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


def _interpolate_tags(
    low: Union[WaterOil, GasOil],
    high: Union[WaterOil, GasOil],
    parameter: float,
    tag: Optional[str],
):
    """Preserve tag/comment. Depending on context, the
    interpolation parameter may or may not make sense. In a SCALrecommendation
    interpolation, the new tag should be constructed in the caller of this function.
    because of the way the parameter value is handled.

    This function is used by interpolate_wo and interpolate_go

    Args:
        low: low case in interpolation
        high: high case
        parameter: between 0 and 1
        tag: If not none, this is directly returned.

    Returns:
        string, a "computed" tag if a tag is not directly supplied
    """
    if tag is None:
        if low.tag == high.tag:
            if low.tag:
                return f"Interpolated to {parameter} in {low.tag}"
            # No tags defined.
            return f"Interpolated to {parameter}"
        return f"Interpolated to {parameter} between {low.tag} and {high.tag}"
    return tag


def interpolate_wo(
    wo_low: WaterOil,
    wo_high: WaterOil,
    parameter: float,
    h: Optional[float] = None,
    tag: Optional[str] = None,
) -> WaterOil:
    """Interpolates between two water-oil curves.

    The saturation endpoints for the curves must be known
    by the objects. They can be estimated by estimate_sorw() etc.
    or can be set manually for finer control.

    The interpolation algorithm is different left and right
    for saturation endpoints, and saturation endpoints are
    interpolated individually.

    Arguments:
        wo_low: a "low" case
        wo_high: a "high" case
        parameter: Between 0 and 1. 0 will return the low case, 1 will return
            the high case. Any number in between will return an interpolated curve
        h: Saturation step-size in interpolant. If defaulted, a value
            smaller than in the input curves are used, to preserve information.
        tag: Tag to associate to the constructed object. If None
            it will be automatically filled. Set to empty string to ensure no tag.
    """
    # Warning: Almost code duplication with corresponding _go function

    assert isinstance(wo_low, WaterOil)
    assert isinstance(wo_high, WaterOil)

    assert 0 <= parameter <= 1
    # Extrapolation is refused, but perhaps later implemented with truncation to (0,1)

    # Running fast mode if both interpolants have fast mode
    fast = wo_low.fast and wo_high.fast

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
    socr_new = weighted_value(wo_low.socr, wo_high.socr)

    # Interpolate kr at saturation endpoints
    krwmax_new = weighted_value(wo_low.table["KRW"].max(), wo_high.table["KRW"].max())
    krwend_new = weighted_value(krw1(1), krw2(1))
    kroend_new = weighted_value(kro1(1), kro2(1))

    # Construct the new WaterOil object, with interpolated
    # endpoints:
    wo_new = WaterOil(
        swl=swl_new, swcr=swcr_new, sorw=sorw_new, socr=socr_new, h=h, fast=fast
    )

    # Add interpolated relperm data in nonlinear parts:
    wo_new.table["KRW"] = weighted_value(
        krw1(wo_new.table["SWN"]), krw2(wo_new.table["SWN"])
    )
    wo_new.table["KROW"] = weighted_value(
        kro1(wo_new.table["SON"]), kro2(wo_new.table["SON"])
    )

    wo_new.set_endpoints_linearpart_krw(krwend=krwend_new, krwmax=krwmax_new)
    wo_new.set_endpoints_linearpart_krow(kroend=kroend_new)

    # We need a new fit-for-purpose normalized swnpc, that ignores
    # the initial swnpc (swirr-influenced)
    wo_new.table["swn_pc_intp"] = (wo_new.table["SW"] - wo_new.table["SW"].min()) / (
        wo_new.table["SW"].max() - wo_new.table["SW"].min()
    )
    wo_new.table["PC"] = weighted_value(
        pc1(wo_new.table["swn_pc_intp"]), pc2(wo_new.table["swn_pc_intp"])
    )

    wo_new.tag = _interpolate_tags(wo_low, wo_high, parameter, tag)

    return wo_new


def interpolate_go(
    go_low: GasOil,
    go_high: GasOil,
    parameter: float,
    h: Optional[float] = None,
    tag: Optional[str] = None,
) -> GasOil:
    """Interpolates between two gas-oil curves.

    The saturation endpoints for the curves must be known
    by the objects. They can be estimated by estimate_sorg() etc.
    or can be set manually for finer control.

    The interpolation algorithm is different left and right
    for saturation endpoints, and saturation endpoints are
    interpolated individually.

    Arguments:
        go_low: a "low" case
        go_high: a "high" case
        parameter: Between 0 and 1. 0 will return the low case, 1 will return
            the high case. Any number in between will return an interpolated curve
        h: Saturation step-size in interpolant. If defaulted, a value
            smaller than in the input curves are used, to preserve information.
        tag: Tag to associate to the constructed object. If None
            it will be automatically filled. Set to empty string to ensure no tag.
    """
    # Warning: Almost code duplication with corresponding _wo function

    assert isinstance(go_low, GasOil)
    assert isinstance(go_high, GasOil)

    assert 0 <= parameter <= 1
    # Extrapolation is refused, but perhaps later implemented with truncation to (0,1)

    # Running fast mode if both interpolants have fast mode
    fast = go_low.fast and go_high.fast

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
    sgro_new = weighted_value(go_low.sgro, go_high.sgro)

    if not (np.isclose(sgro_new, sgcr_new) or np.isclose(sgro_new, 0.0)):
        raise ValueError(
            f"Interpolated sgro ({sgro_new}) not equal "
            f"to zero or interpolated sgcr ({sgcr_new})"
        )

    # Interpolate kr at saturation endpoints
    krgmax_new = weighted_value(go_low.table["KRG"].max(), go_high.table["KRG"].max())
    krgend_new = weighted_value(krg1(1), krg2(1))
    kromax_new = weighted_value(go_low.table["KROG"].max(), go_high.table["KROG"].max())
    kroend_new = weighted_value(kro1(1), kro2(1))

    # Construct the new GasOil object, with interpolated
    # endpoints:
    go_new = GasOil(
        swl=swl_new, sgcr=sgcr_new, sorg=sorg_new, sgro=sgro_new, h=h, fast=fast
    )

    # Add interpolated relperm data in nonlinear parts:
    go_new.table["KRG"] = weighted_value(
        krg1(go_new.table["SGN"]), krg2(go_new.table["SGN"])
    )
    go_new.table["KROG"] = weighted_value(
        kro1(go_new.table["SON"]), kro2(go_new.table["SON"])
    )
    go_new.table["PC"] = weighted_value(
        pc1(go_new.table["SGN"]), pc2(go_new.table["SGN"])
    )

    # We need a new fit-for-purpose normalized sgnpc
    go_new.table["sgn_pc_intp"] = (go_new.table["SG"] - go_new.table["SG"].min()) / (
        go_new.table["SG"].max() - go_new.table["SG"].min()
    )
    go_new.table["PC"] = weighted_value(
        pc1(go_new.table["sgn_pc_intp"]), pc2(go_new.table["sgn_pc_intp"])
    )

    go_new.set_endpoints_linearpart_krog(kroend=kroend_new, kromax=kromax_new)

    # Here we should have honored krgendanchor. Check github issue.
    go_new.set_endpoints_linearpart_krg(krgend=krgend_new, krgmax=krgmax_new)

    go_new.tag = _interpolate_tags(go_low, go_high, parameter, tag)

    return go_new
