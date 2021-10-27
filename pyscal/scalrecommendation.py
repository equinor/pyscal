"""SCALrecommendation, container for low, base and high WaterOilGas objects"""

import copy
from typing import Optional, Set, Type, Union

import numpy as np

from pyscal import GasWater, WaterOilGas, getLogger_pyscal
from pyscal.utils.interpolation import interpolate_go, interpolate_wo

logger = getLogger_pyscal(__name__)


class SCALrecommendation(object):
    """A SCAL recommendation consists of three OilWaterGas objects,
    tagged low, base and high.

    This container exists in order to to interpolation from -1 (low),
    through 0 (base) and to 1 (high).

    Args:
        low: An object representing the low case
        base: An object representing the base case
        high: An object representing the high case
        tag: A string that describes the recommendation. Optional.
    """

    def __init__(
        self,
        low: Union[WaterOilGas, GasWater],
        base: Union[WaterOilGas, GasWater],
        high: Union[WaterOilGas, GasWater],
        tag: Optional[str] = None,
        h: float = 0.01,
    ) -> None:
        """Set up a SCAL recommendation curve set from WaterOilGas objects

        Arguments:
            low: low case
            base: base case
            high: high case
            tag: Describes the recommendation. This string will be used
                as tag strings for the interpolants.
        """

        self.h: float = h
        self.tag: Optional[str] = tag
        self.low: Union[WaterOilGas, GasWater]
        self.base: Union[WaterOilGas, GasWater]
        self.high: Union[WaterOilGas, GasWater]
        self.type: Type

        if (
            isinstance(low, WaterOilGas)
            and isinstance(base, WaterOilGas)
            and isinstance(high, WaterOilGas)
        ):

            self.low = low
            self.base = base
            self.high = high
            self.type = WaterOilGas
        elif (
            isinstance(low, GasWater)
            and isinstance(base, GasWater)
            and isinstance(high, GasWater)
        ):

            self.low = low
            self.base = base
            self.high = high
            self.type = GasWater
        else:
            raise ValueError("Wrong arguments to SCALrecommendation")

        self.fast: bool = False
        if all([self.low.fast, self.base.fast, self.high.fast]):
            self.fast = True
        elif any([self.low.fast, self.base.fast, self.high.fast]):
            self.fast = self.low.fast = self.base.fast = self.high.fast = False
            logger.warning(
                (
                    "One or more of the low/base/high objects are set to be run in "
                    "fast mode, but not all. Fast mode set to false in all objects. "
                    "Code is run in normal mode."
                )
            )

    # User should add capillary pressure explicitly by calling add**
    # on the class objects, or run the following method to add the
    # same to all curves:
    def add_simple_J(
        self,
        a: float = 5.0,
        b: float = -1.5,
        poro_ref: float = 0.25,
        perm_ref: float = 100.0,
        drho: float = 300.0,
        g: float = 9.81,
    ) -> None:
        """Add (identical) simplified J-function to all water-oil
        curves in the SCAL recommendation set"""
        assert self.low.wateroil is not None
        assert self.base.wateroil is not None
        assert self.high.wateroil is not None
        self.low.wateroil.add_simple_J(
            a=a, b=b, poro_ref=poro_ref, perm_ref=perm_ref, drho=drho, g=g
        )
        self.base.wateroil.add_simple_J(
            a=a, b=b, poro_ref=poro_ref, perm_ref=perm_ref, drho=drho, g=g
        )
        self.high.wateroil.add_simple_J(
            a=a, b=b, poro_ref=poro_ref, perm_ref=perm_ref, drho=drho, g=g
        )

    def interpolate(
        self,
        parameter: float,
        parameter2: Optional[float] = None,
        h: Optional[float] = None,
    ) -> Union[WaterOilGas, GasWater]:
        """Interpolate between low, base and high

        Endpoints are located for input curves, and interpolated
        individually. Interpolation for the nonlinear part
        is done on a normalized interval between the endpoints

        Interpolation is linear in relperm-direction, and will thus not be
        linear in log-relperm-direction

        This method returns an WaterOilGas object which can be
        realized into printed tables. No attempt is made to
        parametrize the interpolant in L,E,T parameter space, or Corey-space.

        Args:
            parameter: Between -1 and 1, inclusive. -1 reproduces low/
                pessimistic curve, 0 gives base, 1 gives high/optimistic.
            parameter2: If not None, used for the gas-oil interpolation,
                enables having interpolation uncorrelated for WaterOil and
                GasOil. Ignored for GasWater (no warning).
            h: Saturation step length in generated tables. Does not
                need to be the same as the tables interpolation is done from.
        """

        if parameter2 is not None:
            gasparameter = parameter2
        else:
            gasparameter = parameter

        # Either wateroil or gasoil can be None in the low, base, high
        # If they are None, it is a two-phase problem and we
        # should support this.
        do_gaswater = False
        do_wateroil = False
        do_gasoil = False
        if self.type == GasWater:
            do_gaswater = True
        elif self.type == WaterOilGas:
            do_wateroil = (
                self.base.wateroil is not None
                and self.low.wateroil is not None
                and self.high.wateroil is not None
            )

            do_gasoil = (
                self.base.gasoil is not None
                and self.low.gasoil is not None
                and self.high.gasoil is not None
            )

        if parameter2 is not None:
            if not do_gasoil:
                logger.warning("parameter2 is meaningless for water-oil only")
            if do_gaswater:
                logger.warning("parameter2 is meaningless for gas-water")

        # Initialize wateroil and gasoil curves to be filled with
        # interpolated curves:
        interpolant: Union[WaterOilGas, GasWater]
        tags: Set[str] = set()
        if do_wateroil or do_gaswater:
            assert self.low.wateroil is not None
            assert self.base.wateroil is not None
            assert self.high.wateroil is not None
            tags = tags.union(
                set(
                    [
                        self.base.wateroil.tag,
                        self.low.wateroil.tag,
                        self.high.wateroil.tag,
                    ]
                )
            )
        if do_gasoil:
            assert self.low.gasoil is not None
            assert self.base.gasoil is not None
            assert self.high.gasoil is not None
            tags = tags.union(
                set([self.base.gasoil.tag, self.low.gasoil.tag, self.high.gasoil.tag])
            )
        tagstring = "\n".join(tags)
        if do_gaswater:
            interpolant = GasWater(h=h, tag=tagstring)
        else:
            interpolant = WaterOilGas(h=h, tag=tagstring)

        if do_wateroil or do_gaswater:
            tag = f"SCAL recommendation interpolation to {parameter}\n" + tagstring
            assert self.low.wateroil is not None
            assert self.base.wateroil is not None
            assert self.high.wateroil is not None
            if abs(parameter) > 1.0:
                raise ValueError(
                    f"Interpolation parameter must be in [-1,1], got {parameter}"
                )
            if np.isclose(parameter, 0.0):
                interpolant.wateroil = copy.deepcopy(self.base.wateroil)
                interpolant.wateroil.tag = tag
            elif np.isclose(parameter, -1.0):
                interpolant.wateroil = copy.deepcopy(self.low.wateroil)
                interpolant.wateroil.tag = tag
            elif np.isclose(parameter, 1.0):
                interpolant.wateroil = copy.deepcopy(self.high.wateroil)
                interpolant.wateroil.tag = tag
            elif parameter < 0.0:
                interpolant.wateroil = interpolate_wo(
                    self.base.wateroil,
                    self.low.wateroil,
                    -parameter,
                    h=h,
                    tag=tag,
                )
            elif parameter > 0.0:
                interpolant.wateroil = interpolate_wo(
                    self.base.wateroil,
                    self.high.wateroil,
                    parameter,
                    h=h,
                    tag=tag,
                )
        else:
            interpolant.wateroil = None

        if do_gasoil or do_gaswater:
            assert self.low.gasoil is not None
            assert self.base.gasoil is not None
            assert self.high.gasoil is not None
            tag = f"SCAL recommendation interpolation to {gasparameter}\n" + tagstring
            if abs(gasparameter) > 1.0:
                raise ValueError(
                    "Interpolation parameter for gas must "
                    f"be in [-1,1], got {gasparameter}"
                )
            if np.isclose(gasparameter, 0.0):
                interpolant.gasoil = copy.deepcopy(self.base.gasoil)
                interpolant.gasoil.tag = tag
            elif np.isclose(gasparameter, -1.0):
                interpolant.gasoil = copy.deepcopy(self.low.gasoil)
                interpolant.gasoil.tag = tag
            elif np.isclose(gasparameter, 1.0):
                interpolant.gasoil = copy.deepcopy(self.high.gasoil)
                interpolant.gasoil.tag = tag
            elif gasparameter < 0.0:
                interpolant.gasoil = interpolate_go(
                    self.base.gasoil,
                    self.low.gasoil,
                    -1 * gasparameter,
                    h=h,
                    tag=tag,
                )
            elif gasparameter > 0.0:
                interpolant.gasoil = interpolate_go(
                    self.base.gasoil,
                    self.high.gasoil,
                    gasparameter,
                    h=h,
                    tag=tag,
                )
        else:
            interpolant.gasoil = None

        interpolant.fast = self.fast

        return interpolant
