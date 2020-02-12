"""SCALrecommendation, container for low, base and high WaterOilGas objects"""

import logging

import numpy as np

from pyscal import WaterOilGas, utils


logging.basicConfig()
logger = logging.getLogger(__name__)


class SCALrecommendation(object):
    """A SCAL recommendation consists of three OilWaterGas objects,
    tagged low, base and high.

    This container exists in order to to interpolation from -1 (low),
    through 0 (base) and to 1 (high).

    Args:
        low (WaterOilGas): An object representing the low case
        base (WaterOilGas): An object representing the base case
        high (WaterOilGas): An object representing the high case
        tag (str): A string that describes the recommendation. Optional.
    """

    def __init__(self, low, base, high, tag="", h=0.01):
        """Set up a SCAL recommendation curve set from WaterOilGas objects

        Arguments:
            low (WaterOilGas): low case
            base (WaterOilGas): base case
            high (WaterOilGas): high case
            tag (str): Describes the recommendtion. This string will be used
                as tag strings for the interpolants.
        """

        self.h = h
        self.tag = tag

        if isinstance(low, dict) and isinstance(base, dict) and isinstance(high, dict):

            logger.warning(
                (
                    "Making SCALrecommendation from dicts is deprecated "
                    "and will not be supported in future versions\n"
                    "Use WaterOilGas objects instead"
                )
            )

            self.defaultshandling("swirr", 0.0, [low, base, high])
            self.defaultshandling("swcr", 0.0, [low, base, high])
            self.defaultshandling("sorg", 0.0, [low, base, high])
            self.defaultshandling("sgcr", 0.0, [low, base, high])
            if "kroend" in low:
                logger.error("Use krogend or krowend instead of kroend. kroend ignored")
            self.defaultshandling("krogend", 1.0, [low, base, high])
            self.defaultshandling("krowend", 1.0, [low, base, high])
            self.defaultshandling("krwmax", 1.0, [low, base, high])
            self.defaultshandling("krgend", 1.0, [low, base, high])
            self.defaultshandling("krgmax", 1.0, [low, base, high])

            # Initialize saturation ranges for all curves
            self.low = WaterOilGas(
                swirr=low["swirr"],
                swl=low["swl"],
                sorw=low["sorw"],
                sorg=low["sorg"],
                sgcr=low["sgcr"],
                swcr=low["swcr"],
                h=h,
                tag=tag,
            )
            self.base = WaterOilGas(
                swirr=base["swirr"],
                swl=base["swl"],
                sorw=base["sorw"],
                sorg=base["sorg"],
                sgcr=base["sgcr"],
                swcr=base["swcr"],
                h=h,
                tag=tag,
            )
            self.high = WaterOilGas(
                swirr=high["swirr"],
                swl=high["swl"],
                sorw=high["sorw"],
                sorg=high["sorg"],
                sgcr=high["sgcr"],
                swcr=high["swcr"],
                h=h,
                tag=tag,
            )

            # Add water and oil curves
            self.low.wateroil.add_LET_water(
                l=low["Lw"],
                e=low["Ew"],
                t=low["Tw"],
                krwend=low["krwend"],
                krwmax=low["krwmax"],
            )
            self.base.wateroil.add_LET_water(
                l=base["Lw"],
                e=base["Ew"],
                t=base["Tw"],
                krwend=base["krwend"],
                krwmax=base["krwmax"],
            )
            self.high.wateroil.add_LET_water(
                l=high["Lw"],
                e=high["Ew"],
                t=high["Tw"],
                krwend=high["krwend"],
                krwmax=high["krwmax"],
            )

            self.low.wateroil.add_LET_oil(
                l=low["Lo"], e=low["Eo"], t=low["To"], kroend=low["krowend"]
            )
            self.base.wateroil.add_LET_oil(
                l=base["Lo"], e=base["Eo"], t=base["To"], kroend=base["krowend"]
            )
            self.high.wateroil.add_LET_oil(
                l=high["Lo"], e=high["Eo"], t=high["To"], kroend=high["krowend"]
            )

            # Add gas and oil curves:
            self.low.gasoil.add_LET_gas(
                l=low["Lg"],
                e=low["Eg"],
                t=low["Tg"],
                krgend=low["krgend"],
                krgmax=low["krgmax"],
            )
            self.base.gasoil.add_LET_gas(
                l=base["Lg"],
                e=base["Eg"],
                t=base["Tg"],
                krgend=base["krgend"],
                krgmax=base["krgmax"],
            )
            self.high.gasoil.add_LET_gas(
                l=high["Lg"],
                e=high["Eg"],
                t=high["Tg"],
                krgend=high["krgend"],
                krgmax=high["krgmax"],
            )
            self.low.gasoil.add_LET_oil(
                l=low["Log"], e=low["Eog"], t=low["Tog"], kroend=low["krogend"]
            )
            self.base.gasoil.add_LET_oil(
                l=base["Log"], e=base["Eog"], t=base["Tog"], kroend=base["krogend"]
            )
            self.high.gasoil.add_LET_oil(
                l=high["Log"], e=high["Eog"], t=high["Tog"], kroend=high["krogend"]
            )
        elif (
            isinstance(low, WaterOilGas)
            and isinstance(base, WaterOilGas)
            and isinstance(high, WaterOilGas)
        ):

            self.low = low
            self.base = base
            self.high = high
        else:
            raise ValueError("Wrong arguments to SCALrecommendation")

    # User should add capillary pressure explicitly by calling add**
    # on the class objects, or run the following method to add the
    # same to all curves:
    def add_simple_J(self, a=5, b=-1.5, poro_ref=0.25, perm_ref=100, drho=300, g=9.81):
        """Add (identical) simplified J-function to all water-oil
        curves in the SCAL recommendation set"""
        self.low.wateroil.add_simple_J(
            a=a, b=b, poro_ref=poro_ref, perm_ref=perm_ref, drho=drho, g=g
        )
        self.base.wateroil.add_simple_J(
            a=a, b=b, poro_ref=poro_ref, perm_ref=perm_ref, drho=drho, g=g
        )
        self.high.wateroil.add_simple_J(
            a=a, b=b, poro_ref=poro_ref, perm_ref=perm_ref, drho=drho, g=g
        )

    def interpolate(self, parameter, parameter2=None, h=0.02):
        """Interpolate between low, base and high
        parameter = -1 reproduces low curve
        parameter = 0 reproduces base curve
        parameter = 1 reproduces high curve

        Endpoints are located for input curves, and interpolated
        individually. Interpolation for the nonlinear part
        is done on a normalized interval between the endpoints

        Interpolation is linear in relperm-direction, and will thus not be
        linear in log-relperm-direction

        This method returns an WaterOilGas object which can be
        realized into printed tables. No attempt is made to
        parametrize the interpolant in L,E,T parameter space, or Corey-space.

        If a second parameter is supplied ("parameter2") this is used
        for the gas-oil interpolation. This enables the gas-oil
        interpolant to be fully uncorrelated to the water-oil
        interpolant. CHECK how this affects endpoints and Eclipse
        consistency!!

        """

        if parameter2 is not None:
            gasparameter = parameter2
        else:
            gasparameter = parameter

        # Either wateroil or gasoil can be None in the low, base, high
        # If they are None, it is a two-phase problem and we
        # should support this.
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

        if not do_wateroil and not do_gasoil:
            raise ValueError(
                "Neither wateroil or gasoil is complete in SCAL recommendation"
            )

        if parameter2 is not None and not do_gasoil:
            logger.warning("parameter2 is meaningless for water-oil only")

        # Initialize wateroil and gasoil curves to be filled with
        # interpolated curves:
        interpolant = WaterOilGas(h=h)

        if do_wateroil:
            if abs(parameter) > 1.0:
                logger.error("Interpolation parameter must be in [-1,1]")
                assert abs(parameter) <= 1.0
            elif np.isclose(parameter, 0.0):
                interpolant.wateroil = self.base.wateroil
            elif np.isclose(parameter, -1.0):
                interpolant.wateroil = self.low.wateroil
            elif np.isclose(parameter, 1.0):
                interpolant.wateroil = self.high.wateroil
            elif parameter < 0.0:
                interpolant.wateroil = utils.interpolate_wo(
                    self.base.wateroil, self.low.wateroil, -parameter, h=h
                )
            elif parameter > 0.0:
                interpolant.wateroil = utils.interpolate_wo(
                    self.base.wateroil, self.high.wateroil, parameter, h=h
                )
        else:
            interpolant.wateroil = None

        if do_gasoil:
            if abs(gasparameter) > 1.0:
                logger.error("Interpolation parameter must be in [-1,1]")
                assert abs(gasparameter) <= 1.0
            elif np.isclose(gasparameter, 0.0):
                interpolant.gasoil = self.base.gasoil
            elif np.isclose(gasparameter, -1.0):
                interpolant.gasoil = self.low.gasoil
            elif np.isclose(gasparameter, 1.0):
                interpolant.gasoil = self.high.gasoil
            elif gasparameter < 0.0:
                interpolant.gasoil = utils.interpolate_go(
                    self.base.gasoil, self.low.gasoil, -1 * gasparameter, h=h
                )
            elif gasparameter > 0.0:
                interpolant.gasoil = utils.interpolate_go(
                    self.base.gasoil, self.high.gasoil, gasparameter, h=h
                )
        else:
            interpolant.gasoil = None

        return interpolant

    @staticmethod
    def defaultshandling(key, value, dicts):
        """Helper function for __init__ to fill out missing values in
        dicts with relperm parameter

        This function IS DEPRECATED and will be removed
        when __init__ no longer supports dicts as arguments.
        """
        for dic in dicts:
            if key not in dic:
                dic[key] = value
