# -*- coding: utf-8 -*-
"""Wateroil module"""
from __future__ import division, absolute_import
from __future__ import print_function

import math
import logging
import six

import numpy as np
import pandas as pd

import pyscal
from pyscal.constants import EPSILON as epsilon
from pyscal.constants import SWINTEGERS, MAX_EXPONENT
from pyscal import utils


logging.basicConfig()
logger = logging.getLogger(__name__)


class WaterOil(object):
    """A representation of two-phase properties for oil-water.

    Can hold relative permeability data, and capillary pressure.

    Parametrizations for relative permeability:
     * Corey
     * LET

    For capillary pressure:
     * Simplified J-function

    For object initialization, only saturation endpoints must be inputted,
    and saturation resolution. An optional string can be added as a 'tag'
    that can be used when outputting.

    Relative permeability and/or capillary pressure can be added through
    parametrizations, or from a dataframe (will incur interpolation).

    Can be dumped as include files for Eclipse/OPM and Nexus simulators.

    Args:
        swirr (float): Absolute minimal water saturation at infinite capillary
            pressure.
        swl (float): First water saturation point in generated table. Used
            for normalized saturations.
        swcr (float): Critical water saturation. Water will not be mobile before the
            water saturation is above this value.
        sorw (float): Residual oil saturation after water flooding. At this oil
            saturation, the oil has zero relative permeability.
        h (float): Saturation step-length in the outputted table.
        tag (str): Optional string identifier, only used in comments.
        fast (bool): Set to True if in order to skip some integrity checks
            and nice-to-have features. Not needed to set for normal pyscal
            runs, as speed is seldom crucial. Default False
    """

    def __init__(
        self, swirr=0.0, swl=0.0, swcr=0.0, sorw=0.0, h=0.01, tag="", fast=False
    ):
        """Sets up the saturation range. Swirr is only relevant
        for the capillary pressure, not for relperm data."""
        assert -epsilon < swirr < 1.0 + epsilon
        assert -epsilon < swl < 1.0 + epsilon
        assert -epsilon < swcr < 1.0 + epsilon
        assert -epsilon < sorw < 1.0 + epsilon
        if h is None:
            h = 0.01
        assert epsilon < h <= 1
        assert swl < 1 - sorw
        assert swcr < 1 - sorw
        assert swirr < 1 - sorw
        if not isinstance(tag, six.string_types):
            tag = ""
        self.swirr = swirr
        self.swl = max(swl, swirr)  # Cannot allow swl < swirr. Warn?
        if not np.isclose(sorw, 0) and sorw < 1 / SWINTEGERS:
            # Give up handling sorw very close to zero
            sorw = 0.0
        if self.swl < swcr < self.swl + 1 / SWINTEGERS + epsilon:
            # Give up handling swcr so close to swl
            swcr = self.swl
        self.swcr = max(self.swl, swcr)  # Cannot allow swcr < swl. Warn?
        self.sorw = sorw
        self.h = h
        self.tag = tag
        self.fast = fast
        sw_list = (
            list(np.arange(self.swl, 1 - sorw, h)) + [self.swcr] + [1 - sorw] + [1]
        )
        sw_list.sort()  # Using default timsort on nearly sorted data.
        self.table = pd.DataFrame(sw_list, columns=["sw"])

        # Ensure that we do not have sw values that are too close
        # to each other, determined rougly by the distance 1/10000
        self.table["swint"] = list(
            map(int, list(map(round, self.table["sw"] * SWINTEGERS)))
        )
        self.table.drop_duplicates("swint", inplace=True)

        # Now, sw=1-sorw might be accidentaly dropped, so make sure we
        # have it by replacing the closest value by 1-sorw exactly
        sorwindex = (self.table["sw"] - (1 - self.sorw)).abs().sort_values().index[0]
        self.table.loc[sorwindex, "sw"] = 1 - self.sorw

        # Same for sw=swcr:
        swcrindex = (self.table["sw"] - (self.swcr)).abs().sort_values().index[0]
        self.table.loc[swcrindex, "sw"] = self.swcr

        # If sw=1 was dropped, then sorw was close to zero:
        if not np.isclose(self.table["sw"].max(), 1.0):
            # Add it as an extra row:
            self.table.loc[len(self.table) + 1, "sw"] = 1.0
            self.table.sort_values(by="sw", inplace=True)

        self.table.reset_index(inplace=True)
        self.table = self.table[["sw"]]  # Drop the swint column

        # Normalize for krw:
        self.table["swn"] = (self.table.sw - self.swcr) / (1 - self.swcr - self.sorw)
        # Normalize for krow:
        self.table["son"] = (1 - self.table.sw - self.sorw) / (
            1 - self.sorw - self.swcr
        )

        # Different normalization for Sw used for capillary pressure
        self.table["swnpc"] = (self.table.sw - swirr) / (1 - swirr)

        self.swcomment = "-- swirr=%g swl=%g swcr=%g sorw=%g\n" % (
            self.swirr,
            self.swl,
            self.swcr,
            self.sorw,
        )
        self.krwcomment = ""
        self.krowcomment = ""
        self.pccomment = ""

        logger.info(
            "Initialized WaterOil with %s saturation points", str(len(self.table))
        )

    def add_oilwater_fromtable(self, *args, **kwargs):
        """Deprecated, use ``add_fromtable()``"""
        logger.warning("add_oilwater_fromtable() is deprecated, use add_fromtable()")
        self.add_fromtable(*args, **kwargs)

    def add_fromtable(
        self,
        dframe,
        swcolname="Sw",
        krwcolname="krw",
        krowcolname="krow",
        pccolname="pcow",
        krwcomment="",
        krowcomment="",
        pccomment="",
        sorw=None,
    ):
        """Interpolate relpermdata from a dataframe.

        The saturation range with endpoints must be set up beforehand,
        and must be compatible with the tabular input. The tabular
        input will be interpolated to the initialized Sw-table

        If you have krw and krow in different dataframes, call this
        function twice

        Calling function is responsible for checking if any data was
        actually added to the table.

        The relpermdata will be interpolated using a monotone cubic
        interpolator below 1-sorw, and linearly above 1-sorw. Capillary
        pressure data will be interpolated monotone cubicly over the
        entire saturation interval

        The python package ecl2df has a tool for converting Eclipse input
        files to dataframes.

        Args:
            dframe (pd.DataFrame): containing data
            swcolname (string): column name with the saturation data in the dataframe df
            krwcolname (string): name of column in df with krw
            krowcolname (string): name of column in df with krow
            pccolname (string): name of column in df with capillary pressure data
            krwcomment (string): Inserted into comment
            krowcomment (string): Inserted into comment
            pccomment (str): Inserted into comment
            sorw (float): Explicit sorw. If None, it will be estimated from
                the numbers in krw (or krow)
        """
        from scipy.interpolate import PchipInterpolator, interp1d

        # Avoid having to deal with multi-indices:
        if len(dframe.index.names) > 1:
            logger.warning(
                "add_fromtable() did a reset_index(), consider not supplying MultiIndex"
            )
            dframe = dframe.reset_index()

        if swcolname not in dframe:
            logger.critical(
                "%s not found in dataframe, can't read table data", swcolname
            )
            raise ValueError
        if (dframe[swcolname].diff() < 0).any():
            raise ValueError("sw data not sorted")
        if krwcolname in dframe:
            if not sorw:
                sorw = dframe[swcolname].max() - utils.estimate_diffjumppoint(
                    dframe, xcol=swcolname, ycol=krwcolname, side="right"
                )
                logger.info("Estimated sorw in tabular data to %f", sorw)
            assert -epsilon <= sorw <= 1 + epsilon
            linearpart = dframe[swcolname] >= 1 - sorw
            nonlinearpart = dframe[swcolname] <= 1 - sorw  # (overlapping at sorw)
            if sum(linearpart) < 2:
                linearpart = pd.Series([False] * len(linearpart))
                nonlinearpart = ~linearpart
                sorw = 0
            if sum(nonlinearpart) < 2:
                nonlinearpart = pd.Series([False] * len(nonlinearpart))
                linearpart = ~nonlinearpart
            if not np.isclose(dframe[swcolname].min(), self.table["sw"].min()):
                raise ValueError("Incompatible swl")
            # Verify that incoming data is increasing (or level):
            if not (dframe[krwcolname].diff().dropna() > -epsilon).all():
                raise ValueError("Incoming krw not increasing")
            if sum(nonlinearpart) >= 2:
                pchip = PchipInterpolator(
                    dframe[nonlinearpart][swcolname].astype(float),
                    dframe[nonlinearpart][krwcolname].astype(float),
                )
                self.table.loc[self.table["sw"] <= 1 - sorw, "krw"] = pchip(
                    self.table.loc[self.table["sw"] <= 1 - sorw, "sw"]
                )
            if sum(linearpart) >= 2:
                linearinterpolator = interp1d(
                    dframe[linearpart][swcolname].astype(float),
                    dframe[linearpart][krwcolname].astype(float),
                )
                self.table.loc[self.table["sw"] > 1 - sorw, "krw"] = linearinterpolator(
                    self.table.loc[self.table["sw"] > 1 - sorw, "sw"]
                )
            self.sorw = sorw
            self.krwcomment = "-- krw from tabular input" + krwcomment + "\n"

        if krowcolname in dframe:
            if not sorw:
                sorw = dframe[swcolname].max() - utils.estimate_diffjumppoint(
                    dframe, xcol=swcolname, ycol=krowcolname, side="right"
                )
                logger.info("Estimated sorw in tabular data from krow to %s", sorw)
            assert -epsilon <= sorw <= 1 + epsilon
            linearpart = dframe[swcolname] >= 1 - sorw
            nonlinearpart = dframe[swcolname] <= 1 - sorw  # (overlapping at sorw)
            if sum(linearpart) < 2:
                linearpart = pd.Series([False] * len(linearpart))
                nonlinearpart = ~linearpart
                sorw = 0
            if sum(nonlinearpart) < 2:
                nonlinearpart = pd.Series([False] * len(nonlinearpart))
                linearpart = ~nonlinearpart
            if not np.isclose(dframe[swcolname].min(), self.table["sw"].min()):
                raise ValueError("Incompatible swl")
            if not (dframe[krowcolname].diff().dropna() < epsilon).all():
                raise ValueError("Incoming krow not decreasing")
            if sum(nonlinearpart) >= 2:
                pchip = PchipInterpolator(
                    dframe[nonlinearpart][swcolname].astype(float),
                    dframe[nonlinearpart][krowcolname].astype(float),
                )
                self.table.loc[self.table["sw"] <= 1 - sorw, "krow"] = pchip(
                    self.table.loc[self.table["sw"] <= 1 - sorw, "sw"]
                )
            if sum(linearpart) >= 2:
                linearinterpolator = interp1d(
                    dframe[linearpart][swcolname].astype(float),
                    dframe[linearpart][krowcolname].astype(float),
                )
                self.table.loc[
                    self.table["sw"] > 1 - sorw, "krow"
                ] = linearinterpolator(
                    self.table.loc[self.table["sw"] > 1 - sorw, "sw"]
                )
            self.sorw = sorw
            self.krowcomment = "-- krow from tabular input" + krowcomment + "\n"
        if pccolname in dframe:
            # Incoming dataframe must cover the range:
            if dframe[swcolname].min() > self.table["sw"].min():
                raise ValueError("Too large swl for pc interpolation")
            if dframe[swcolname].max() < self.table["sw"].max():
                raise ValueError("max(sw) of incoming data not large enough")
            if np.isinf(dframe[pccolname]).any():
                logger.warning(
                    (
                        "Infinity pc values detected. Will be dropped. "
                        "Risk of extrapolation"
                    )
                )
            dframe = dframe.replace([np.inf, -np.inf], np.nan)
            dframe.dropna(subset=[pccolname], how="all", inplace=True)
            # If nonzero, then it must be decreasing:
            if dframe[pccolname].abs().sum() > 0:
                if not (dframe[pccolname].diff().dropna() < 0.0).all():
                    raise ValueError("Incoming pc not decreasing")
            pchip = PchipInterpolator(
                dframe[swcolname].astype(float), dframe[pccolname].astype(float)
            )
            self.table["pc"] = pchip(self.table["sw"])
            if np.isnan(self.table["pc"]).any() or np.isinf(self.table["pc"]).any():
                raise ValueError("inf/nan in interpolated data, check input")
            self.pccomment = "-- pc from tabular input" + pccomment + "\n"

    def add_corey_water(self, nw=2, krwend=1, krwmax=None):
        """ Add krw data through the Corey parametrization

        A column named 'krw' will be added. If it exists, it will
        be replaced.

        It is assumed that there are no sw points between
        sw=1-sorw and sw=1, which should give linear
        interpolations in simulators. The corey parameter
        applies up to 1-sorw.

        krwmax will be ignored if sorw is close to zero

        Args:
            nw (float): Corey parameter for water.
            krwend (float): value of krw at 1 - sorw.
            krwmax (float): maximal value at Sw=1. Default 1

        """
        assert epsilon < nw < MAX_EXPONENT
        if krwmax:
            assert 0 < krwend <= krwmax <= 1.0
        else:
            assert 0 < krwend <= 1.0

        self.table["krw"] = krwend * self.table.swn ** nw

        self.set_endpoints_linearpart_krw(krwend, krwmax)

        if not krwmax:
            krwmax = 1
        self.krwcomment = "-- Corey krw, nw=%g, krwend=%g, krwmax=%g\n" % (
            nw,
            krwend,
            krwmax,
        )

    def set_endpoints_linearpart_krw(self, krwend, krwmax=None):
        """Set linear parts of krw outside endpoints.

        Curve will be linear from [1 - sorw, 1] (from krwmax to krwend)
        and zero in [swl, swcr]

        This function is used by add_corey_water(), and perhaps by other
        utility functions. It should not be necessary for end-users.

        Args:
            krwend (float): krw at 1 - sorwr
            krwmax (float): krw at Sw=1. Default 1.
        """
        # Avoid machine accuracy problems around sorw (monotonicity):
        self.table.loc[self.table["krw"] > krwmax, "krw"] = krwmax

        # We assume there is no points between 1 - sorw and 1, so no
        # need to excplicitly insert linearly interpolated values.
        if self.sorw > 1.0 / SWINTEGERS:
            if not krwmax:
                krwmax = 1.0
            self.table.loc[self.table["sw"] > (1 - self.sorw - epsilon), "krw"] = krwmax
        else:
            if krwmax and krwmax < 1.0:
                # Only warn if non-default value
                logger.warning("krwmax ignored when sorw is zero")
            self.table.loc[self.table["sw"] > (1 - self.sorw - epsilon), "krw"] = krwend
        self.table.loc[np.isclose(self.table["sw"], 1 - self.sorw), "krw"] = krwend
        self.table.loc[self.table.sw < self.swcr, "krw"] = 0

    def set_endpoints_linearpart_krow(self, kroend, kromax=None):
        """Set linear parts of krow outside endpoints

        Curve will be linear in [swl, swcr] (from kromax to kroend)
        and zero in [1 - sorw, 1]

        This function is used by add_corey_water(), and perhaps by other
        utility functions. It should not be necessary for end-users.

        Args:
            kroend (float): value of kro at swcr
            kromax (float): maximal value of kro at sw=swl. Default 1
        """

        # Linear curve between swl and swcr (left part):
        self.table.loc[self.table["son"] > 1.0 + epsilon, "krow"] = np.nan
        if self.swcr < self.swl + epsilon:
            if kromax and kromax < 1.0:
                # Only warn if non-default value
                logger.warning("kromax ignored when swcr is close to swl")
            self.table.loc[self.table["sw"] <= self.swl + epsilon, "krow"] = kroend
        else:
            if not kromax:
                kromax = 1
            self.table.loc[self.table["sw"] <= self.swl + epsilon, "krow"] = kromax
            interp_krow = (
                self.table[["sw", "krow"]]
                .set_index("sw")
                .interpolate(method="index")["krow"]
            )
            self.table.loc[:, "krow"] = interp_krow.values

        # Avoid machine accuracy problems around swcr (monotonicity):
        self.table.loc[self.table["krow"] > kromax, "krow"] = kromax

        # Set to zero above sorw:
        self.table.loc[self.table["sw"] > 1 - self.sorw, "krow"] = 0

    def add_LET_water(self, l=2, e=2, t=2, krwend=1, krwmax=None):
        """Add krw data through LET parametrization

        It is assumed that there are no sw points between
        sw=1-sorw and sw=1, which should give linear
        interpolations in simulators. The LET parameters
        apply up to 1-sorw.

        krwmax will be ignored if sorw is close to zero.

        Args:
            l (float): LET parameter
            e (float): LET parameter
            t (float): LET parameter
            krwend (float): value of krw at 1 - sorw
            krwmax (float): maximal value at Sw=1. Default 1
        """
        assert epsilon < l < MAX_EXPONENT
        assert epsilon < e < MAX_EXPONENT
        assert epsilon < t < MAX_EXPONENT
        if krwmax:
            assert 0 < krwend <= krwmax <= 1.0
        else:
            assert 0 < krwend <= 1.0

        self.table["krw"] = (
            krwend
            * self.table.swn ** l
            / ((self.table.swn ** l) + e * (1 - self.table.swn) ** t)
        )
        # This equation is undefined for t a float and swn=1, set explicitly:
        self.table.loc[np.isclose(self.table["swn"], 1.0), "krw"] = krwend

        self.set_endpoints_linearpart_krw(krwend, krwmax)

        if not krwmax:
            krwmax = 1
        self.krwcomment = "-- LET krw, l=%g, e=%g, t=%g, krwend=%g, krwmax=%g\n" % (
            l,
            e,
            t,
            krwend,
            krwmax,
        )

    def add_LET_oil(self, l=2, e=2, t=2, kroend=1, kromax=None):
        """
        Add kro data through LET parametrization

        kromax will be ignored if swcr is close to swl.

        Args:
            l (float): LET parameter
            e (float): LET parameter
            t (float): LET parameter
            kroend (float): value of kro at swcr
            kromax (float): maximal value of kro at sw=swl. Default 1

        Returns:
            None (modifies object)
        """
        assert epsilon < l < MAX_EXPONENT
        assert epsilon < e < MAX_EXPONENT
        assert epsilon < t < MAX_EXPONENT
        if kromax:
            assert 0 < kroend <= kromax <= 1.0
        else:
            assert 0 < kroend <= 1.0

        self.table["krow"] = (
            kroend
            * self.table.son ** l
            / ((self.table.son ** l) + e * (1 - self.table.son) ** t)
        )
        # This equation is undefined for t a float and son=1, set explicitly:
        self.table.loc[np.isclose(self.table["son"], 1.0), "krow"] = kroend

        self.table.loc[self.table.sw >= (1 - self.sorw), "krow"] = 0

        self.set_endpoints_linearpart_krow(kroend, kromax)

        if not kromax:
            kromax = 1
        self.krowcomment = "-- LET krow, l=%g, e=%g, t=%g, kroend=%g, kromax=%g\n" % (
            l,
            e,
            t,
            kroend,
            kromax,
        )

    def add_corey_oil(self, now=2, kroend=1, kromax=None):
        """Add kro data through the Corey parametrization

        Corey applies to the interval between swcr and 1 - sorw

        Curve is linear between swl and swcr, zero above 1 - sorw.

        kromax will be ignored if swcr is close to swl.

        Args:
            now (float): Corey exponent
            kroend (float): kro value at swcr
            kromax (float): kro value at swl
        Returns:
            None (modifies object)
        """
        assert epsilon < now < MAX_EXPONENT
        if kromax:
            assert 0 < kroend <= kromax <= 1.0
        else:
            assert 0 < kroend <= 1.0

        self.table["krow"] = kroend * self.table.son ** now
        self.table.loc[self.table.sw >= (1 - self.sorw), "krow"] = 0

        self.set_endpoints_linearpart_krow(kroend, kromax)

        if not kromax:
            kromax = 1
        self.krowcomment = "-- Corey krow, now=%g, kroend=%g, kromax=%g\n" % (
            now,
            kroend,
            kromax,
        )

    def add_simple_J(self, a=5, b=-1.5, poro_ref=0.25, perm_ref=100, drho=300, g=9.81):
        """Add capillary pressure function from a simplified J-function

        This is the 'inverse' or 'RMS' version of the a and b, the formula
        is

            J = a S_w^b

        J is not dimensionless.
        Doc: https://wiki.equinor.com/wiki/index.php/Res:Water_saturation_from_Leverett_J-function

        poro_ref is a fraction, between 0 and 1
        perm_ref is in milliDarcy
        drho has SI units kg/m³. Default value is 300
        g has SI units m/s², default value is 9.81
        """  # noqa
        assert g >= 0
        assert b < MAX_EXPONENT
        assert b > -MAX_EXPONENT
        assert 0.0 <= poro_ref <= 1.0
        assert perm_ref > 0.0

        if self.swl < epsilon:
            logger.error(
                "swl must larger than zero to avoid infinite capillary pressure"
            )
            raise ValueError

        if b > 0:
            logger.warning(
                "positive b will give increasing capillary pressure with saturation"
            )

        # drho = rwo_w - rho_o, in units g/cc

        # swnpc is a normalized saturation, but normalized with
        # respect to swirr, not to swl (the swirr here is sometimes
        # called 'swirra' - asymptotic swirr)
        self.table["J"] = a * self.table.swnpc ** b
        self.table["H"] = self.table.J * math.sqrt(poro_ref / perm_ref)
        # Scale drho and g from SI units to g/cc and m/s²100
        self.table["pc"] = self.table.H * drho / 1000 * g / 100.0
        self.pccomment = (
            "-- Simplified J function for Pc; \n--   "
            + "a=%g, b=%g, poro_ref=%g, perm_ref=%g mD, drho=%g kg/m^3, g=%g m/s^2\n"
            % (a, b, poro_ref, perm_ref, drho, g)
        )

    def add_normalized_J(self, a, b, poro, perm, sigma_costau):
        # Don't make this a raw string to avoid the \l warning,
        # it destroys the Latex-formatting in sphinx
        """
        Add capillary pressure in bar through a normalized J-function.

        .. math::

            p_c = \\frac{\left(\\frac{S_w}{a}\\right)^{\\frac{1}{b}}
            \sigma \cos \\tau}{\sqrt{\\frac{k}{\phi}}}

        The Sw saturation used in the formula is normalized with respect
        to the swirr parameter.

        Args:
            a (float): a parameter
            b (float): b exponent (typically negative)
            poro (float): Porosity value, fraction between 0 and 1
            perm (float): Permeability value in mD
            sigma_costau (float): Interfacial tension in mN/m (typical value 30 mN/m)

        Returns:
            None. Modifies pc column in self.table, using bar as pressure unit.
        """  # noqa
        assert epsilon < abs(a) < MAX_EXPONENT
        assert epsilon < abs(b) < MAX_EXPONENT
        assert epsilon < poro <= 1.0
        assert epsilon < perm
        assert isinstance(sigma_costau, (int, float))

        if b < 0 and np.isclose(self.swirr, self.swl):
            logger.error("swl must be set larger than swirr to avoid infinite p_c")
            raise ValueError("swl must be larger than swirr")

        if abs(b) < 0.01:
            logger.warning(
                "b exponent is very small, risk of infinite capillary pressure"
            )

        if abs(a) < 0.01:
            logger.warning(
                "a parameter is very small, risk of infinite capillary pressure"
            )

        if abs(a) > 5:
            logger.warning(
                "a parameter is very high, risk of infinite capillary pressure"
            )

        # 1 atm is equivalent to 101325 pascal = 1.01325 bar
        # pascal_to_atm = 1.0 / 101325.0  # = 9.86923267e-6
        pascal_to_bar = 1e-5

        perm_darcy = perm / 1000
        perm_sq_meters = perm_darcy * 9.869233e-13
        tmp = (self.table["swnpc"] / a) ** (1.0 / b)
        tmp = tmp / math.sqrt(perm_sq_meters / poro)
        tmp = tmp * sigma_costau / 1000  # Converting mN/m to N/m
        self.table["pc"] = tmp * pascal_to_bar
        self.pccomment = (
            "-- Capillary pressure from normalized J-function, in bar\n"
            + "-- a=%g, b=%g, poro=%g, perm=%g mD, sigma_costau=%g mN/m\n"
            % (a, b, poro, perm, sigma_costau)
        )

    def add_skjaeveland_pc(self, cw, co, aw, ao, swr=None, sor=None):
        """Add capillary pressure from the Skjæveland correlation,

        Doc: https://wiki.equinor.com/wiki/index.php/Res:The_Skjaeveland_correlation_for_capillary_pressure

        The implementation is unit independent, units are contained in the
        input constants.

        If swr and sor are not provided, it will be taken from the
        swirr and sorw. Only use different values here if you know
        what you are doing.

        Modifies or adds self.table.pc if succesful.
        Returns false if error occured.

        """  # noqa
        inputerror = False
        if cw < 0:
            logger.error("cw must be larger or equal to zero")
            inputerror = True
        if co > 0:
            logger.error("co must be less than zero")
            inputerror = True
        if aw <= 0:
            logger.error("aw must be larger than zero")
            inputerror = True
        if ao <= 0:
            logger.error("ao must be larger than zero")
            inputerror = True

        if swr is None:
            swr = self.swirr
        if sor is None:
            sor = self.sorw

        if swr >= 1 - sor:
            logger.error("swr (swirr) must be less than 1 - sor")
            inputerror = True
        if swr < 0 or swr > 1:
            logger.error("swr must be contained in [0,1]")
            inputerror = True
        if sor < 0 or sor > 1:
            logger.error("sor must be contained in [0,1]")
            inputerror = True
        if inputerror:
            return
        self.pccomment = (
            "-- Skjæveland correlation for Pc;\n"
            + "-- cw=%g, co=%g, aw=%g, ao=%g, swr=%g, sor=%g\n"
            % (cw, co, aw, ao, swr, sor)
        )

        # swnpc is a normalized saturation, but normalized with
        # respect to swirr, not to swl (the swirr here is sometimes
        # called 'swirra' - asymptotic swirr)

        # swnpc is generated upon object initialization, but overwritten
        # here to most likely the same values.
        self.table["swnpc"] = (self.table.sw - swr) / (1 - swr)

        # sonpc is almost like 'son', but swl is not used here:
        self.table["sonpc"] = (1 - self.table.sw - sor) / (1 - sor)

        # The Skjæveland correlation
        self.table.loc[self.table.sw < 1 - sor, "pc"] = cw / (
            self.table.swnpc ** aw
        ) + co / (self.table.sonpc ** ao)

        # From 1-sor, the pc is not defined. We want to extrapolate constantly,
        # but with a twist as Eclipse does not non-monotone capillary pressure:
        self.table["pc"].fillna(value=self.table.pc.min(), inplace=True)
        nanrows = self.table.sw > 1 - sor - epsilon
        self.table.loc[nanrows, "pc"] = (
            self.table.loc[nanrows, "pc"] - self.table.loc[nanrows, "sw"]
        )  # Just deduct sw to make it monotone..

    def add_LET_pc_pd(self, Lp, Ep, Tp, Lt, Et, Tt, Pcmax, Pct):
        """Add a primary drainage LET capillary pressure curve.

        Docs: https://wiki.equinor.com/wiki/index.php/Res:The_LET_correlation_for_capillary_pressure

        Note that Pc where Sw > 1 - sorw will appear linear because
        there are no saturation points in that interval.
        """  # noqa
        assert epsilon < Lp < MAX_EXPONENT
        assert epsilon < Ep < MAX_EXPONENT
        assert epsilon < Tp < MAX_EXPONENT
        assert epsilon < Lt < MAX_EXPONENT
        assert epsilon < Et < MAX_EXPONENT
        assert epsilon < Tt < MAX_EXPONENT
        assert Pct <= Pcmax

        # The "forced part"
        self.table["Ffpcow"] = (1 - self.table["swnpc"]) ** Lp / (
            (1 - self.table["swnpc"]) ** Lp + Ep * self.table["swnpc"] ** Tp
        )

        # The gradual rise part:
        self.table["Ftpcow"] = self.table["swnpc"] ** Lt / (
            self.table["swnpc"] ** Lt + Et * (1 - self.table["swnpc"]) ** Tt
        )

        # Putting it together:
        self.table["pc"] = (
            (Pcmax - Pct) * self.table["Ffpcow"] - Pct * self.table["Ftpcow"] + Pct
        )

        # Special handling of the interval [0,swirr]
        self.table.loc[self.table["swn"] < epsilon, "pc"] = Pcmax
        self.pccomment = (
            "-- LET correlation for primary drainage Pc;\n"
            "-- Lp=%g, Ep=%g, Tp=%g, Lt=%g, Et=%g, Tt=%g, Pcmax=%g, Pct=%g\n"
            % (Lp, Ep, Tp, Lt, Et, Tt, Pcmax, Pct)
        )

    def add_LET_pc_imb(self, Ls, Es, Ts, Lf, Ef, Tf, Pcmax, Pcmin, Pct):
        """Add an imbition LET capillary pressure curve.

        Docs: https://wiki.equinor.com/wiki/index.php/Res:The_LET_correlation_for_capillary_pressure
        """  # noqa
        assert epsilon < Ls < MAX_EXPONENT
        assert epsilon < Es < MAX_EXPONENT
        assert epsilon < Ts < MAX_EXPONENT
        assert epsilon < Lf < MAX_EXPONENT
        assert epsilon < Ef < MAX_EXPONENT
        assert epsilon < Tf < MAX_EXPONENT
        assert Pcmin <= Pct <= Pcmax

        # Normalized water saturation including sorw
        self.table["swnpco"] = (self.table.sw - self.swirr) / (
            1 - self.sorw - self.swirr
        )

        # The "forced part"
        self.table["Fficow"] = self.table["swnpco"] ** Lf / (
            self.table["swnpco"] ** Lf + Ef * (1 - self.table["swnpco"]) ** Tf
        )

        # The spontaneous part:
        self.table["Fsicow"] = (1 - self.table["swnpco"]) ** Ls / (
            (1 - self.table["swnpco"]) ** Ls + Es * self.table["swnpco"] ** Ts
        )

        # Putting it together:
        self.table["pc"] = (
            (Pcmax - Pct) * self.table["Fsicow"]
            + (Pcmin - Pct) * self.table["Fficow"]
            + Pct
        )

        # Special handling of the interval [0,swirr]
        self.table.loc[self.table["swnpco"] < epsilon, "pc"] = Pcmax
        # and [1-sorw,1]
        self.table.loc[self.table["swnpco"] > 1 - epsilon, "pc"] = Pcmin
        self.pccomment = (
            "-- LET correlation for imbibition Pc;\n"
            "-- Ls=%g, Es=%g, Ts=%g, Lf=%g, Ef=%g, Tf=%g, Pcmax=%g, Pcmin=%g, Pct=%g\n"
            % (Ls, Es, Ts, Lf, Ef, Tf, Pcmax, Pcmin, Pct)
        )

    def estimate_sorw(self, curve="krw"):
        """Estimate sorw of the current krw data.

        This is mostly relevant when add_fromtable() has been used.
        sorw is estimated by searching for a linear part in krw downwards from sw=1.
        In practice it is impossible to infer sorw = 0, since we are limited by
        h, and the last segment from sw=1-h to sw=1 can always be assumed linear.
        Expect sorw = h if the real sorw = 0, but do not depend that it might
        not return zero in the future (one could argue that sorw = h should be
        specially treated to mean sorw = 0)

        If the curve is linear everywhere, sorw will be returned as swl + h

        krow is not used, and should probably not be, as it can be very close to
        zero before approaching sorw.

        Args:
            curve (str): Colum name of column to use, default is krw.
                If this is all linear, but krow is not, you might be better off
                with krow
        Returns:
            float: The estimated sorw.
        """
        assert curve in self.table
        assert self.table[curve].sum() > 0
        return self.table["sw"].max() - utils.estimate_diffjumppoint(
            self.table, xcol="sw", ycol=curve, side="right"
        )

    def estimate_swcr(self, curve="krow"):
        """Estimate swcr of the current krw data.

        swcr is estimated by searching for a linear part in krw upwards from sw=swl.
        In practice it is impossible to infer swcr = 0, since we are limited by
        h, and the first segment is assumed linear anyways.

        If the curve is linear everywhere, swcr can end up at the right end
        of your saturation interval.

        Args:
            curve (str): Colum name of column to use, default is krow.
                If this is all linear, but krw is not, you might be better off
                with krw
        Returns:
            float: The estimated sgcr.
        """
        assert curve in self.table
        assert self.table[curve].sum() > 0
        return utils.estimate_diffjumppoint(
            self.table, xcol="sw", ycol=curve, side="left"
        )

    def crosspoint(self):
        """Locate and return the saturation point where krw = krow

        Accuracy of this crosspoint depends on the resolution chosen
        when initializing the saturation range
       """

        # Make a copy for calculations
        tmp = pd.DataFrame(self.table[["sw", "krw", "krow"]])
        tmp.loc[:, "krwminuskrow"] = tmp["krw"] - tmp["krow"]

        # Add a zero value for the difference column, and interpolate
        # the sg column to the zero value
        zerodf = pd.DataFrame(index=[len(tmp)], data={"krwminuskrow": 0.0})
        tmp = pd.concat([tmp, zerodf], sort=True)
        # When Pandas is upgraded for all users:
        # tmp = pd.concat([tmp, zerodf], sort=True)
        tmp.set_index("krwminuskrow", inplace=True)

        if tmp.index.isnull().any():
            logger.warning("Could not compute crosspoint. Bug?")
            return -1

        tmp.interpolate(method="slinear", inplace=True)

        return tmp[np.isclose(tmp.index, 0.0)].sw.values[0]

    def selfcheck(self, mode="SWOF"):
        """Check validities of the data in the table.

        An unfinished object will return False.

        If you call SWOF, this function must not return False

        This function should not throw an exception, but capture
        the error and give an error.

        Args:
            mode (str): "SWOF" or "SWFN". If SWFN, krow is not required.
        """
        error = False
        if "krw" not in self.table:
            logger.error("krw data not found")
            error = True
        if not (self.table["sw"].diff().dropna().round(10) > -epsilon).all():
            logger.error("sw data not strictly increasing")
            error = True
        if (
            "krw" in self.table
            and not (self.table["krw"].diff().dropna().round(10) >= -epsilon).all()
        ):
            logger.error("krw data not monotonically increasing")
            error = True
        if mode != "SWFN":
            if "krow" not in self.table:
                logger.error("krow data not found")
                error = True

            if (
                "krow" in self.table.columns
                and not (self.table["krow"].diff().dropna().round(10) <= epsilon).all()
            ):
                # In normal Eclipse runs, krow needs to be level or decreasing.
                # In hysteresis runs, it needs to be strictly decreasing, that must
                # be the users responsibility.
                logger.error("krow data not level or monotonically decreasing")
                error = True
        if "pc" in self.table.columns and self.table["pc"][0] > -epsilon:
            if not (self.table["pc"].diff().dropna().round(10) < epsilon).all():
                logger.error("pc data not strictly decreasing")
                error = True
        if "pc" in self.table.columns and np.isinf(self.table["pc"].max()):
            logger.error("pc goes to infinity. Maybe swirr=swl?")
            error = True
        for col in list(set(["sw", "krw", "krow"]) & set(self.table.columns)):
            if not (
                (round(min(self.table[col]), 10) >= -epsilon)
                and (round(max(self.table[col]), 10) <= 1 + epsilon)
            ):
                logger.error("%s data should be contained in [0,1]", col)
                error = True
        if error:
            return False
        logger.info("WaterOil object is checked to be valid")
        return True

    def SWOF(self, header=True, dataincommentrow=True):
        """
        Produce SWOF input for Eclipse reservoir simulator.

        The columns sw, krw, krow and pc are outputted and
        formatted accordingly.

        Meta-information for the tabulated data are printed
        as Eclipse comments.

        Args:
            header (bool): Indicate whether the SWOF string should
                be emitted. If you have multiple SATNUMs, you should
                set this to True only for the first (or False for all,
                and emit the SWOF yourself). Default True
            dataincommentrow (bool): Wheter metadata should
                be printed. Defualt True
        """
        if not self.fast and not self.selfcheck():
            # selfcheck failed and has issued an error message
            return ""
        string = ""
        if header:
            string += "SWOF\n"
        string += "-- " + self.tag + "\n"
        string += "-- pyscal: " + str(pyscal.__version__) + "\n"
        if "pc" not in self.table.columns:
            self.table["pc"] = 0
            self.pccomment = "-- Zero capillary pressure\n"
        if dataincommentrow:
            string += self.swcomment
            string += self.krwcomment
            string += self.krowcomment
            if not self.fast:
                string += "-- krw = krow @ sw=%1.5f\n" % self.crosspoint()
            string += self.pccomment
        width = 10
        string += (
            "-- "
            + "SW".ljust(width - 3)
            + "KRW".ljust(width)
            + "KROW".ljust(width)
            + "PC".ljust(width)
            + "\n"
        )
        string += self.table[["sw", "krw", "krow", "pc"]].to_csv(
            sep=" ", float_format="%1.7f", header=None, index=False
        )
        string += "/\n"  # Empty line at the end
        return string

    def SWFN(self, header=True, dataincommentrow=True):
        """Return a SWFN keyword with data to Eclipse"""
        if not self.selfcheck(mode="SWFN"):
            # selfcheck will print errors/warnings
            return ""
        string = ""
        if "pc" not in self.table.columns:
            self.table["pc"] = 0
            self.pccomment = "-- Zero capillary pressure\n"
        if header:
            string += "SWFN\n"
        string += "-- " + self.tag + "\n"
        string += "-- pyscal: " + str(pyscal.__version__) + "\n"
        if dataincommentrow:
            string += self.swcomment
            string += self.krwcomment
            if "krow" in self.table.columns and not self.fast:
                string += "-- krw = krow @ sw=%1.5f\n" % self.crosspoint()
            string += self.pccomment
        width = 10
        string += (
            "-- "
            + "SW".ljust(width - 3)
            + "KRW".ljust(width)
            + "PC".ljust(width)
            + "\n"
        )
        string += self.table[["sw", "krw", "pc"]].to_csv(
            sep=" ", float_format="%1.7f", header=None, index=False
        )
        string += "/\n"  # Empty line at the end
        return string

    def WOTABLE(self, header=True, dataincommentrow=True):
        """Return a string for a Nexus WOTABLE"""
        string = ""
        if "pc" not in self.table.columns:
            self.table["pc"] = 0
            self.pccomment = "-- Zero capillary pressure\n"

        if header:
            string += "WOTABLE\n"
            string += "SW KRW KROW PC\n"
        string += "! pyscal: " + str(pyscal.__version__) + "\n"
        if dataincommentrow:
            string += self.swcomment.replace("--", "!")
            string += self.krwcomment.replace("--", "!")
            string += self.krowcomment.replace("--", "!")
            if not self.fast:
                string += "! krw = krow @ sw=%1.5f\n" % self.crosspoint()
            string += self.pccomment.replace("--", "!")
        width = 10
        string += (
            "! "
            + "SW".ljust(width - 2)
            + "KRW".ljust(width)
            + "KROW".ljust(width)
            + "PC".ljust(width)
            + "\n"
        )
        string += self.table[["sw", "krw", "krow", "pc"]].to_csv(
            sep=" ", float_format="%1.7f", header=None, index=False
        )
        return string

    def plotpc(
        self,
        mpl_ax=None,
        color="blue",
        alpha=1,
        linewidth=1,
        linestyle="-",
        label="",
        logyscale=False,
    ):
        """Plot capillary pressure (pc)

        If mpl_ax is supplied, the curve will be drawn on
        that, if not, a new axis (plot) will be made
        """
        import matplotlib.pyplot as plt
        import matplotlib

        if mpl_ax is None:
            matplotlib.style.use("ggplot")
            _, useax = plt.subplots()
        else:
            useax = mpl_ax
        if logyscale:
            useax.set_yscale("log")
            useax.set_ylim([1e-6, 100])
        self.table.plot(
            ax=useax,
            x="sw",
            y="pc",
            c=color,
            alpha=alpha,
            label=label,
            legend=None,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        if mpl_ax is None:
            plt.show()

    def plotkrwkrow(
        self,
        mpl_ax=None,
        color="blue",
        alpha=1,
        linewidth=1,
        linestyle="-",
        label="",
        logyscale=False,
    ):
        """Plot krw and krow

        If the argument 'mpl_ax' is not supplied, a new plot
        window will be made. If supplied, it will draw on
        the specified axis."""
        import matplotlib.pyplot as plt
        import matplotlib

        if mpl_ax is None:
            matplotlib.style.use("ggplot")
            _, useax = plt.subplots()
        else:
            useax = mpl_ax
        if logyscale:
            useax.set_yscale("log")
            useax.set_ylim([1e-8, 1])
        self.table.plot(
            ax=useax,
            x="sw",
            y="krw",
            c=color,
            alpha=alpha,
            legend=None,
            label=label,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        self.table.plot(
            ax=useax,
            x="sw",
            y="krow",
            c=color,
            alpha=alpha,
            label=None,
            legend=None,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        if mpl_ax is None:
            plt.show()
