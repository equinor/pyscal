# -*- coding: utf-8 -*-
from __future__ import division, absolute_import
from __future__ import print_function

import math
import logging
import numpy as np
import pandas as pd

from pyscal.constants import EPSILON as epsilon
from pyscal.constants import SWINTEGERS
from pyscal.constants import MAX_EXPONENT


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
    """

    def __init__(self, swirr=0.0, swl=0.0, swcr=0.0, sorw=0.0, h=0.01, tag=""):
        """Sets up the saturation range. Swirr is only relevant
        for the capillary pressure, not for relperm data."""

        assert -epsilon < swirr < 1.0 + epsilon
        assert -epsilon < swl < 1.0 + epsilon
        assert -epsilon < swcr < 1.0 + epsilon
        assert -epsilon < sorw < 1.0 + epsilon
        assert epsilon < h <= 1
        assert swl < 1 - sorw
        assert swcr < 1 - sorw
        assert swirr < 1 - sorw
        if not isinstance(tag, str):
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
        sw = list(np.arange(self.swl, 1 - sorw, h)) + [self.swcr] + [1 - sorw] + [1]
        self.table = pd.DataFrame(sw, columns=["sw"])
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

        logging.info(
            "Initialized WaterOil with %s saturation points", str(len(self.table))
        )

    def add_oilwater_fromtable(self, *args, **kwargs):
        logging.warning("add_oilwater_fromtable() is deprecated, use add_fromtable()")
        self.add_fromtable(*args, **kwargs)

    def add_fromtable(
        self,
        df,
        swcolname="Sw",
        krwcolname="krw",
        krowcolname="krow",
        pccolname="pcow",
        krwcomment="",
        krowcomment="",
        pccomment="",
    ):
        """Interpolate relpermdata from a dataframe.

        The saturation range with endpoints must be set up beforehand,
        and must be compatible with the tabular input. The tabular
        input will be interpolated to the initialized Sw-table

        If you have krw and krow in different dataframes, call this
        function twice

        Calling function is responsible for checking if any data was
        actually added to the table.

        The python package ecl2df has a tool for converting Eclipse input
        files to dataframes.

        Args:
            df: Pandas dataframe containing data
            swcolname: string, column name with the saturation data in the dataframe.
            krwcolname: string, name of the column with krw
            krowcolname: string
            pccolname: string
            krwcomment: string
            krowcomment: string
            pccomment: string
        """
        from scipy.interpolate import PchipInterpolator

        if swcolname not in df:
            logging.critical(
                swcolname + " not found in dataframe, can't read table data"
            )
            raise ValueError
        if krwcolname in df:
            if not np.isclose(df[swcolname].min(), self.table["sw"].min()):
                raise ValueError("Incompatible swl")
            # Verify that incoming data is increasing (or level):
            if not (df[krwcolname].diff().dropna() > -epsilon).all():
                raise ValueError("Incoming krw not increasing")
            pchip = PchipInterpolator(
                df[swcolname].astype(float), df[krwcolname].astype(float)
            )
            self.table["krw"] = pchip(self.table["sw"])
            self.krwcomment = "-- krw from tabular input" + krwcomment + "\n"
        if krowcolname in df:
            if not np.isclose(df[swcolname].min(), self.table["sw"].min()):
                raise ValueError("Incompatible swl")
            if not (df[krowcolname].diff().dropna() < epsilon).all():
                raise ValueError("Incoming krow not decreasing")
            pchip = PchipInterpolator(
                df[swcolname].astype(float), df[krowcolname].astype(float)
            )
            self.table["krow"] = pchip(self.table.sw)
            self.krowcomment = "-- krow from tabular input" + krowcomment + "\n"
        if pccolname in df:
            # Incoming dataframe must cover the range:
            if df[swcolname].min() > self.table["sw"].min():
                raise ValueError("Too large swl for pc interpolation")
            if df[swcolname].max() < self.table["sw"].max():
                raise ValueError("max(sw) of incoming data not large enough")
            if np.isinf(df[pccolname]).any():
                logging.warning(
                    (
                        "Infinity pc values detected. Will be dropped. "
                        "Risk of extrapolation"
                    )
                )
            df = df.replace([np.inf, -np.inf], np.nan)
            df.dropna(subset=[pccolname], how="all", inplace=True)
            # If nonzero, then it must be decreasing:
            if df[pccolname].abs().sum() > 0:
                if not (df[pccolname].diff().dropna() < 0.0).all():
                    raise ValueError("Incoming pc not decreasing")
            pchip = PchipInterpolator(
                df[swcolname].astype(float), df[pccolname].astype(float)
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

        self._handle_endpoints_linearpart_water(krwend, krwmax)

        if not krwmax:
            krwmax = 1
        self.krwcomment = "-- Corey krw, nw=%g, krwend=%g, krwmax=%g\n" % (
            nw,
            krwend,
            krwmax,
        )

    def _handle_endpoints_linearpart_water(self, krwend, krwmax=None):
        """Internal utility function to handle krw
        around endpoints.

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
            if krwmax:
                logging.warning("krwmax ignored when sorw is zero")
            self.table.loc[self.table["sw"] > (1 - self.sorw - epsilon), "krw"] = krwend
        self.table.loc[np.isclose(self.table["sw"], 1 - self.sorw), "krw"] = krwend
        self.table.loc[self.table.sw < self.swcr, "krw"] = 0

    def _handle_endpoints_linearpart_oil(self, kroend, kromax=None):
        """Internal utility function to handle krow
        around endpoints.

        Ensures we obey the endpoints, and linearity where needed"""

        # Linear curve between swl and swcr:
        self.table.loc[self.table["son"] > 1.0 + epsilon, "krow"] = np.nan
        if self.swcr < self.swl + self.h:
            if kromax:
                logging.warning("kromax ignored when swcr is close to swl")
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
        self._handle_endpoints_linearpart_water(krwend, krwmax)

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
        self.table.loc[self.table.sw >= (1 - self.sorw), "krow"] = 0

        self._handle_endpoints_linearpart_oil(kroend, kromax)

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

        self._handle_endpoints_linearpart_oil(kroend, kromax)

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
            logging.error(
                "swl must larger than zero to avoid infinite capillary pressure"
            )
            raise ValueError

        if b > 0:
            logging.warning(
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
            + "a=%g, b=%g, poro_ref=%g, perm_ref=%g mD, drho=%g kg/m³, g=%g m/s²\n"
            % (a, b, poro_ref, perm_ref, drho, g)
        )

    def add_normalized_J(self, a, b, poro, perm, sigma_costau):
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
            logging.error("swl must be set larger than swirr to avoid infinite p_c")
            raise ValueError("swl must be larger than swirr")

        if abs(b) < 0.01:
            logging.warning(
                "b exponent is very small, risk of infinite capillary pressure"
            )

        if abs(a) < 0.01:
            logging.warning(
                "a parameter is very small, risk of infinite capillary pressure"
            )

        if abs(a) > 5:
            logging.warning(
                "a parameter is very high, risk of infinite capillary pressure"
            )

        # 1 atm is equivalent to 101325 pascal = 1.01325 bar
        # pascal_to_atm = 1.0 / 101325.0  # = 9.86923267e-6
        pascal_to_bar = 1e-5

        perm_D = perm / 1000
        perm_sq_meters = perm_D * 9.869233e-13
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
            logging.error("cw must be larger or equal to zero")
            inputerror = True
        if co > 0:
            logging.error("co must be less than zero")
            inputerror = True
        if aw <= 0:
            logging.error("aw must be larger than zero")
            inputerror = True
        if ao <= 0:
            logging.error("ao must be larger than zero")
            inputerror = True

        if swr is None:
            swr = self.swirr
        if sor is None:
            sor = self.sorw

        if swr >= 1 - sor:
            logging.error("swr (swirr) must be less than 1 - sor")
            inputerror = True
        if swr < 0 or swr > 1:
            logging.error("swr must be contained in [0,1]")
            inputerror = True
        if sor < 0 or sor > 1:
            logging.error("sor must be contained in [0,1]")
            inputerror = True
        if inputerror:
            return False
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
            + "-- Lp=%g, Ep=%g, Tp=%g, Lt=%g, Et=%g, Tt=%g, Pcmax=%g, Pct=%g\n"
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
            "-- LET correlation for imbibition Pc;\n -- "
            + "Ls=%g, Es=%g, Ts=%g, Lf=%g, Ef=%g, Tf=%g, Pcmax=%g, Pcmin=%g, Pct=%g\n"
            % (Ls, Es, Ts, Lf, Ef, Tf, Pcmax, Pcmin, Pct)
        )

    def selfcheck(self):
        """Check validities of the data in the table.

        If you call SWOF, this function must not return False
        """
        error = False
        if not (self.table.sw.diff().dropna().round(10) > -epsilon).all():
            logging.error("sw data not strictly increasing")
            error = True
        if not (self.table.krw.diff().dropna().round(10) >= -epsilon).all():
            logging.error("krw data not monotonically increasing")
            error = True
        if (
            "krow" in self.table.columns
            and not (self.table.krow.diff().dropna().round(10) <= epsilon).all()
        ):
            logging.error("krow data not monotonically decreasing")
            error = True
        if "pc" in self.table.columns and self.table.pc[0] > -epsilon:
            if not (self.table.pc.diff().dropna().round(10) < epsilon).all():
                logging.error("pc data not strictly decreasing")
                error = True
        if "pc" in self.table.columns and np.isinf(self.table.pc.max()):
            logging.error("pc goes to infinity. Maybe swirr=swl?")
            error = True
        for col in list(set(["sw", "krw", "krow"]) & set(self.table.columns)):
            if not (
                (round(min(self.table[col]), 10) >= -epsilon)
                and (round(max(self.table[col]), 10) <= 1 + epsilon)
            ):
                logging.error("%s data should be contained in [0,1]", col)
                error = True
        if error:
            return False
        else:
            logging.info("WaterOil object is checked to be valid")
            return True

    def SWOF(self, header=True, dataincommentrow=True):
        if not self.selfcheck():
            return
        string = ""
        if "pc" not in self.table.columns:
            self.table["pc"] = 0
            self.pccomment = "-- Zero capillary pressure\n"
        if header:
            string += "SWOF\n"
        string += "-- " + self.tag + "\n"
        string += "-- Sw Krw Krow Pc\n"
        if dataincommentrow:
            string += self.swcomment
            string += self.krwcomment
            string += self.krowcomment
            string += "-- krw = krow @ sw=%1.5f\n" % self.crosspoint()
            string += self.pccomment
        string += self.table[["sw", "krw", "krow", "pc"]].to_csv(
            sep=" ", float_format="%1.7f", header=None, index=False
        )
        string += "/\n"  # Empty line at the end
        return string

    def SWFN(self, header=True, dataincommentrow=True):
        if not self.selfcheck():
            return
        string = ""
        if "pc" not in self.table.columns:
            self.table["pc"] = 0
            self.pccomment = "-- Zero capillary pressure\n"
        if header:
            string += "SWFN\n"
        string += "-- " + self.tag + "\n"
        string += "-- Sw Krw Pc\n"
        if dataincommentrow:
            string += self.swcomment
            string += self.krwcomment
            if "krow" in self.table.columns:
                string += "-- krw = krow @ sw=%1.5f\n" % self.crosspoint()
            string += self.pccomment
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
        if dataincommentrow:
            string += self.swcomment.replace("--", "!")
            string += self.krwcomment.replace("--", "!")
            string += self.krowcomment.replace("--", "!")
            string += "! krw = krow @ sw=%1.5f\n" % self.crosspoint()
            string += self.pccomment.replace("--", "!")
        string += self.table[["sw", "krw", "krow", "pc"]].to_csv(
            sep=" ", float_format="%1.7f", header=None, index=False
        )
        return string

    def plotpc(
        self,
        ax=None,
        color="blue",
        alpha=1,
        label=None,
        linewidth=1,
        linestyle="-",
        logscale=False,
    ):
        """Plot capillary pressure (pc) a supplied matplotlib axis"""
        import matplotlib.pyplot as plt
        import matplotlib

        if not ax:
            matplotlib.style.use("ggplot")
            _, useax = plt.subplots()
        else:
            useax = ax
        self.table.plot(
            ax=useax,
            x="sw",
            y="pc",
            c=color,
            alpha=alpha,
            legend=None,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        if logscale:
            useax.set_yscale("log")
        if not ax:
            plt.show()

    def plotkrwkrow(
        self, ax=None, color="blue", alpha=1, label=None, linewidth=1, linestyle="-"
    ):
        """Plot krw and krow on a supplied matplotlib axis"""
        import matplotlib.pyplot as plt
        import matplotlib

        if not ax:
            matplotlib.style.use("ggplot")
            _, useax = plt.subplots()
        else:
            useax = ax
        self.table.plot(
            ax=useax,
            x="sw",
            y="krw",
            c=color,
            alpha=alpha,
            legend=None,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        self.table.plot(
            ax=useax,
            x="sw",
            y="krow",
            c=color,
            alpha=alpha,
            legend=None,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        if not ax:
            plt.show()

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
        tmp.interpolate(method="slinear", inplace=True)

        return tmp[np.isclose(tmp.index, 0.0)].sw.values[0]
