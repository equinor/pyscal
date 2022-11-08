"""Wateroil module"""

import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, interp1d

import pyscal
from pyscal.constants import EPSILON as epsilon
from pyscal.constants import MAX_EXPONENT, SWINTEGERS
from pyscal.utils.capillarypressure import simple_J
from pyscal.utils.relperm import crosspoint, estimate_diffjumppoint, truncate_zeroness
from pyscal.utils.string import comment_formatter, df2str

logger = pyscal.getLogger_pyscal(__name__)


class WaterOil(object):
    """A representation of two-phase properties for oil-water.

    Can hold relative permeability data, and capillary pressure.

    Parametrizations for relative permeability
     * Corey
     * LET

    For capillary pressure
     * Simplified J-function

    For object initialization, only saturation endpoints must be inputted,
    and saturation resolution. An optional string can be added as a 'tag'
    that can be used when outputting.

    Relative permeability and/or capillary pressure can be added through
    parametrizations, or from a dataframe (will incur interpolation).

    Can be dumped as include files for Eclipse/OPM and Nexus simulators.

    Args:
        swirr: Absolute minimal water saturation at infinite capillary
            pressure.
        swl: First water saturation point in generated table. Used
            for normalized saturations.
        swcr: Critical water saturation. Water will not be mobile before the
            water saturation is above this value.
        sorw: Residual oil saturation after water flooding. At this oil
            saturation, the oil has zero relative permeability.
        socr: Critical oil saturation, oil will not be mobile before the
            oil saturation is above socr. This parameter will default to sorw
            and is to be used larger than sorw for oil paleo zone modelling
            cases.
        h: Saturation step-length in the outputted table.
        tag: Optional string identifier, only used in comments.
        fast: Set to True if in order to skip some integrity checks
            and nice-to-have features. Not needed to set for normal pyscal
            runs, as speed is seldom crucial. Default False
    """

    def __init__(
        self,
        swirr: float = 0.0,
        swl: float = 0.0,
        swcr: float = 0.0,
        sorw: float = 0.0,
        socr: Optional[float] = None,
        h: Optional[float] = None,
        tag: str = "",
        fast: bool = False,
        _sgcr: Optional[float] = None,
        _sgl: Optional[float] = None,
    ) -> None:
        """Sets up the saturation range. Swirr is only relevant
        for the capillary pressure, not for relperm data.

        _sgcr and _sgl are only to be used by the GasWater object.
        """

        assert -epsilon < swirr < 1.0 + epsilon
        assert -epsilon < swl < 1.0 + epsilon
        assert -epsilon < swcr < 1.0 + epsilon
        assert -epsilon < sorw < 1.0 + epsilon
        if socr is not None:
            assert -epsilon < socr < 1.0 + epsilon

        if h is None:
            h = 0.01

        assert swl < 1 - sorw
        assert swcr < 1 - sorw
        assert swirr < 1 - sorw

        self.swcomment: str = ""

        h_min = 1.0 / float(SWINTEGERS)
        if h < h_min:
            logger.warning(
                "Requested saturation step length (%g) too small, reset to %g", h, h_min
            )
            self.h = h_min
        else:
            self.h = h

        if _sgcr is not None:
            self.sgcr = _sgcr
        if _sgl is not None:
            assert swl < 1 - _sgl < 1 + epsilon, "1-sgl must be between swl and 1"
            assert self.sgcr + epsilon > _sgl, "sgcr must be larger than sgl"
            assert sorw + epsilon > _sgl, "sorw/sgrw must be larger than sgl"
            self.sgl = truncate_zeroness(_sgl, name="_sgl")
        else:
            self.sgl = 0.0

        self.swirr = swirr
        self.swl = max(swl, swirr)  # Cannot allow swl < swirr. Warn?
        self.sorw = truncate_zeroness(sorw, name="sorw")
        if self.swl < swcr < self.swl + 1 / SWINTEGERS + epsilon:
            # Give up handling swcr so close to swl
            swcr = self.swl
        self.swcr = max(self.swl, swcr)  # Cannot allow swcr < swl. Warn?

        if socr is not None:
            self.socr = truncate_zeroness(socr, name="socr")
            if socr < self.sorw - epsilon:
                raise ValueError(
                    "socr must be equal to or larger than sorw, "
                    f"socr={socr}, sorw={self.sorw}"
                )
            if self.sorw - epsilon < self.socr < self.sorw + 1 / SWINTEGERS + epsilon:
                if self.sorw < self.socr or self.sorw > self.socr:
                    # Only warn if it seems the user actually tried to set socr
                    logger.warning("socr was close to sorw, reset to sorw")
                self.socr = self.sorw
        else:
            self.socr = self.sorw

        self.tag = tag
        self.fast = fast
        sw_list = (
            list(np.arange(self.swl, 1 - self.sgl, self.h))
            + [self.swcr]
            + [1 - self.sorw]
            + [1 - self.socr]
            + [1 - self.sgl]
        )
        sw_list.sort()  # Using default timsort on nearly sorted data.
        self.table = pd.DataFrame(sw_list, columns=["SW"])

        # Ensure that we do not have sw values that are too close
        # to each other, determined rougly by the distance 1/10000
        self.table["swint"] = list(
            map(int, list(map(round, self.table["SW"] * SWINTEGERS)))
        )
        self.table.drop_duplicates("swint", inplace=True)

        # Now, sw=1-sorw might be accidentaly dropped, so make sure we
        # have it by replacing the closest value by 1-sorw exactly
        sorwindex = (self.table["SW"] - (1 - self.sorw)).abs().sort_values().index[0]
        self.table.loc[sorwindex, "SW"] = 1 - self.sorw

        # Same for sw=1-socr
        socrindex = (self.table["SW"] - (1 - self.socr)).abs().sort_values().index[0]
        self.table.loc[socrindex, "SW"] = 1 - self.socr

        # Same for sw=swcr:
        swcrindex = (self.table["SW"] - (self.swcr)).abs().sort_values().index[0]
        self.table.loc[swcrindex, "SW"] = self.swcr

        # If sw=1 was dropped, then sorw was close to zero:
        if not np.isclose(self.table["SW"].max(), 1.0 - self.sgl):
            # Add it as an extra row:
            self.table.loc[len(self.table) + 1, "SW"] = 1.0 - self.sgl
            self.table.sort_values(by="SW", inplace=True)

        self.table.reset_index(inplace=True)
        self.table = self.table[["SW"]]  # Drop the swint column

        # Normalize for krw:
        self.table["SWN"] = (self.table["SW"] - self.swcr) / (1 - self.swcr - self.sorw)
        # Normalize for krow:
        self.table["SON"] = (1 - self.table["SW"] - self.socr) / (
            1 - self.swl - self.socr
        )

        # Different normalization for Sw used for capillary pressure
        self.table["SWNPC"] = (self.table["SW"] - swirr) / (1 - swirr)

        if _sgcr is None:
            self.swcomment = (
                f"-- swirr={self.swirr:g} swl={self.swl:g} "
                f"swcr={self.swcr:g} sorw={self.sorw:g}"
            )
            if self.socr > sorw:
                self.swcomment += f" socr={self.socr:g}"
        else:
            # When _sgcr is defined, this object is in use by GasWater
            # (sorw is aliased as sgrw, and sgcr is relevant)
            self.swcomment = (
                f"-- swirr={self.swirr:g} swl={self.swl:g} "
                f"swcr={self.swcr:g} sgrw={self.sorw:g} sgcr={self.sgcr:g}"
            )
            if _sgl is not None:
                self.swcomment += f" sgl={self.sgl:g}"
        self.swcomment += "\n"

        self.krwcomment = ""
        self.krowcomment = ""
        self.pccomment = ""

        logger.debug(
            "Initialized WaterOil with %s saturation points", str(len(self.table))
        )

    def add_fromtable(
        self,
        dframe: pd.DataFrame,
        swcolname: str = "SW",
        krwcolname: str = "KRW",
        krowcolname: str = "KROW",
        pccolname: str = "PCOW",
        krwcomment: str = "",
        krowcomment: str = "",
        pccomment: str = "",
        sorw: Optional[float] = None,
        socr: Optional[float] = None,
    ) -> None:
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
            dframe: containing data
            swcolname: column name with the saturation data in the dataframe df
            krwcolname: name of column in df with krw
            krowcolname: name of column in df with krow
            pccolname: name of column in df with capillary pressure data
            krwcomment: Inserted into comment
            krowcomment: Inserted into comment
            pccomment: Inserted into comment
            sorw: Explicit sorw. If None, it will be estimated from
                the numbers in krw
            socr: Explicit socr. If None, it will be estimated from
                the numbers in krow
        """
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

        # Typecheck/convert all numerical columns:
        for col in [swcolname, krwcolname, krowcolname, pccolname]:
            if col in dframe and not pd.api.types.is_numeric_dtype(dframe[col]):
                # Try to convert to numeric type
                try:
                    dframe[col] = dframe[col].astype(float)
                    logger.info("Converted column %s to numbers for fromtable()", col)
                except (TypeError, ValueError) as err:
                    raise ValueError(
                        f"Failed to parse column {col} as numbers for add_fromtable()"
                    ) from err

        if (dframe[swcolname].diff() < 0).any():
            raise ValueError("SW data not sorted")
        if sorw is None and krwcolname in dframe:
            sorw = float(dframe[swcolname].max()) - estimate_diffjumppoint(
                dframe, xcol=swcolname, ycol=krwcolname, side="right"
            )
            logger.info("Estimated sorw in tabular data to %f", sorw)
        else:
            sorw = 0
        if krwcolname in dframe:
            assert -epsilon <= sorw <= 1 + epsilon
            if dframe[krwcolname].max() > 1.0:
                raise ValueError("KRW is above 1 in incoming table")
            if dframe[krwcolname].min() < 0.0:
                raise ValueError("KRW is below 0 in incoming table")
            linearpart = dframe[swcolname] >= 1 - sorw
            nonlinearpart = dframe[swcolname] <= 1 - sorw  # (overlapping at sorw)
            if sum(linearpart) < 2:
                # A linear section of length 1 is not a linear section,
                # recategorize as nonlinear:
                linearpart = pd.Series([False] * len(linearpart))
                nonlinearpart = ~linearpart
                sorw = 0
            if sum(nonlinearpart) < 2:
                # A nonlinear section of length 1 is not a nonlinear section,
                # recategorize as linear:
                nonlinearpart = pd.Series([False] * len(nonlinearpart))
                linearpart = ~nonlinearpart
                sorw = 1 - float(self.table["SW"].min())
            if not np.isclose(dframe[swcolname].min(), self.table["SW"].min()):
                raise ValueError("Incompatible swl")
            # Verify that incoming data is increasing (or level):
            if not (dframe[krwcolname].diff().dropna() > -epsilon).all():
                raise ValueError("Incoming KRW not increasing")
            if sum(nonlinearpart) >= 2:
                pchip = PchipInterpolator(
                    dframe[nonlinearpart][swcolname].astype(float),
                    dframe[nonlinearpart][krwcolname].astype(float),
                )
                self.table.loc[self.table["SW"] <= 1 - sorw, "KRW"] = pchip(
                    self.table.loc[self.table["SW"] <= 1 - sorw, "SW"]
                )
            if sum(linearpart) >= 2:
                linearinterpolator = interp1d(
                    dframe[linearpart][swcolname].astype(float),
                    dframe[linearpart][krwcolname].astype(float),
                )
                self.table.loc[
                    self.table["SW"] >= 1 - sorw, "KRW"
                ] = linearinterpolator(
                    self.table.loc[self.table["SW"] >= 1 - sorw, "SW"]
                )
            self.table["KRW"].clip(lower=0.0, upper=1.0, inplace=True)
            self.sorw = sorw
            self.krwcomment = "-- krw from tabular input" + krwcomment + "\n"
            self.swcr = self.estimate_swcr()

        if krowcolname in dframe:
            if socr is None:
                socr = float(dframe[swcolname].max()) - estimate_diffjumppoint(
                    dframe, xcol=swcolname, ycol=krowcolname, side="right"
                )
                if socr > sorw + epsilon:
                    logger.info("Estimated socr in tabular data from krow to %s", socr)
                else:
                    socr = sorw
            assert -epsilon <= socr <= 1 + epsilon
            linearpart = dframe[swcolname] >= 1 - socr
            nonlinearpart = dframe[swcolname] <= 1 - socr  # (overlapping at sorw)
            if sum(linearpart) < 2:
                linearpart = pd.Series([False] * len(linearpart))
                nonlinearpart = ~linearpart
                socr = 0
            if sum(nonlinearpart) < 2:
                nonlinearpart = pd.Series([False] * len(nonlinearpart))
                linearpart = ~nonlinearpart
                socr = 1 - float(self.table["SW"].min())
            if not np.isclose(dframe[swcolname].min(), self.table["SW"].min()):
                raise ValueError("Incompatible swl")
            if not (dframe[krowcolname].diff().dropna() < epsilon).all():
                raise ValueError("Incoming KROW Not decreasing")
            if dframe[krowcolname].max() > 1.0:
                raise ValueError("KROW is above 1 in incoming table")
            if dframe[krowcolname].min() < 0.0:
                raise ValueError("KROW is below 0 in incoming table")
            if sum(nonlinearpart) >= 2:
                pchip = PchipInterpolator(
                    dframe.loc[nonlinearpart, swcolname].astype(float),
                    dframe.loc[nonlinearpart, krowcolname].astype(float),
                )
                self.table.loc[self.table["SW"] <= 1 - socr, "KROW"] = pchip(
                    self.table.loc[self.table["SW"] <= 1 - socr, "SW"]
                )
            if sum(linearpart) >= 2:
                linearinterpolator = interp1d(
                    dframe.loc[linearpart, swcolname].astype(float),
                    dframe.loc[linearpart, krowcolname].astype(float),
                )
                self.table.loc[
                    self.table["SW"] >= 1 - socr, "KROW"
                ] = linearinterpolator(
                    self.table.loc[self.table["SW"] >= 1 - socr, "SW"]
                )
            self.table["KROW"].clip(lower=0.0, upper=1.0, inplace=True)
            self.socr = socr
            self.krowcomment = "-- krow from tabular input" + krowcomment + "\n"
        if pccolname in dframe:
            # Incoming dataframe must cover the range:
            if dframe[swcolname].min() > self.table["SW"].min():
                raise ValueError("Too large swl for pc interpolation")
            if dframe[swcolname].max() < self.table["SW"].max():
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
            self.table["PC"] = pchip(self.table["SW"])
            if np.isnan(self.table["PC"]).any() or np.isinf(self.table["PC"]).any():
                raise ValueError("inf/nan in interpolated data, check input")
            self.pccomment = "-- pc from tabular input" + pccomment + "\n"

    def add_corey_water(
        self, nw: float = 2.0, krwend: float = 1.0, krwmax: Optional[float] = None
    ) -> None:
        """Add krw data through the Corey parametrization

        A column named 'krw' will be added. If it exists, it will
        be replaced.

        The Corey model applies for sw < 1 - sorw. For higher
        water saturations, krw is linear between krwend and krwmax.

        krwmax will be ignored if sorw is close to zero

        Args:
            nw: Corey parameter for water.
            krwend: value of krw at 1 - sorw.
            krwmax): maximal value at Sw=1. Default 1
        """
        assert 10 * epsilon < nw < MAX_EXPONENT
        if krwmax:
            assert 0 < krwend <= krwmax <= 1.0
        else:
            assert 0 < krwend <= 1.0

        self.table["KRW"] = krwend * self.table["SWN"] ** nw

        self.set_endpoints_linearpart_krw(krwend, krwmax)

        if not krwmax:
            krwmax = 1
        self.krwcomment = (
            f"-- Corey krw, nw={nw:g}, krwend={krwend:g}, krwmax={krwmax:g}\n"
        )

    def set_endpoints_linearpart_krw(
        self, krwend: float, krwmax: Optional[float] = None
    ) -> None:
        """Set linear parts of krw outside endpoints.

        Curve will be linear from [1 - sorw, 1] (from krwmax to krwend)
        and zero in [swl, swcr].

        This function is used by add_corey_water(), and perhaps by other
        utility functions. It should not be necessary for end-users.

        Args:
            krwend: krw at 1 - sorwr
            krwmax: krw at Sw=1. Default 1.
        """
        # The rows and indices involved in the linear section [1-sorw, 1]:
        linear_section_rows = self.table["SW"] > (1 - self.sorw - epsilon)
        linear_section_indices = self.table[linear_section_rows].index
        # (these lists are never empty)

        # Set krwend always (overrides krwmax if sorw=0)
        self.table.iloc[linear_section_indices[0]]["KRW"] = krwend

        if len(linear_section_indices) > 1:
            if krwmax is None:
                krwmax = 1
            self.table.iloc[linear_section_indices[-1]]["KRW"] = krwmax
        else:
            if krwmax is not None:
                logger.info("krwmax ignored when sorw is zero")

        # If the linear section is longer than two rows, do linear
        # interpolation inside for krw:
        if len(linear_section_indices) > 2:
            self.table.loc[linear_section_indices[1:-1], "KRW"] = np.nan
            interp_krw = (
                self.table[["SW", "KRW"]]
                .set_index("SW")
                .interpolate(method="index")["KRW"]
            )
            self.table.loc[:, "KRW"] = interp_krw.values

        # Left linear section is all zero:
        self.table.loc[self.table["SW"] < self.swcr, "KRW"] = 0

    def set_endpoints_linearpart_krow(self, kroend: float) -> None:
        """Set linear parts of krow outside endpoints

        Curve will be zero in [1 - socr, 1].

        This function is used by add_corey_water(), and perhaps by other
        utility functions. It should not be necessary for end-users.

        Args:
            kroend: value of kro at swcr
        """
        # Set to zero above socr (usually equal to sorw):
        self.table.loc[self.table["SW"] > 1 - self.socr - epsilon, "KROW"] = 0

        # Floating point issues can cause this to have become
        # slightly bigger than krowend.
        self.table.loc[self.table["KROW"] > kroend, "KROW"] = kroend

    def add_LET_water(
        self,
        l: float = 2.0,
        e: float = 2.0,
        t: float = 2.0,
        krwend: float = 1.0,
        krwmax: Optional[float] = None,
    ) -> None:
        """Add krw data through LET parametrization

        The LET model applies for sw < 1 - sgrw. For higher
        water saturations, krw is linear between krwend and krwmax.

        krwmax will be ignored if sorw is close to zero.

        Args:
            l: LET parameter
            e: LET parameter
            t: LET parameter
            krwend: value of krw at 1 - sorw
            krwmax: maximal value at Sw=1. Default 1
        """
        # Similar code in gasoil.add_LET_gas, but readability is
        # better by having them separate.
        # pylint: disable=duplicate-code
        assert epsilon < l < MAX_EXPONENT
        assert epsilon < e < MAX_EXPONENT
        assert epsilon < t < MAX_EXPONENT
        if krwmax:
            assert 0 < krwend <= krwmax <= 1.0
        else:
            assert 0 < krwend <= 1.0

        self.table["KRW"] = (
            krwend
            * self.table["SWN"] ** l
            / ((self.table["SWN"] ** l) + e * (1 - self.table["SWN"]) ** t)
        )
        # This equation is undefined for t a float and swn=1, set explicitly:
        self.table.loc[np.isclose(self.table["SWN"], 1.0), "KRW"] = krwend

        self.set_endpoints_linearpart_krw(krwend, krwmax)

        if not krwmax:
            krwmax = 1
        self.krwcomment = (
            f"-- LET krw, l={l:g}, e={e:g}, t={t:g}, "
            f"krwend={krwend:g}, krwmax={krwmax:g}\n"
        )

    def add_LET_oil(
        self,
        l: float = 2.0,
        e: float = 2.0,
        t: float = 2.0,
        kroend: float = 1,
    ) -> None:
        """
        Add kro data through LET parametrization

        Args:
            l: LET parameter
            e: LET parameter
            t: LET parameter
            kroend: value of kro at swl

        Returns:
            None (modifies object)
        """
        assert epsilon < l < MAX_EXPONENT
        assert epsilon < e < MAX_EXPONENT
        assert epsilon < t < MAX_EXPONENT
        assert 0 < kroend <= 1.0

        self.table["KROW"] = (
            kroend
            * self.table["SON"] ** l
            / ((self.table["SON"] ** l) + e * (1 - self.table["SON"]) ** t)
        )
        # This equation is undefined for t a float and son=1, set explicitly:
        self.table.loc[np.isclose(self.table["SON"], 1.0), "KROW"] = kroend

        self.table.loc[self.table["SW"] >= (1 - self.sorw), "KROW"] = 0

        self.set_endpoints_linearpart_krow(kroend)

        self.krowcomment = (
            f"-- LET krow, l={l:g}, e={e:g}, t={t:g}, kroend={kroend:g}\n"
        )

    def add_corey_oil(self, now: float = 2.0, kroend: float = 1.0) -> None:
        """Add kro data through the Corey parametrization

        Corey applies to the interval between swcr and 1 - sorw

        Curve is linear between swl and swcr, zero above 1 - sorw.

        Args:
            now: Corey exponent
            kroend: kro value at swcr
        Returns:
            None (modifies object)
        """
        assert epsilon < now < MAX_EXPONENT
        assert 0 < kroend <= 1.0

        self.table["KROW"] = kroend * self.table["SON"] ** now
        self.table.loc[self.table["SW"] >= (1 - self.sorw), "KROW"] = 0

        self.set_endpoints_linearpart_krow(kroend)

        self.krowcomment = f"-- Corey krow, now={now:g}, kroend={kroend:g}\n"

    def add_simple_J(
        self,
        a: float = 5.0,
        b: float = -1.5,
        poro_ref: float = 0.25,
        perm_ref: float = 100,
        drho: float = 300,
        g: float = 9.81,
    ) -> None:
        # pylint: disable=anomalous-backslash-in-string
        r"""Add capillary pressure function from a simplified J-function

        This is the RMS version of the coefficients *a* and *b*, the formula
        used is

        .. math::

            J = a S_w^b

        *J* is not dimensionless in this equation. The capillary pressure will
        be in bars.

        This is identical to the also seen formula

        .. math::

           J = 10^{b \log(S_w) + \log(a)}

        :math:`S_w` in this formula is normalized with respect to the *swirr* variable
        of the WaterOil object.

        Args:
            a: a coefficient
            b: b coefficient
            poro_ref: Reference porosity for scaling to Pc, between 0 and 1
            perm_ref: Reference permeability for scaling to Pc, in milliDarcy
            drho: Density difference between water and oil, in SI units kg/m³.
                Default value is 300
            g: Gravitational acceleration, in SI units m/s²,
                default value is 9.81

        Returns:
            None. Modifies pc column in self.table, using bar as pressure unit.
        """  # noqa
        assert g >= 0
        assert b < MAX_EXPONENT
        assert b > -MAX_EXPONENT
        assert 0.0 <= poro_ref <= 1.0
        assert perm_ref > 0.0

        if self.swl < epsilon:
            raise ValueError(
                "swl must be larger than zero to avoid infinite capillary pressure"
            )

        if b > 0:
            logger.warning(
                "positive b will give increasing capillary pressure with saturation"
            )

        # swnpc is a normalized saturation, but normalized with
        # respect to swirr, not to swl (the swirr here is sometimes
        # called 'swirra' - asymptotic swirr)

        self.table["PC"] = simple_J(
            self.table["SWNPC"], a, b, poro_ref, perm_ref, drho, g
        )
        self.pccomment = (
            "-- Simplified J-function for Pc; rms version, in bar\n--   "
            f"a={a:g}, b={b:g}, poro_ref={poro_ref:g}, perm_ref={perm_ref:g} mD,"
            f" drho={drho:g} kg/m^3, g={g:g} m/s^2\n"
        )

    def add_simple_J_petro(
        self,
        a: float,
        b: float,
        poro_ref: float = 0.25,
        perm_ref: float = 100,
        drho: float = 300,
        g: float = 9.81,
    ) -> None:
        # pylint: disable=anomalous-backslash-in-string
        r"""Add capillary pressure function from a simplified J-function

        This is the *petrophysical* version of the coefficients *a* and *b*, the formula
        used is

        .. math::

            J = \left(\frac{S_w}{a}\right)^{\frac{1}{b}}


        which is identical to

        .. math::

            J = 10^\frac{\log(S_w) - \log(a)}{b}

        *J* is not dimensionless in this equation.

        :math:`S_w` in this formula is normalized with respect to the *swirr* variable
        of the WaterOil object.

        Args:
            a: a coefficient, petrophysical version
            b: b coefficient, petrophysical version
            poro_ref: Reference porosity for scaling to Pc, between 0 and 1
            perm_ref: Reference permeability for scaling to Pc, in milliDarcy
            drho: Density difference between water and oil, in SI units kg/m³.
                Default value is 300
            g: Gravitational acceleration, in SI units m/s²,
                default value is 9.81

        Returns:
            None. Modifies pc column in self.table, using bar as pressure unit.
        """  # noqa
        assert g >= 0
        assert b < MAX_EXPONENT
        assert b > -MAX_EXPONENT
        assert 0.0 <= poro_ref <= 1.0
        assert perm_ref > 0.0

        if self.swl < epsilon:
            raise ValueError(
                "swl must be larger than zero to avoid infinite capillary pressure"
            )

        if b > 0:
            raise ValueError(
                "positive b will give increasing capillary pressure with saturation"
            )

        # Convert from "Petrophysical" a's and b's to "RMS" a's and b's:
        rms_a = math.pow(1 / a, 1 / b)
        rms_b = 1 / b

        # Use the other variant of this function for actual computation
        self.add_simple_J(rms_a, rms_b, poro_ref, perm_ref, drho, g)

        self.pccomment = (
            "-- Simplified J-function for Pc, petrophysical version, in bar \n--   "
            f"a={a:g}, b={b:g}, poro_ref={poro_ref:g}, perm_ref={perm_ref:g} mD, "
            f"drho={drho:g} kg/m^3, g={g:g} m/s^2\n"
        )

    def add_normalized_J(
        self, a: float, b: float, poro: float, perm: float, sigma_costau: float
    ) -> None:
        # Don't make this a raw string to avoid the \l warning,
        # it destroys the Latex-formatting in sphinx
        # pylint: disable=anomalous-backslash-in-string
        r"""
        Add capillary pressure in bar through a normalized J-function.

        .. math::

            p_c = \frac{\left(\frac{S_w}{a}\right)^{\frac{1}{b}}
            \sigma \cos \tau}{\sqrt{\frac{k}{\phi}}}

        The :math:`S_w` saturation used in the formula is normalized with respect
        to the swirr parameter.

        Args:
            a: a parameter
            b: b exponent (typically negative)
            poro: Porosity value, fraction between 0 and 1
            perm: Permeability value in mD
            sigma_costau: Interfacial tension in mN/m (typical value 30 mN/m)

        Returns:
            None. Modifies pc column in self.table, using bar as pressure unit.
        """  # noqa
        assert epsilon < abs(a) < MAX_EXPONENT
        assert epsilon < abs(b) < MAX_EXPONENT
        assert epsilon < poro <= 1.0
        assert epsilon < perm
        assert isinstance(sigma_costau, (int, float))

        if b < 0 and np.isclose(self.swirr, self.swl):
            raise ValueError("swl must be larger than swirr to avoid infinite p_c")

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
        tmp = (self.table["SWNPC"] / a) ** (1.0 / b)
        tmp = tmp / math.sqrt(perm_sq_meters / poro)
        tmp = tmp * sigma_costau / 1000  # Converting mN/m to N/m
        self.table["PC"] = tmp * pascal_to_bar
        self.pccomment = (
            "-- Capillary pressure from normalized J-function, in bar\n"
            f"-- a={a:g}, b={b:g}, poro={poro:g}, perm={perm:g} mD, "
            f"sigma_costau={sigma_costau:g} mN/m\n"
        )

    def add_skjaeveland_pc(
        self,
        cw: float,
        co: float,
        aw: float,
        ao: float,
        swr: Optional[float] = None,
        sor: Optional[float] = None,
    ):
        # pylint: disable=line-too-long
        """Add capillary pressure from the Skjæveland correlation,

        Doc: https://wiki.equinor.com/wiki/index.php/Res:The_Skjaeveland_correlation_for_capillary_pressure

        The implementation is unit independent, units are contained in the
        input constants.

        If swr and sor are not provided, it will be taken from the
        swirr and sorw. Only use different values here if you know
        what you are doing.

        Modifies or adds self.table["PC"] if succesful.
        Returns false if error occured.

        """  # noqa
        if cw < 0:
            raise ValueError("cw must be larger or equal to zero")
        if co > 0:
            raise ValueError("co must be less than zero")
        if aw <= 0:
            raise ValueError("aw must be larger than zero")
        if ao <= 0:
            raise ValueError("ao must be larger than zero")

        if swr is None:
            swr = self.swirr
        if sor is None:
            sor = self.sorw

        if swr < 0 or swr > 1:
            raise ValueError("swr must be contained in [0,1]")
        if sor < 0 or sor > 1:
            raise ValueError("sor must be contained in [0,1]")
        if swr >= 1 - sor:
            raise ValueError("swr (swirr) must be less than 1 - sor")

        self.pccomment = (
            "-- Skjæveland correlation for Pc;\n"
            f"-- cw={cw:g}, co={co:g}, aw={aw:g}, ao={ao:g}, swr={swr:g}, sor={sor:g}\n"
        )

        # swnpc is a normalized saturation, but normalized with
        # respect to swirr, not to swl (the swirr here is sometimes
        # called 'swirra' - asymptotic swirr)

        # swnpc is generated upon object initialization, but overwritten
        # here to most likely the same values.
        self.table["SWNPC"] = (self.table["SW"] - swr) / (1 - swr)

        # sonpc is almost like 'son', but swl is not used here:
        self.table["SONPC"] = (1 - self.table["SW"] - sor) / (1 - sor)

        # The Skjæveland correlation
        self.table.loc[self.table["SW"] < 1 - sor, "PC"] = cw / (
            self.table["SWNPC"] ** aw
        ) + co / (self.table["SONPC"] ** ao)

        # From 1-sor, the pc is not defined. Extrapolate constantly, and let
        # the non-monotonicity be fixed in the output generators.
        self.table["PC"].fillna(method="ffill", inplace=True)

    def add_LET_pc_pd(
        self,
        Lp: float,
        Ep: float,
        Tp: float,
        Lt: float,
        Et: float,
        Tt: float,
        Pcmax: float,
        Pct: float,
    ) -> None:
        # pylint: disable=line-too-long
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
        self.table["Ffpcow"] = (1 - self.table["SWNPC"]) ** Lp / (
            (1 - self.table["SWNPC"]) ** Lp + Ep * self.table["SWNPC"] ** Tp
        )

        # The gradual rise part:
        self.table["Ftpcow"] = self.table["SWNPC"] ** Lt / (
            self.table["SWNPC"] ** Lt + Et * (1 - self.table["SWNPC"]) ** Tt
        )

        # Putting it together:
        self.table["PC"] = (
            (Pcmax - Pct) * self.table["Ffpcow"] - Pct * self.table["Ftpcow"] + Pct
        )

        # Special handling of the interval [0,swirr]
        self.table.loc[self.table["SWN"] < epsilon, "PC"] = Pcmax
        self.pccomment = (
            "-- LET correlation for primary drainage Pc;\n"
            f"-- Lp={Lp:g}, Ep={Ep:g}, Tp={Tp:g}, "
            f"Lt={Lt:g}, Et={Et:g}, Tt={Tt:g}, "
            f"Pcmax={Pcmax:g}, Pct={Pct:g}\n"
        )

    def add_LET_pc_imb(
        self,
        Ls: float,
        Es: float,
        Ts: float,
        Lf: float,
        Ef: float,
        Tf: float,
        Pcmax: float,
        Pcmin: float,
        Pct: float,
    ) -> None:
        # pylint: disable=line-too-long
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
        self.table["SWNPCO"] = (self.table["SW"] - self.swirr) / (
            1 - self.sorw - self.swirr
        )

        # The "forced part"
        self.table["Fficow"] = self.table["SWNPCO"] ** Lf / (
            self.table["SWNPCO"] ** Lf + Ef * (1 - self.table["SWNPCO"]) ** Tf
        )

        # The spontaneous part:
        self.table["Fsicow"] = (1 - self.table["SWNPCO"]) ** Ls / (
            (1 - self.table["SWNPCO"]) ** Ls + Es * self.table["SWNPCO"] ** Ts
        )

        # Putting it together:
        self.table["PC"] = (
            (Pcmax - Pct) * self.table["Fsicow"]
            + (Pcmin - Pct) * self.table["Fficow"]
            + Pct
        )

        # Special handling of the interval [0,swirr]
        self.table.loc[self.table["SWNPCO"] < epsilon, "PC"] = Pcmax
        # and [1-sorw,1]
        self.table.loc[self.table["SWNPCO"] > 1 - epsilon, "PC"] = Pcmin
        self.pccomment = (
            "-- LET correlation for imbibition Pc;\n"
            f"-- Ls={Ls:g}, Es={Es:g}, Ts={Ts:g}, "
            f"Lf={Lf:g}, Ef={Ef:g}, Tf={Tf:g}, "
            f"Pcmax={Pcmax:g}, Pcmin={Pcmin:g}, Pct={Pct:g}\n"
        )

    def estimate_sorw(self, curve: str = "KRW") -> float:
        """Estimate sorw of the current krw data.

        This is mostly relevant when add_fromtable() has been used.
        sorw is estimated by searching for a linear part in krw downwards from sw=1.
        In practice it is impossible to infer sorw = 0, since we are limited by
        h, and the last segment from sw=1-h to sw=1 can always be assumed linear.
        Expect sorw = h if the real sorw = 0, but do not depend that it might
        not return zero in the future (one could argue that sorw = h should be
        specially treated to mean sorw = 0)

        If the curve is linear everywhere, sorw will be returned as 1 - (swl + h)

        krow is not used, and should probably not be, as it can be very close to
        zero before approaching sorw.

        Args:
            curve: Colum name of column to use, default is krw.
                If this is all linear, but krow is not, you might be better off
                with krow
        Returns:
            The estimated sorw.
        """
        assert curve in self.table
        assert self.table[curve].sum() > 0
        return self.table["SW"].max() - estimate_diffjumppoint(
            self.table, xcol="SW", ycol=curve, side="right"
        )

    def estimate_socr(self) -> float:
        """Estimate socr from the current kro data."""
        assert self.table["KROW"].sum() > 0
        return self.table["SW"].max() - estimate_diffjumppoint(
            self.table, xcol="SW", ycol="KROW", side="right"
        )

    def estimate_swcr(self, curve: str = "KRW") -> float:
        """Estimate swcr of the current krw data.

        kwcr is estimated by searching for a linear part in krw upwards from sw=swl.
        In practice it is impossible to infer swcr = 0, since we are limited by
        h, and the first segment is assumed linear anyways.

        If the curve is linear everywhere, swcr can end up at the right end
        of your saturation interval.

        Args:
            curve: Colum name of column to use, default is krow.
                If this is all linear, but krw is not, you might be better off
                with krw
        Returns:
            The estimated sgcr.
        """
        assert curve in self.table
        assert self.table[curve].sum() > 0
        return estimate_diffjumppoint(self.table, xcol="SW", ycol=curve, side="left")

    def crosspoint(self) -> float:
        """Locate and return the saturation point where krw = krow

        Accuracy of this crosspoint depends on the resolution chosen
        when initializing the saturation range

        Returns:
            The water saturation where krw == krow, for relperm
            linearly interpolated in water saturation.
        """
        return crosspoint(self.table, "SW", "KRW", "KROW")

    def selfcheck(self, mode: str = "SWOF") -> bool:
        """Check validities of the data in the table.

        An unfinished object will return False.

        If you call SWOF, this function must not return False

        This function should not throw an exception, but capture
        the error and give an error.

        Args:
            mode: "SWOF" or "SWFN". If SWFN, krow is not required.
        """
        error = False
        if "KRW" not in self.table:
            logger.error("krw data not found")
            error = True
        if not (self.table["SW"].diff().dropna().round(10) > -epsilon).all():
            logger.error("SW data not strictly increasing")
            error = True
        if (
            "KRW" in self.table
            and not (self.table["KRW"].diff().dropna().round(10) >= -epsilon).all()
        ):
            logger.error("KRW data not monotonically increasing")
            error = True
        if mode != "SWFN":
            if "KROW" not in self.table:
                logger.error("KROW data not found")
                error = True

            if (
                "KROW" in self.table.columns
                and not (self.table["KROW"].diff().dropna().round(10) <= epsilon).all()
            ):
                # In normal Eclipse runs, krow needs to be level or decreasing.
                # In hysteresis runs, it needs to be strictly decreasing, that must
                # be the users responsibility.
                logger.error("KROW data not level or monotonically decreasing")
                error = True
        if "PC" in self.table.columns and self.table["PC"][0] > -epsilon:
            if not (self.table["PC"].diff().dropna().round(10) < epsilon).all():
                logger.error("PC data not strictly decreasing")
                error = True
        if "PC" in self.table.columns and np.isnan(self.table["PC"]).any():
            logger.error("pc data contains NaN")
            error = True
        if "PC" in self.table.columns and np.isinf(self.table["PC"].max()):
            logger.error("pc goes to infinity. Maybe swirr=swl?")
            error = True
        for col in list(set(["SW", "KRW", "KROW"]) & set(self.table.columns)):
            if not (
                (round(min(self.table[col]), 10) >= -epsilon)
                and (round(max(self.table[col]), 10) <= 1 + epsilon)
            ):
                logger.error("%s data should be contained in [0,1]", col)
                error = True
        if error:
            return False
        logger.debug("WaterOil object is checked to be valid")
        return True

    def SWOF(self, header: bool = True, dataincommentrow: bool = True) -> str:
        """
        Produce SWOF input for Eclipse reservoir simulator.

        The columns sw, krw, krow and pc are outputted and
        formatted accordingly.

        Meta-information for the tabulated data are printed
        as Eclipse comments.

        Args:
            header: Indicate whether the SWOF string should
                be emitted. If you have multiple SATNUMs, you should
                set this to True only for the first (or False for all,
                and emit the SWOF yourself). Default True
            dataincommentrow: Wheter metadata should
                be printed. Defualt True

        """
        if not self.fast and not self.selfcheck():
            # selfcheck failed and has issued an error message
            return ""
        string = ""
        if header:
            string += "SWOF\n"
        string += comment_formatter(self.tag)
        string += "-- pyscal: " + str(pyscal.__version__) + "\n"
        if "PC" not in self.table.columns:
            self.table["PC"] = 0.0
            self.pccomment = "-- Zero capillary pressure\n"
        if dataincommentrow:
            string += self.swcomment
            string += self.krwcomment
            string += self.krowcomment
            if not self.fast:
                string += f"-- krw = krow @ sw={self.crosspoint():1.5f}\n"
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
        string += df2str(
            self.table[["SW", "KRW", "KROW", "PC"]],
            monotonicity={
                "KROW": {"sign": -1, "lower": 0, "upper": 1},
                "KRW": {"sign": 1, "lower": 0, "upper": 1},
                "PC": {"sign": -1, "allowzero": True},
            }
            if not self.fast
            else None,
        )
        string += "/\n"  # Empty line at the end
        return string

    def SWFN(
        self,
        header: bool = True,
        dataincommentrow: bool = True,
        swcomment: Optional[str] = None,
        crosspointcomment: Optional[str] = None,
    ) -> str:
        """Return a SWFN keyword with data to Eclipse

        The columns sw, krw and pc are outputted and formatted
        accordingly.

        Meta-information for the tabulated data are printed
        as Eclipse comments.

        Args:
            header: boolean for whether the SWFN string should be emitted.
                If you have multiple satnums, you should have True only
                for the first (or False for all, and emit the SWFN yourself).
                Defaults to True.
            dataincommentrow: boolean for wheter metadata should be printed,
                defaults to True.
            swcomment: String to be used for swcomment, overrides what
                this object can provide. Used by GasWater
            crosspointcomment: String to be used for crosspoint comment
                string, overrides what this object can provide. Used by GasWater.
                If None, it will be computed, use empty string to avoid.
        """
        if not self.selfcheck(mode="SWFN"):
            # selfcheck will print errors/warnings
            return ""
        string = ""
        if "PC" not in self.table.columns:
            self.table["PC"] = 0.0
            self.pccomment = "-- Zero capillary pressure\n"
        if header:
            string += "SWFN\n"
        string += comment_formatter(self.tag)
        string += "-- pyscal: " + str(pyscal.__version__) + "\n"
        if dataincommentrow:
            if swcomment is not None:
                string += swcomment
            else:
                string += self.swcomment
            string += self.krwcomment
            if crosspointcomment is None:
                if "KROW" in self.table.columns and not self.fast:
                    string += f"-- krw = krow @ sw={self.crosspoint():1.5f}\n"
            else:
                string += crosspointcomment
            string += self.pccomment
        width = 10
        string += (
            "-- "
            + "SW".ljust(width - 3)
            + "KRW".ljust(width)
            + "PC".ljust(width)
            + "\n"
        )
        string += df2str(
            self.table[["SW", "KRW", "PC"]],
            monotonicity={
                "KRW": {"sign": 1, "lower": 0, "upper": 1},
                "PC": {"sign": -1, "allowzero": True},
            }
            if not self.fast
            else None,
        )
        string += "/\n"  # Empty line at the end
        return string

    def WOTABLE(self, header: bool = True, dataincommentrow: bool = True) -> str:
        """Return a string for a Nexus WOTABLE"""
        string = ""
        if "PC" not in self.table.columns:
            self.table["PC"] = 0.0
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
                string += f"! krw = krow @ sw={self.crosspoint():1.5f}\n"
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
        string += df2str(
            self.table[["SW", "KRW", "KROW", "PC"]],
            monotonicity={
                "KROW": {"sign": -1, "lower": 0, "upper": 1},
                "KRW": {"sign": 1, "lower": 0, "upper": 1},
                "PC": {"sign": -1, "allowzero": True},
            }
            if not self.fast
            else None,
        )
        return string

    def plotpc(
        self,
        mpl_ax=None,
        color: str = "blue",
        alpha: float = 1,
        linewidth: int = 1,
        linestyle: str = "-",
        label: str = "",
        logyscale: bool = False,
    ) -> None:
        """Plot capillary pressure (pc)

        If mpl_ax is supplied, the curve will be drawn on
        that, if not, a new axis (plot) will be made
        """
        # pylint: disable=import-outside-toplevel
        # Lazy import for speed reaons.
        import matplotlib
        import matplotlib.pyplot as plt

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
            x="SW",
            y="PC",
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
        color: str = "blue",
        alpha: float = 1,
        linewidth: float = 1,
        linestyle: str = "-",
        marker: Optional[str] = None,
        label: str = "",
        logyscale: bool = False,
    ) -> None:
        """Plot krw and krow

        If the argument 'mpl_ax' is not supplied, a new plot
        window will be made. If supplied, it will draw on
        the specified axis."""
        # pylint: disable=import-outside-toplevel
        # Lazy import for speed reaons.
        import matplotlib
        import matplotlib.pyplot as plt

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
            x="SW",
            y="KRW",
            c=color,
            alpha=alpha,
            legend=None,
            label=label,
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
        )
        self.table.plot(
            ax=useax,
            x="SW",
            y="KROW",
            c=color,
            alpha=alpha,
            label=None,
            legend=None,
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
        )
        if mpl_ax is None:
            plt.show()
