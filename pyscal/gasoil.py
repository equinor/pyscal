""" Representing a GasOil object """
from __future__ import division, absolute_import
from __future__ import print_function

import logging
import six

import numpy as np
import pandas as pd

from scipy.interpolate import PchipInterpolator

import pyscal
from pyscal import utils
from pyscal.constants import EPSILON as epsilon
from pyscal.constants import SWINTEGERS, MAX_EXPONENT

logging.basicConfig()
logger = logging.getLogger(__name__)


class GasOil(object):
    """Object to represent two-phase properties for gas and oil.

    Parametrizations available for relative permeability:

     * Corey
     * LET

    or data can alternatively be read in from tabulated data
    (as Pandas DataFrame).

    No support (yet) to add capillary pressure.

    krgend is by default anchored both to `1-swl-sorg`, but can be set to
    anchor to `1-swl` instead. If the krgendanchor argument is something
    else than the string `sorg`, it will be anchored to `1-swl`.

    Arguments:
        swirr (float): Absolute minimal water saturation at infinite capillary
            pressure.
            Not in use currently, except for in informational headers and
            for consistency checks.
        swl (float): First water saturation point in water tables. In GasOil, it
            is used to obtain the normalized oil and gas saturation.
        sgcr (float): Critical gas saturation. Gas will not be mobile before the
            gas saturation is above this value.
        sorg (float): Residual oil saturation after gas flooding. At this oil
            saturation, the oil has zero relative permeability.
        krgendanchor (str): Set to `sorg` (default) or something else, where to
            anchor `krgend`. If `sorg`, then the normalized gas
            saturation will be equal to 1 at `1 - swl - sorg`,
            if not, it will be 1 at `1 - swl`. If `sorg` is zero
            it does not matter. krgmax is only relevant when this anchor is sorg.
        h (float): Saturation step-length in the outputted table.
        tag (str): Optional string identifier, only used in comments.
        fast (bool): Set to True if in order to skip some integrity checks
            and nice-to-have features. Not needed to set for normal pyscal
            runs, as speed is seldom crucial. Default False
    """

    def __init__(
        self,
        swirr=0,
        sgcr=0.0,
        h=0.01,
        swl=0.0,
        sorg=0.0,
        tag="",
        krgendanchor="sorg",
        fast=False,
    ):
        if h is None:
            h = 0.01
        assert -epsilon < swirr < 1.0 + epsilon
        assert -epsilon < sgcr < 1
        assert -epsilon < swl < 1
        assert -epsilon < sorg < 1
        if not isinstance(tag, six.string_types):
            tag = ""
        if krgendanchor is None:
            krgendanchor = ""

        assert isinstance(krgendanchor, six.string_types)

        h_min = 1.0 / float(SWINTEGERS)
        if h < h_min:
            logger.warning(
                "Requested saturation step length (%g) too small, reset to %g", h, h_min
            )
            self.h = h_min
        else:
            self.h = h

        swl = max(swl, swirr)  # Can't allow swl < swirr, should we warn user?
        self.swl = swl
        self.swirr = swirr
        if not np.isclose(sorg, 0.0) and sorg < 1.0 / SWINTEGERS:
            # Too small but nonzero sorg gives too many numerical issues.
            logger.warning("sorg was close to zero, set to zero")
            sorg = 0.0
        self.sorg = sorg

        self.sgcr = sgcr
        self.tag = tag

        if not 1 - sorg - swl > 0:
            raise Exception(
                "No saturation range left " + "after endpoints, check input"
            )
        if krgendanchor in ["sorg", ""]:
            self.krgendanchor = krgendanchor
        else:
            logger.warning("Unknown krgendanchor %s, ignored", str(krgendanchor))
            self.krgendanchor = ""

        self.fast = fast

        if np.isclose(sorg, 0.0) and self.krgendanchor == "sorg":
            self.krgendanchor = ""  # This is critical to avoid bugs due to numerics.

        sg_list = (
            [0]
            + [sgcr]
            + list(np.arange(sgcr + self.h, 1 - swl, self.h))
            + [1 - sorg - swl]
            + [1 - swl]
        )
        sg_list.sort()
        self.table = pd.DataFrame(sg_list, columns=["sg"])
        self.table["sgint"] = list(
            map(int, list(map(round, self.table["sg"] * SWINTEGERS)))
        )
        self.table.drop_duplicates("sgint", inplace=True)

        # Now sg=1-sorg-swl might be accidentally dropped, so make sure we
        # have it by replacing the closest value by 1 - sorg exactly
        sorgindex = (
            (self.table["sg"] - (1 - self.sorg - self.swl)).abs().sort_values().index[0]
        )
        self.table.loc[sorgindex, "sg"] = 1 - self.sorg - self.swl

        # Same for sg=sgcr
        sgcrindex = (self.table["sg"] - (self.sgcr)).abs().sort_values().index[0]
        self.table.loc[sgcrindex, "sg"] = self.sgcr
        if sgcrindex == 0 and sgcr > 0.0:
            # Need to conserve sg=0
            zero_row = pd.DataFrame({"sg": 0}, index=[0])
            self.table = pd.concat([zero_row, self.table], sort=False).reset_index(
                drop=True
            )

        # If sg=1-swl was dropped, then sorg was close to zero:
        if not np.isclose(self.table["sg"].max(), 1 - self.swl):
            # Add it as an extra row:
            self.table.loc[len(self.table) + 1, "sg"] = 1 - self.swl
            self.table.sort_values(by="sg", inplace=True)
        # Ensure the value closest to 1-swl is actually 1-swl:
        swl_right_index = (
            (self.table["sg"] - (1 - self.swl)).abs().sort_values().index[0]
        )
        self.table.loc[swl_right_index, "sg"] = 1 - self.swl

        self.table.reset_index(inplace=True)
        self.table = self.table[["sg"]]
        self.table["sl"] = 1 - self.table["sg"]
        if krgendanchor == "sorg":
            # Normalized sg (sgn) is 0 at sgcr, and 1 at 1-swl-sorg
            assert 1 - swl - sgcr - sorg > epsilon
            self.table["sgn"] = (self.table["sg"] - sgcr) / (1 - swl - sgcr - sorg)
        else:
            assert 1 - swl - sgcr > epsilon
            self.table["sgn"] = (self.table["sg"] - sgcr) / (1 - swl - sgcr)

        # Normalized oil saturation should be 0 at 1-sorg, and 1 at swl+sgcr
        self.table["son"] = (self.table["sl"] - sorg - swl) / (1 - sorg - swl)
        self.sgcomment = "-- swirr=%g, sgcr=%g, swl=%g, sorg=%g, krgendanchor=%s\n" % (
            self.swirr,
            self.sgcr,
            self.swl,
            self.sorg,
            self.krgendanchor,
        )
        self.krgcomment = ""
        self.krogcomment = ""
        self.pccomment = ""

        logger.debug(
            "Initialized GasOil with %s saturation points", str(len(self.table))
        )

    def resetsorg(self):
        """Recalculate sorg in case it has table data has been manipulated"""
        if "krog" in self.table.columns:
            self.sorg = (
                1 - self.swl - self.table[np.isclose(self.table.krog, 0.0)].min()["sg"]
            )
            self.sgcomment = (
                "-- swirr=%g, sgcr=%g, swl=%g, sorg=%g, krgendanchor=%s\n"
                % (self.swirr, self.sgcr, self.swl, self.sorg, self.krgendanchor)
            )

    def add_gasoil_fromtable(self, *args, **kwargs):
        """Deprecated. Use ``add_fromtable()``"""
        logger.warning("add_gasoil_fromtable() is deprecated, use add_fromtable()")
        self.add_fromtable(*args, **kwargs)

    def add_fromtable(
        self,
        dframe,
        sgcolname="Sg",
        krgcolname="krg",
        krogcolname="krog",
        pccolname="pcog",
        krgcomment="",
        krogcomment="",
        pccomment="",
    ):
        """Interpolate relpermdata from a dataframe.

        The saturation range with endpoints must be set up beforehand,
        and must be compatible with the tabular input. The tabular
        input will be interpolated to the initialized Sg-table.

        If you have krg and krog in different dataframes, call this
        function twice

        Calling function is responsible for checking if any data was
        actually added to the table.
        """
        # Avoid having to deal with multi-indices:
        if len(dframe.index.names) > 1:
            logger.warning(
                "add_fromtable() did a reset_index(), consider not supplying MultiIndex"
            )
            dframe = dframe.reset_index()

        if sgcolname not in dframe:
            logger.critical(
                "%s not found in dataframe, can't read table data", sgcolname
            )
            raise ValueError

        for col in [sgcolname, krgcolname, krogcolname, pccolname]:
            # Typecheck/convert all numerical columns:
            if col in dframe and not pd.api.types.is_numeric_dtype(dframe[col]):
                # Try to convert to numeric type
                try:
                    dframe[col] = dframe[col].astype(float)
                    logger.info("Converted column %s to numbers for fromtable()", col)
                except ValueError as e_msg:
                    logger.error(
                        "Failed to parse column %s as numbers for add_fromtable()", col
                    )
                    raise ValueError(e_msg)
                except TypeError as e_msg:
                    logger.error(
                        "Failed to parse column %s as numbers for add_fromtable()", col
                    )
                    raise TypeError(e_msg)

        if dframe[sgcolname].min() > 0.0:
            raise ValueError("sg must start at zero")
        swlfrominput = 1 - dframe[sgcolname].max()
        if abs(swlfrominput - self.swl) > epsilon:
            logger.warning(
                "swl=%f and 1-max(sg)=%f from incoming table does not seem compatible",
                self.swl,
                swlfrominput,
            )
            logger.warning("         Do not trust the result near the endpoint.")
        if krgcolname in dframe:
            if not (dframe[krgcolname].diff().dropna() > -epsilon).all():
                raise ValueError("Incoming krg not increasing")
            if dframe[krgcolname].max() > 1.0:
                raise ValueError("krg is above 1 in incoming table")
            if dframe[krgcolname].min() < 0.0:
                raise ValueError("krg is below 0 in incoming table")
            pchip = PchipInterpolator(
                dframe[sgcolname].astype(float), dframe[krgcolname].astype(float)
            )
            # Do not extrapolate this data. We will bfill and ffill afterwards
            self.table["krg"] = pchip(self.table.sg, extrapolate=False)
            self.table["krg"].fillna(method="ffill", inplace=True)
            self.table["krg"].fillna(method="bfill", inplace=True)
            self.table["krg"].clip(lower=0.0, upper=1.0, inplace=True)
            self.krgcomment = "-- krg from tabular input" + krgcomment + "\n"
        if krogcolname in dframe:
            if not (dframe[krogcolname].diff().dropna() < epsilon).all():
                raise ValueError("Incoming krogcolname not decreasing")
            if dframe[krogcolname].max() > 1.0:
                raise ValueError("krog is above 1 in incoming table")
            if dframe[krogcolname].min() < 0.0:
                raise ValueError("krog is below 0 in incoming table")
            pchip = PchipInterpolator(
                dframe[sgcolname].astype(float), dframe[krogcolname].astype(float)
            )
            self.table["krog"] = pchip(self.table.sg, extrapolate=False)
            self.table["krog"].fillna(method="ffill", inplace=True)
            self.table["krog"].fillna(method="bfill", inplace=True)
            self.table["krog"].clip(lower=0.0, upper=1.0, inplace=True)
            self.krogcomment = "-- krog from tabular input" + krogcomment + "\n"
        if pccolname in dframe:
            # Incoming dataframe must cover the range:
            if dframe[sgcolname].min() > self.table["sg"].min():
                raise ValueError("Too large sgcr for pcog interpolation")
            if dframe[sgcolname].max() < self.table["sg"].max():
                raise ValueError("Too large swl for pcog interpolation")
            if np.isinf(dframe[pccolname]).any():
                logger.warning(
                    (
                        "Infinity pc values detected. Will be dropped, "
                        "risk of extrapolation"
                    )
                )
            dframe = dframe.replace([np.inf, -np.inf], np.nan)
            dframe.dropna(subset=[pccolname], how="all", inplace=True)
            # If nonzero, then it must be increasing:
            if dframe[pccolname].abs().sum() > 0:
                if not (dframe[pccolname].diff().dropna() > 0.0).all():
                    raise ValueError("Incoming pc not increasing")
            pchip = PchipInterpolator(
                dframe[sgcolname].astype(float), dframe[pccolname].astype(float)
            )
            self.table["pc"] = pchip(self.table.sg, extrapolate=False)
            if np.isnan(self.table["pc"]).any() or np.isinf(self.table["pc"]).any():
                raise ValueError("inf/nan in interpolated data, check input")
            self.pccomment = "-- pc from tabular input" + pccomment + "\n"

    def set_endpoints_linearpart_krg(self, krgend, krgmax=None):
        """Set linear parts of krg outside endpoints.

        Curve is set to zero in [0, sgcr].

        Given the default krgendanchor==sorg, the
        curve will be linear in [1 - swl - sorg, 1 - swl]
        (from krgend to krgmax). If not anchored to sorg, there
        is no linear part near sg=1-swl.

        If krgendanchor is set to `sorg` (default), then the normalized
        gas saturation `sgn` (which is what is raised to the power of `ng`)
        is 1 at `sg = 1 - swl - sorg`. If not, it is 1 at `sg = 1 - swl`.

        krgmax is only relevant if krgendanchor is 'sorg'

        This function is used by add_corey/LET_gas(), and perhaps by other
        utility functions. It should not be necessary for end-users.

        Args:
            krgend (float): krg at sg = 1 - swl - sorg.
            krgmax (float): krg at Sg = 1 - swl. Default 1.

        """
        self.table.loc[self.table.sg <= self.sgcr, "krg"] = 0

        if self.krgendanchor == "sorg":
            # Linear curve between krgendcanchor and 1-swl if krgend
            # is anchored to sorg
            if not krgmax:
                krgmax = 1
            tmp = pd.DataFrame(self.table[["sg"]])
            tmp["sgendnorm"] = (tmp["sg"] - (1 - (self.sorg + self.swl))) / (self.sorg)
            tmp["krg"] = (
                tmp["sgendnorm"] * krgmax + (1 - tmp["sgendnorm"]) * krgend
            ).clip(lower=0.0, upper=1.0)
            self.table.loc[
                self.table.sg >= (1 - (self.sorg + self.swl + epsilon)), "krg"
            ] = tmp.loc[tmp.sg >= (1 - (self.sorg + self.swl + epsilon)), "krg"]
        else:
            self.table.loc[self.table.sg > (1 - (self.swl + epsilon)), "krg"] = krgend
            if krgmax and krgmax < 1.0 and self.sorg > 0:
                # Only warn if something else than default is in use
                logger.warning("krgmax ignored when not anchoring to sorg")

    def set_endpoints_linearpart_krog(self, kroend, kromax=None):
        """Set linear parts of krog outside endpoints.

        Zero for sg above 1 - sorg - swl.

        This function is used by add_corey/LET_oil(), and perhaps by other
        utility functions. It should not be necessary for end-users.

        Args:
            kroend (float): krog at sg=0
        """
        if kromax is not None:
            logger.error("kromax is DEPRECATED, ignored")

        # Special handling of the part close to sg=1, set to zero.
        self.table.loc[
            self.table["sg"] > 1 - self.sorg - self.swl - epsilon, "krog"
        ] = 0

        # Floating point issues can cause a slight overshoot at sg=0:
        self.table.loc[self.table["krog"] > kroend, "krog"] = kroend

    def add_corey_gas(self, ng=2, krgend=1, krgmax=None):
        """ Add krg data through the Corey parametrization

        A column called 'krg' will be added. If it exists, it will
        be replaced.

        If krgendanchor is sorg, the Corey curve ends at krgend at
        sg = 1 - swl - sorg, and then linear up to krgmax at
        sg = 1 - swl. If not, it ends at krgend at sg = 1 - swl.

        krgmax is only relevant if krgendanchor is 'sorg'
        """
        assert epsilon < ng < MAX_EXPONENT
        assert 0 < krgend <= 1.0
        if krgmax is not None:
            assert 0 < krgend <= krgmax <= 1.0
        self.table["krg"] = krgend * self.table.sgn ** ng

        self.set_endpoints_linearpart_krg(krgend, krgmax)

        if not krgmax:
            krgmax = 1
        self.krgcomment = "-- Corey krg, ng=%g, krgend=%g, krgmax=%g\n" % (
            ng,
            krgend,
            krgmax,
        )

    def add_corey_oil(self, nog=2, kroend=1, kromax=None):
        """
        Add kro data through the Corey parametrization

        A column named 'kro' will be added to the internal DataFrame,
        replaced if it exists.

        All values above 1 - sorg - swl are set to zero.

        Arguments:
            nog (float): Corey exponent for oil
            kroend (float): Value for krog at normalized oil saturation 1

        Returns:
            None (modifies internal class state)
        """
        assert epsilon < nog < MAX_EXPONENT
        assert 0 < kroend <= 1.0

        if kromax is not None:
            logger.error("kromax is DEPRECATED, ignored")

        self.table["krog"] = kroend * self.table.son ** nog

        self.set_endpoints_linearpart_krog(kroend)

        self.krogcomment = "-- Corey krog, nog=%g, kroend=%g\n" % (nog, kroend,)

    def add_LET_gas(self, l=2, e=2, t=2, krgend=1, krgmax=None):
        """
        Add gas relative permability data through the LET parametrization

        A column called 'krg' will be added, replaced if it does not exist

        If krgendanchor is sorg, the LET curve ends at krgend at
        sg = 1 - swl - sorg, and then linear up to krgmax at
        sg = 1 - swl. If not, it ends at krgend at sg = 1 - swl.

        Arguments:
            l (float): L parameter in LET
            e (float): E parameter in LET
            t (float): T parameter in LET
            krgend (float): Value of krg at normalized gas saturation 1
            krgmax (float): Value of krg at gas saturation 1

        Returns:
            None (modifies internal state)
        """
        # Similar code in wateroil.add_LET_water, but readability
        # is better by having them separate
        # pylint: disable=duplicate-code
        assert epsilon < l < MAX_EXPONENT
        assert epsilon < e < MAX_EXPONENT
        assert epsilon < t < MAX_EXPONENT
        if krgmax:
            assert 0 < krgend <= krgmax <= 1.0
        else:
            assert 0 < krgend <= 1.0

        self.table["krg"] = (
            krgend
            * self.table.sgn ** l
            / ((self.table.sgn ** l) + e * (1 - self.table.sgn) ** t)
        )
        # This equation is undefined for t a float and sgn=1, set explicitly:
        self.table.loc[np.isclose(self.table["sgn"], 1.0), "krg"] = krgend

        self.set_endpoints_linearpart_krg(krgend, krgmax)

        if not krgmax:
            krgmax = 1
        self.krgcomment = "-- LET krg, l=%g, e=%g, t=%g, krgend=%g, krgmax=%g\n" % (
            l,
            e,
            t,
            krgend,
            krgmax,
        )

    def add_LET_oil(self, l=2, e=2, t=2, kroend=1, kromax=None):
        """Add oil (vs gas) relative permeability data through the Corey
        parametrization.

        A column named 'krog' will be added, replaced if it exists.

        All values where sg > 1 - sorg - swl are set to zero.

        Arguments:
            l (float): L parameter
            e (float): E parameter
            t (float): T parameter
            kroend (float): The value at gas saturation sgcr
        """
        assert epsilon < l < MAX_EXPONENT
        assert epsilon < e < MAX_EXPONENT
        assert epsilon < t < MAX_EXPONENT
        assert 0 < kroend <= 1.0

        if kromax is not None:
            logger.error("kromax is DEPRECATED, ignored")

        # LET shape for the interval [sgcr, 1 - swl - sorg]
        self.table["krog"] = (
            kroend
            * self.table["son"] ** l
            / ((self.table["son"] ** l) + e * (1 - self.table["son"]) ** t)
        )
        # This equation is undefined for t a float and son=1, set explicitly:
        self.table.loc[np.isclose(self.table["son"], 1.0), "krog"] = kroend

        self.set_endpoints_linearpart_krog(kroend)

        self.krogcomment = "-- LET krog, l=%g, e=%g, t=%g, kroend=%g\n" % (
            l,
            e,
            t,
            kroend,
        )

    def estimate_sorg(self):
        """Estimate sorg of the current krg or krog data.

        sorg is estimated by searching for a linear part in krg downwards
        from sg=1-swl. In practice it is impossible to infer sorg = 0,
        since we are limited by h, and the last segment from sg=1-swl-h
        to sg=1-swl can always be assumed linear.

        If krgend is anchored to sorg, krg data is used to infer sorg. If not,
        krg cannot be used for this, and krog is used. sorg might be overestimated
        when krog is used if it very close to zero before reaching sorw.

        If the curve is linear everywhere, sorg will be returned as sgcr + h

        Args:
            None
        Returns:
            float: The estimated sorg.
        """
        if self.krgendanchor == "sorg":
            assert "krg" in self.table
            assert self.table["krg"].sum() > 0
            return self.table["sg"].max() - utils.estimate_diffjumppoint(
                self.table, xcol="sg", ycol="krg", side="right"
            )
        assert "krog" in self.table
        assert self.table["krog"].sum() > 0
        return self.table["sg"].max() - utils.estimate_diffjumppoint(
            self.table, xcol="sg", ycol="krog", side="right"
        )

    def estimate_sgcr(self, curve="krog"):
        """Estimate sgcr of the current krog data.

        sgcr is estimated by searching for a linear part in krog upwards from sg=0.
        In practice it is impossible to infer sgcr = 0, since we are limited by
        h, and we always have to assume that the first segment is linear.

        If the curve is linear everywhere, sgcr will be returned as the right endpoint.

        Args:
            curve (str): Column name to use for search for linearity. Default is krog,
                if all of that is linear, you may try krg instead.
        Returns:
            float: The estimated sgcr.
        """
        assert curve in self.table
        assert self.table[curve].sum() > 0
        return utils.estimate_diffjumppoint(
            self.table, xcol="sg", ycol=curve, side="left"
        )

    def crosspoint(self):
        """Locate and return the saturation point where krg = krog

        Accuracy of this crosspoint depends on the resolution chosen
        when initializing the saturation range (it uses linear
        interpolation to solve for the zero)

        Returns:
            float: the gas saturation where krg == krog, for relperm
                linearly interpolated in gas saturation.
        """
        return utils.crosspoint(self.table, "sg", "krg", "krog")

    def selfcheck(self, mode="SGOF"):
        """Check validities of the data in the table.

        This is to catch errors that are either physically wrong
        or at least causes Eclipse 100 to stop.

        Returns True if no errors are found, False if at least one
        is found.

        If you call SGOF/SLGOF, this function must not return False.

        Args:
            mode (str): If mode is "SGFN", krog is not required.
        """
        error = False
        if "krg" not in self.table:
            logger.error("krg data missing")
            error = True
        if not (self.table["sg"].diff().dropna() > -epsilon).all():
            logger.error("sg data not strictly increasing")
            error = True
        if (
            "krg" in self.table
            and not (self.table["krg"].diff().dropna() >= -epsilon).all()
        ):
            logger.error("krg data not monotonically decreasing")
            error = True

        if mode != "SGFN":
            if "krog" not in self.table:
                logger.error("krog data missing")
                error = True
            if (
                "krog" in self.table
                and not (self.table["krog"].diff().dropna() <= epsilon).all()
            ):
                logger.error("krog data not monotonically increasing")
                error = True
        if "krg" in self.table and not np.isclose(min(self.table["krg"]), 0.0):
            logger.error("krg must start at zero")
            error = True
        if "pc" in self.table and self.table["pc"][0] > 0:
            if not (self.table["pc"].diff().dropna() < epsilon).all():
                logger.error("pc data for gas-oil not strictly deceasing")
                error = True
        if "pc" in self.table and np.isinf(self.table["pc"].max()):
            logger.error("pc goes to infinity for gas-oil. ")
            error = True
        for col in list(set(["sg", "krg", "krog"]) & set(self.table.columns)):
            if not (
                (min(self.table[col]) >= -epsilon)
                and (max(self.table[col]) <= 1 + epsilon)
            ):
                logger.error("%s data should be contained in [0,1]", col)
                error = True
        if error:
            return False
        logger.debug("GasOil object is checked to be valid")
        return True

    def SGOF(self, header=True, dataincommentrow=True):
        """
        Produce SGOF input for Eclipse reservoir simulator.

        The columns sg, krg, krog and pc are outputted and
        formatted accordingly.

        Meta-information for the tabulated data are printed
        as Eclipse comments.

        Args:
            header (bool): Whether the SGOF string should be emitted.
                If you have multiple satnums, you should have True only
                for the first (or False for all, and emit the SGOF yourself).
                Defaults to True.
            dataincommentrow (bool): Whether metadata should be printed,
                defaults to True.
        """
        if not self.fast and not self.selfcheck():
            # selfcheck() will log error/warning messages
            return ""
        string = ""
        if "pc" not in self.table:
            self.table["pc"] = 0
            self.pccomment = "-- Zero capillary pressure\n"
        if header:
            string += "SGOF\n"
        string += utils.comment_formatter(self.tag)
        string += "-- pyscal: " + str(pyscal.__version__) + "\n"
        if dataincommentrow:
            string += self.sgcomment
            string += self.krgcomment
            string += self.krogcomment
            if not self.fast:
                string += "-- krg = krog @ sg=%1.5f\n" % self.crosspoint()
            string += self.pccomment
        width = 10
        string += (
            "-- "
            + "SG".ljust(width - 3)
            + "KRG".ljust(width)
            + "KROG".ljust(width)
            + "PC".ljust(width)
            + "\n"
        )
        string += utils.df2str(
            self.table[["sg", "krg", "krog", "pc"]],
            monotone_column="pc",
            monotone_direction="inc",
        )
        string += "/\n"
        return string

    def slgof_df(self):
        """Slice out an SLGOF table.

        This is a used by the SLGOF() function, it is
        extracted as a single function to facilitate testing."""
        if "pc" not in self.table.columns:
            # Only happens when the SLGOF function is skipped (test code)
            self.table["pc"] = 0
        slgof = (
            self.table[
                self.table["sg"] <= 1 - self.sorg - self.swl + 1.0 / float(SWINTEGERS)
            ]
            .sort_values("sl")[["sl", "krg", "krog", "pc"]]
            .reset_index(drop=True)
        )
        # It is a strict requirement that the first sl value should be swl + sorg,
        # so we modify it if it close. If it is not close, we do not dare to fix
        # it, to ensure we don't cover bugs.
        slgof_sl_mismatch = abs(slgof["sl"].values[0] - (self.sorg + self.swl))
        if slgof_sl_mismatch > epsilon:
            if slgof_sl_mismatch < 2 * 1.0 / float(SWINTEGERS):
                # Repair the table in-place:
                slgof.loc[0, "sl"] = self.sorg + self.swl
                # After modification, we can get duplicate sl values,
                # so drop duplicates:
                slgof["slint"] = list(
                    map(int, list(map(round, slgof["sl"] * SWINTEGERS)))
                )
                slgof.drop_duplicates("slint", inplace=True)
                # Delete the temporary column:
                slgof.drop(labels="slint", axis="columns", inplace=True)
            else:
                # Give up repairing the table:
                logger.critical(
                    "SLGOF does not start at the correct value. Please report as bug."
                )
                logger.error("slgof_sl_mismatch: %f", slgof_sl_mismatch)
                logger.error(str(slgof.head()))
        return slgof

    def SLGOF(self, header=True, dataincommentrow=True):
        """Produce SLGOF input for Eclipse reservoir simulator.

        The columns sl (liquid saturation), krg, krog and pc are
        outputted and formatted accordingly.

        Meta-information for the tabulated data are printed
        as Eclipse comments.

        Args:
            header: boolean for whether the SLGOF string should be emitted.
                If you have multiple satnums, you should have True only
                for the first (or False for all, and emit the SGOF yourself).
                Defaults to True.
            dataincommentrow: boolean for wheter metadata should be printed,
                defaults to True.
        """
        if not self.selfcheck():
            # Selfcheck will issue error messages.
            return ""
        string = ""
        if "pc" not in self.table:
            self.table["pc"] = 0
            self.pccomment = "-- Zero capillary pressure\n"
        if header:
            string += "SLGOF\n"
        string += utils.comment_formatter(self.tag)
        string += "-- pyscal: " + str(pyscal.__version__) + "\n"
        if dataincommentrow:
            string += self.sgcomment
            string += self.krgcomment
            string += self.krogcomment
            string += "-- krg = krog @ sg=%1.5f\n" % self.crosspoint()
            string += self.pccomment
        width = 10
        string += (
            "-- "
            + "SL".ljust(width - 3)
            + "KRG".ljust(width)
            + "KROG".ljust(width)
            + "PC".ljust(width)
            + "\n"
        )
        string += utils.df2str(
            self.slgof_df(), monotone_column="pc", monotone_direction="dec"
        )
        string += "/\n"
        return string

    def SGFN(
        self, header=True, dataincommentrow=True, sgcomment=None, crosspointcomment=None
    ):
        """
        Produce SGFN input for Eclipse reservoir simulator.

        The columns sg, krg, and pc are outputted and
        formatted accordingly.

        Meta-information for the tabulated data are printed
        as Eclipse comments.

        Args:
            header: boolean for whether the SGFN string should be emitted.
                If you have multiple satnums, you should have True only
                for the first (or False for all, and emit the SGFN yourself).
                Defaults to True.
            dataincommentrow: boolean for wheter metadata should be printed,
                defaults to True.
            sgcomment (str): Provide the string to include in the comment
                section for describing the saturation endpoints. Used
                by GasWater.
            crosspointcomment (str): String to be used for crosspoint comment
                string, overrides what this object can provide. Used by GasWater.
                If None, it will be computed, use empty string to avoid.
        """
        string = ""
        if "pc" not in self.table.columns:
            self.table["pc"] = 0
            self.pccomment = "-- Zero capillary pressure\n"
        if header:
            string += "SGFN\n"
        string += utils.comment_formatter(self.tag)
        string += "-- pyscal: " + str(pyscal.__version__) + "\n"
        if dataincommentrow:
            if sgcomment is not None:
                string += sgcomment
            else:
                string += self.sgcomment
            string += self.krgcomment
            if crosspointcomment is None:
                if "krog" in self.table.columns:
                    string += "-- krg = krog @ sg=%1.5f\n" % self.crosspoint()
            else:
                string += crosspointcomment
            string += self.pccomment
        width = 10
        string += (
            "-- "
            + "SG".ljust(width - 3)
            + "KRG".ljust(width)
            + "PC".ljust(width)
            + "\n"
        )
        string += utils.df2str(
            self.table[["sg", "krg", "pc"]],
            monotone_column="pc",
            monotone_direction="inc",
        )
        string += "/\n"
        return string

    def GOTABLE(self, header=True, dataincommentrow=True):
        """
        Produce GOTABLE input for the Nexus reservoir simulator.

        The columns sg, krg, krog and pc are outputted and
        formatted accordingly.

        Meta-information for the tabulated data are printed
        as Eclipse comments.

        Args:
            header: boolean for whether the SGOF string should be emitted.
                If you have multiple satnums, you should have True only
                for the first (or False for all, and emit the SGOF yourself).
                Defaults to True.
            dataincommentrow: boolean for wheter metadata should be printed,
                defaults to True.
        """
        string = ""
        if "pc" not in self.table.columns:
            self.table["pc"] = 0
            self.pccomment = "-- Zero capillary pressure\n"
        if header:
            string += "GOTABLE\n"
            string += "SG KRG KROG PC\n"
        string += "! pyscal: " + str(pyscal.__version__) + "\n"
        if dataincommentrow:
            string += self.sgcomment.replace("--", "!")
            string += self.krgcomment.replace("--", "!")
            string += self.krogcomment.replace("--", "!")
            string += "! krg = krog @ sw=%1.5f\n" % self.crosspoint()
            string += self.pccomment.replace("--", "!")
        width = 10
        string += (
            "! "
            + "SG".ljust(width - 2)
            + "KRG".ljust(width)
            + "KROG".ljust(width)
            + "PC".ljust(width)
            + "\n"
        )
        string += utils.df2str(
            self.table[["sg", "krg", "krog", "pc"]],
            monotone_column="pc",
            monotone_direction="inc",
        )
        return string

    def plotkrgkrog(
        self,
        mpl_ax=None,
        color="blue",
        alpha=1,
        linewidth=1,
        linestyle="-",
        marker=None,
        label=None,
        logyscale=False,
    ):
        """Plot krg and krog

        If mpl_ax is not None, it will be used as a
        matplotlib axis to plot on, if None, a fresh plot
        will be made.
        """
        import matplotlib.pyplot as plt
        import matplotlib

        if mpl_ax is None:
            matplotlib.style.use("ggplot")
            _, useax = matplotlib.pyplot.subplots()
        else:
            useax = mpl_ax
        if logyscale:
            useax.set_yscale("log")
            useax.set_ylim([1e-8, 1])
        self.table.plot(
            ax=useax,
            x="sg",
            y="krg",
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
            x="sg",
            y="krog",
            c=color,
            alpha=alpha,
            legend=None,
            label=None,
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
        )
        if mpl_ax is None:
            plt.show()
