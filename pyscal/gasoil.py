# -*- coding: utf-8 -*-
from __future__ import division, absolute_import
from __future__ import print_function

import logging
import numpy as np
import pandas as pd

from pyscal.constants import EPSILON as epsilon
from pyscal.constants import SWINTEGERS
from pyscal.constants import MAX_EXPONENT


class GasOil(object):
    """Object to represent two-phase properties for gas and oil.

    Parametrizations available for relative permeability:

     * Corey
     * LET

    or data can alternatively be read in from tabulated data
    (as Pandas DataFrame).

    No support (yet) to add capillary pressure.

    krgend can be anchored both to `1-swl-sorg` and to `1-swl`. Default is
    to anchor to `1-swl-sorg`. If the krgendanchor argument is something
    else than the string `sorg`, it will be anchored to `1-swl`.

    Code duplication warning: Code is analogous to WaterOil, but with
    some subtle details sufficiently different for now warranting its
    own code (no easy inheritance)

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
        krgendanchor (str): Set to `sorg` or something else, where to
            anchor `krgend`. If `sorg`, then the normalized gas
            saturation will be equal to 1 at `1 - swl - sgcr - sorg`,
            if not, it will be 1 at `1 - swl - sgcr`. If `sorg` is zero
            it does not matter.
        h (float): Saturation step-length in the outputted table.
        tag (str): Optional string identifier, only used in comments.
    """

    def __init__(
        self, swirr=0, sgcr=0.0, h=0.01, swl=0.0, sorg=0.0, tag="", krgendanchor="sorg"
    ):
        assert epsilon < h <= 1
        assert -epsilon < swirr < 1.0 + epsilon
        assert -epsilon < sgcr < 1
        assert -epsilon < swl < 1
        assert -epsilon < sorg < 1
        if not isinstance(tag, str):
            tag = ""
        assert isinstance(krgendanchor, str)

        self.h = h
        swl = max(swl, swirr)  # Can't allow swl < swirr, should we warn user?
        self.swl = swl
        self.swirr = swirr
        if not np.isclose(sorg, 0.0) and sorg < 1.0 / SWINTEGERS:
            # Too small but nonzero sorg gives too many numerical issues.
            logging.warning("sorg was close to zero, set to zero")
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
            logging.warning("Unknown krgendanchor %s, ignored", str(krgendanchor))
            self.krgendanchor = ""
        if np.isclose(sorg, 0.0) and self.krgendanchor == "sorg":
            # This is too much info..
            self.krgendanchor = ""  # This is critical to avoid bugs due to numerics.

        sg = (
            [0]
            + [sgcr]
            + list(np.arange(sgcr + h, 1 - swl, h))
            + [1 - sorg - swl]
            + [1 - swl]
        )
        self.table = pd.DataFrame(sg, columns=["sg"])
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
        # Ensure the value closest to 1-swl is actually 1-swl:
        swl_right_index = (
            (self.table["sg"] - (1 - self.swl)).abs().sort_values().index[0]
        )
        self.table.loc[swl_right_index, "sg"] = 1 - self.swl

        self.table.sort_values(by="sg", inplace=True)
        self.table.reset_index(inplace=True)
        self.table = self.table[["sg"]]
        self.table["sl"] = 1 - self.table["sg"]
        if krgendanchor == "sorg":
            # Normalized sg (sgn) is 0 at sgcr, and 1 at 1-swl-sorg
            self.table["sgn"] = (self.table["sg"] - sgcr) / (1 - swl - sgcr - sorg)
        else:
            self.table["sgn"] = (self.table["sg"] - sgcr) / (1 - swl - sgcr)

        # Normalized oil saturation should be 0 at 1-sorg, and 1 at swl+sgcr
        self.table["son"] = (self.table["sl"] - sorg - swl) / (1 - sorg - swl - sgcr)
        self.sgcomment = (
            '-- swirr=%g, sgcr=%g, swl=%g, sorg=%g, krgendanchor="%s"\n'
            % (self.swirr, self.sgcr, self.swl, self.sorg, self.krgendanchor)
        )
        self.krgcomment = ""
        self.krogcomment = ""
        self.pccomment = ""

        logging.info(
            "Initialized GasOil with %s saturation points", str(len(self.table))
        )

    def resetsorg(self):
        """Recalculate sorg in case it has table data has been manipulated"""
        if "krog" in self.table.columns:
            self.sorg = (
                1 - self.swl - self.table[np.isclose(self.table.krog, 0.0)].min()["sg"]
            )
            self.sgcomment = (
                '-- swirr=%g, sgcr=%g, swl=%g, sorg=%g\n, krgendanchor="%s"'
                % (self.swirr, self.sgcr, self.swl, self.sorg, self.krgendanchor)
            )

    def add_gasoil_fromtable(self, *args, **kwargs):
        logging.warning("add_gasoil_fromtable() is deprecated, use add_fromtable()")
        self.add_fromtable(*args, **kwargs)

    def add_fromtable(
        self,
        df,
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
        IMPORTANT: Set sgcr and swl to sensible values.

        If you have krg and krog in different dataframes, call this
        function twice

        Calling function is responsible for checking if any data was
        actually added to the table.

        The dataframe input can be constructed using e.g. swof2csv functionality

        """
        from scipy.interpolate import PchipInterpolator

        if sgcolname not in df:
            logging.critical(
                sgcolname + " not found in dataframe, " + "can't read table data"
            )
            raise ValueError
        if df[sgcolname].min() > 0.0:
            raise ValueError("sg must start at zero")
        swlfrominput = 1 - df[sgcolname].max()
        if abs(swlfrominput - self.swl) < epsilon:
            logging.warning(
                "swl and 1-max(sg) from incoming table does not seem compatible"
            )
            logging.warning("         Do not trust the result near the endpoint.")
        if krgcolname in df:
            if not (df[krgcolname].diff().dropna() > -epsilon).all():
                raise ValueError("Incoming krg not increasing")
            pchip = PchipInterpolator(
                df[sgcolname].astype(float), df[krgcolname].astype(float)
            )
            # Do not extrapolate this data. We will bfill and ffill afterwards
            self.table["krg"] = pchip(self.table.sg, extrapolate=False)
            self.table["krg"].fillna(method="ffill", inplace=True)
            self.table["krg"].fillna(method="bfill", inplace=True)
            self.krgcomment = "-- krg from tabular input" + krgcomment + "\n"
        if krogcolname in df:
            if not (df[krogcolname].diff().dropna() < epsilon).all():
                raise ValueError("Incoming krogcolname not decreasing")
            pchip = PchipInterpolator(
                df[sgcolname].astype(float), df[krogcolname].astype(float)
            )
            self.table["krog"] = pchip(self.table.sg, extrapolate=False)
            self.table["krog"].fillna(method="ffill", inplace=True)
            self.table["krog"].fillna(method="bfill", inplace=True)
            self.krogcomment = "-- krog from tabular input" + krogcomment + "\n"
        if pccolname in df:
            # Incoming dataframe must cover the range:
            if df[sgcolname].min() > self.table["sg"].min():
                raise ValueError("Too large sgcr for pcog interpolation")
            if df[sgcolname].max() < self.table["sg"].max():
                raise ValueError("Too large swl for pcog interpolation")
            if np.isinf(df[pccolname]).any():
                logging.warning(
                    (
                        "Infinity pc values detected. Will be dropped, "
                        "risk of extrapolation"
                    )
                )
            df = df.replace([np.inf, -np.inf], np.nan)
            df.dropna(subset=[pccolname], how="all", inplace=True)
            # If nonzero, then it must be decreasing:
            if df[pccolname].abs().sum() > 0:
                if not (df[pccolname].diff().dropna() < 0.0).all():
                    raise ValueError("Incoming pc not decreasing")
            pchip = PchipInterpolator(
                df[sgcolname].astype(float), df[pccolname].astype(float)
            )
            self.table["pc"] = pchip(self.table.sg, extrapolate=False)
            if np.isnan(self.table["pc"]).any() or np.isinf(self.table["pc"]).any():
                raise ValueError("inf/nan in interpolated data, check input")
            self.pccomment = "-- pc from tabular input" + pccomment + "\n"

    def _handle_endpoints_linearpart_gas(self, krgend, krgmax=None):
        """Internal utility function to handle krg
        around endpoints.
        """
        self.table.loc[self.table.sg <= self.sgcr, "krg"] = 0

        if self.krgendanchor == "sorg":
            # Linear curve between krgendcanchor and 1-swl if krgend
            # is anchored to sorg
            if not krgmax:
                krgmax = 1
            tmp = pd.DataFrame(self.table[["sg"]])
            tmp["sgendnorm"] = (tmp["sg"] - (1 - (self.sorg + self.swl))) / (self.sorg)
            tmp["krg"] = tmp["sgendnorm"] * krgmax + (1 - tmp["sgendnorm"]) * krgend
            self.table.loc[
                self.table.sg >= (1 - (self.sorg + self.swl + epsilon)), "krg"
            ] = tmp.loc[tmp.sg >= (1 - (self.sorg + self.swl + epsilon)), "krg"]
        else:
            self.table.loc[self.table.sg > (1 - (self.swl + epsilon)), "krg"] = krgend
            if krgmax:
                logging.warning("krgmax ignored when not anchoring to sorg")

    def _handle_endpoints_linearpart_oil(self, kroend, kromax):
        """Internal utility function to handle kro
        around endpoints.
        """

        # Special handling of the part close to sg=1, set to zero.
        self.table.loc[
            self.table["sg"] > 1 - self.sorg - self.swl - epsilon, "krog"
        ] = 0

        # Set kromax at sg=0, but only if sgcr is sufficiently larger than swl.
        if self.sgcr > self.swl + 1.0 / SWINTEGERS:
            if not kromax:
                kromax = 1
            self.table.loc[self.table["sg"] < epsilon, "krog"] = kromax
        else:
            if kromax:
                logging.warning("kromax ignored when sgcr is close to swl")
            self.table.loc[self.table["sg"] < epsilon, "krog"] = kroend

    def add_corey_gas(self, ng=2, krgend=1, krgmax=None):
        """ Add krg data through the Corey parametrization

        A column called 'krg' will be added. If it exists, it will
        be replaced.

        If krgendanchor is set to `sorg` (default), then the normalized
        gas saturation `sgn` (which is what is raised to the power of `ng`)
        is 1 at `1 - swl - sgcr - sorg`. If not, it is 1 at `1 - swl - sgcr`.

        krgmax is only relevant if krgendanchor is 'sorg'
        """
        assert epsilon < ng < MAX_EXPONENT
        assert 0 < krgend <= 1.0
        if krgmax is not None:
            assert 0 < krgend <= krgmax <= 1.0
        self.table["krg"] = krgend * self.table.sgn ** ng

        self._handle_endpoints_linearpart_gas(krgend, krgmax)

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

        kromax is ignored if sgcr is close to zero

        Arguments:
            nog (float): Corey exponent for oil
            kroend (float): Value for krog at normalized oil saturation 1
            kromax (float): Value for krog at gas saturation 0.

        Returns:
            None (modifies internal class state)
        """
        assert epsilon < nog < MAX_EXPONENT
        if kromax:
            assert 0 < kroend <= kromax <= 1.0
        else:
            assert 0 < kroend <= 1.0

        self.table["krog"] = kroend * self.table.son ** nog

        self._handle_endpoints_linearpart_oil(kroend, kromax)

        if not kromax:
            kromax = 1
        self.krogcomment = "-- Corey krog, nog=%g, kroend=%g, kromax=%g\n" % (
            nog,
            kroend,
            kromax,
        )

    def add_LET_gas(self, l=2, e=2, t=2, krgend=1, krgmax=None):
        """
        Add gas relative permability data through the LET parametrization

        A column called 'krg' will be added, replaced if it does not exist

        If krgendanchor is set to `sorg` (default), then the normalized
        gas saturation `sgn` (which is what is raised to the power of `ng`)
        is 1 at `1 - swl - sgcr - sorg`. If not, it is 1 at `1 - swl - sgcr`

        Arguments:
            l (float): L parameter in LET
            e (float): E parameter in LET
            t (float): T parameter in LET
            krgend (float): Value of krg at normalized gas saturation 1
            krgmax (float): Value of krg at gas saturation 1

        Returns:
            None (modifies internal state)
        """
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

        self._handle_endpoints_linearpart_gas(krgend, krgmax)

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

        kromax is ignored if sgcr is close to zero

        Arguments:
            l (float): L parameter
            e (float): E parameter
            t (float): T parameter
            kroend (float): The value at gas saturation sgcr
            kromax (float): The value at gas saturation equal to 0.
        """
        assert epsilon < l < MAX_EXPONENT
        assert epsilon < e < MAX_EXPONENT
        assert epsilon < t < MAX_EXPONENT
        if kromax:
            assert 0 < kroend <= kromax <= 1.0
        else:
            assert 0 < kroend <= 1.0

        # LET shape for the interval [sgcr, 1 - swl - sorg]
        self.table["krog"] = (
            kroend
            * self.table["son"] ** l
            / ((self.table["son"] ** l) + e * (1 - self.table["son"]) ** t)
        )

        self._handle_endpoints_linearpart_oil(kroend, kromax)

        if not kromax:
            kromax = 1
        self.krogcomment = "-- LET krog, l=%g, e=%g, t=%g, kroend=%g, kromax=%g\n" % (
            l,
            e,
            t,
            kroend,
            kromax,
        )

    def selfcheck(self):
        """Check validities of the data in the table.

        This is to catch errors that are either physically wrong
        or at least causes Eclipse 100 to stop.

        Returns True if no errors are found, False if at least one
        is found.

        If you call SGOF/SLGOF, this function must not return False.
        """
        error = False
        if not (self.table.sg.diff().dropna() > -epsilon).all():
            logging.error("sg data not strictly increasing")
            error = True
        if not (self.table.krg.diff().dropna() >= -epsilon).all():
            logging.error("krg data not monotonically decreasing")
            error = True

        if (
            "krog" in self.table.columns
            and not (self.table.krog.diff().dropna() <= epsilon).all()
        ):
            logging.error("krog data not monotonically increasing")
            error = True
        if not np.isclose(min(self.table.krg), 0.0):
            logging.error("krg must start at zero")
            error = True
        if "pc" in self.table.columns and self.table.pc[0] > 0:
            if not (self.table.pc.diff().dropna() < epsilon).all():
                logging.error("pc data for gas-oil not strictly deceasing")
                error = True
        if "pc" in self.table.columns and np.isinf(self.table.pc.max()):
            logging.error("pc goes to infinity for gas-oil. ")
            error = True
        for col in list(set(["sg", "krg", "krog"]) & set(self.table.columns)):
            if not (
                (min(self.table[col]) >= -epsilon)
                and (max(self.table[col]) <= 1 + epsilon)
            ):
                logging.error("%s data should be contained in [0,1]", col)
                error = True
        if error:
            return False
        else:
            logging.info("GasOil object is checked to be valid")
            return True

    def SGOF(self, header=True, dataincommentrow=True):
        """
        Produce SGOF input for Eclipse reservoir simulator.

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
        if not self.selfcheck():
            return
        string = ""
        if "pc" not in self.table.columns:
            self.table["pc"] = 0
            self.pccomment = "-- Zero capillary pressure\n"
        if header:
            string += "SGOF\n"
        string += "-- " + self.tag + "\n"
        string += "-- Sg Krg Krog Pc\n"
        if dataincommentrow:
            string += self.sgcomment
            string += self.krgcomment
            string += self.krogcomment
            string += "-- krg = krog @ sw=%1.5f\n" % self.crosspoint()
            string += self.pccomment
        string += self.table[["sg", "krg", "krog", "pc"]].to_csv(
            sep=" ", float_format="%1.7f", header=None, index=False
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
            else:
                # Give up repairing the table:
                logging.critical(
                    "SLGOF does not start at the correct value. Please report as bug."
                )
                logging.error("slgof_sl_mismatch: %f", slgof_sl_mismatch)
                logging.error(str(slgof.head()))
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
            return
        string = ""
        if "pc" not in self.table.columns:
            self.table["pc"] = 0
            self.pccomment = "-- Zero capillary pressure\n"
        if header:
            string += "SLGOF\n"
        string += "-- " + self.tag + "\n"
        string += "-- Sl Krg Krog Pc\n"
        if dataincommentrow:
            string += self.sgcomment
            string += self.krgcomment
            string += self.krogcomment
            string += "-- krg = krog @ sw=%1.5f\n" % self.crosspoint()
            string += self.pccomment
        string += self.slgof_df().to_csv(
            sep=" ", float_format="%1.7f", header=None, index=False
        )
        string += "/\n"
        return string

    def SGFN(self, header=True, dataincommentrow=True):
        """
        Produce SGFN input for Eclipse reservoir simulator.

        The columns sg, krg, and pc are outputted and
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
            string += "SGFN\n"
        string += "-- " + self.tag + "\n"
        string += "-- Sg Krg Pc\n"
        if dataincommentrow:
            string += self.sgcomment
            string += self.krgcomment
            if "krog" in self.table.columns:
                string += "-- krg = krog @ sw=%1.5f\n" % self.crosspoint()
            string += self.pccomment
        string += self.table[["sg", "krg", "pc"]].to_csv(
            sep=" ", float_format="%1.7f", header=None, index=False
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
        if dataincommentrow:
            string += self.sgcomment.replace("--", "!")
            string += self.krgcomment.replace("--", "!")
            string += self.krogcomment.replace("--", "!")
            string += "! krg = krog @ sw=%1.5f\n" % self.crosspoint()
            string += self.pccomment.replace("--", "!")
        string += self.table[["sg", "krg", "krog", "pc"]].to_csv(
            sep=" ", float_format="%1.7f", header=None, index=False
        )
        return string

    def crosspoint(self):
        """Locate and return the saturation point where krg = krog

        Accuracy of this crosspoint depends on the resolution chosen
        when initializing the saturation range (it uses linear
        interpolation to solve for the zero)

        Warning: Code duplication from WaterOil, with
        column names changed only
        """

        # Make a copy for calculations
        tmp = pd.DataFrame(self.table[["sg", "krg", "krog"]])
        tmp.loc[:, "krgminuskrog"] = tmp["krg"] - tmp["krog"]

        # Add a zero value for the difference column, and interpolate
        # the sw column to the zero value
        zerodf = pd.DataFrame(index=[len(tmp)], data={"krgminuskrog": 0.0})
        tmp = pd.concat([tmp, zerodf], sort=True)

        tmp.set_index("krgminuskrog", inplace=True)
        tmp.interpolate(method="slinear", inplace=True)

        return tmp[np.isclose(tmp.index, 0.0)].sg.values[0]

    def plotkrgkrog(
        self, ax=None, color="blue", alpha=1, label=None, linewidth=1, linestyle="-"
    ):
        """Plot krg and krog on a supplied matplotlib axis"""
        import matplotlib.pyplot as plt
        import matplotlib

        if not ax:
            matplotlib.style.use("ggplot")
            _, useax = matplotlib.pyplot.subplots()
        else:
            useax = ax
        self.table.plot(
            ax=useax,
            x="sg",
            y="krg",
            c=color,
            alpha=alpha,
            legend=None,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        self.table.plot(
            ax=useax,
            x="sg",
            y="krog",
            c=color,
            alpha=alpha,
            legend=None,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        if not ax:
            plt.show()
