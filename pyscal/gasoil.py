"""Representing a GasOil object"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

import pyscal
from pyscal.constants import EPSILON as epsilon
from pyscal.constants import MAX_EXPONENT, SWINTEGERS
from pyscal.utils.relperm import crosspoint, estimate_diffjumppoint, truncate_zeroness
from pyscal.utils.string import comment_formatter, df2str

logger = pyscal.getLogger_pyscal(__name__)


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
        swirr: Absolute minimal water saturation at infinite capillary
            pressure.  Not in use currently, except for in informational headers and
            for consistency checks.
        swl: First water saturation point in water tables. In GasOil, it
            is used to obtain the normalized oil and gas saturation.
        sgcr: Critical gas saturation. Gas will not be mobile before the
            gas saturation is above this value.
        sorg: Residual oil saturation after gas flooding. At this oil
            saturation, the oil has zero relative permeability.
        sgro: Residual gas, for use in gas-condensate modelling. Used
            as an endpoint for parametrized oil curve. Must be zero or equal
            to sgcr for compatibility with Eclipse three-point scaling.
        krgendanchor: Set to `sorg` (default) or something else, where to
            anchor `krgend`. If `sorg`, then the normalized gas
            saturation will be equal to 1 at `1 - swl - sorg`,
            if not, it will be 1 at `1 - swl`. If `sorg` is zero
            it does not matter. krgmax is only relevant when this anchor is sorg.
        h: Saturation step-length in the outputted table.
        tag: Optional string identifier, only used in comments.
        fast: Set to True if in order to skip some integrity checks
            and nice-to-have features. Not needed to set for normal pyscal
            runs, as speed is seldom crucial. Default False
    """

    def __init__(
        self,
        swirr: float = 0.0,
        sgcr: float = 0.0,
        h: Optional[float] = None,
        swl: float = 0.0,
        sorg: float = 0.0,
        sgro: float = 0.0,
        tag: str = "",
        krgendanchor: str = "sorg",
        fast: bool = False,
        _sgl: Optional[float] = None,  # Only to be used by GasWater.
    ) -> None:
        if h is None:
            h = 0.01

        assert -epsilon < swirr < 1.0 + epsilon, "0 <= swirr <= 1 is required"
        assert -epsilon < sgcr < 1, "0 <= sgcr < 1 is required"
        assert -epsilon < swl < 1, "0 <= swl < 1 is required"
        assert -epsilon < sorg < 1, "0 <= sorg <  1 is required"

        if krgendanchor is None:
            krgendanchor = ""

        assert isinstance(krgendanchor, str), "krgendanchor must be a string"

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

        # Avoid endpoints close to zero, set them to zero if
        # they are below a certain limit:
        self.sorg = truncate_zeroness(sorg, name="sorg")
        self.sgcr = truncate_zeroness(sgcr, name="sgcr")
        self.sgro = truncate_zeroness(sgro, name="sgro")

        if not (np.isclose(self.sgro, 0) or np.isclose(self.sgro - self.sgcr, 0)):
            raise ValueError(
                "sgro must be zero or equal to sgcr, for compatibility with "
                "Eclipse three-point scaling. "
                f"sgro: {sgro}, sgcr: {sgcr}."
            )

        self.tag = tag

        if _sgl is not None:
            assert -epsilon < _sgl < sgcr + epsilon, "0 <= sgl <= sgcr is required"
            self.sgl = truncate_zeroness(_sgl, name="_sgl")
        else:
            self.sgl = 0.0

        if krgendanchor in ["sorg", ""]:
            self.krgendanchor = krgendanchor
        else:
            logger.warning("Unknown krgendanchor %s, ignored", str(krgendanchor))
            self.krgendanchor = ""

        self.fast = fast

        if np.isclose(self.sorg, 0.0) and self.krgendanchor == "sorg":
            self.krgendanchor = ""  # This is critical to avoid bugs due to numerics.

        if self.krgendanchor == "sorg" and not 1 - self.sorg - self.swl - self.sgcr > 0:
            raise ValueError(
                "No saturation range left for gas curve between endpoints, check input"
            )
        if self.krgendanchor == "" and not 1 - self.swl - self.sgcr > 0:
            raise ValueError(
                "No saturation range left for gas curve between endpoints, check input"
            )

        sg_list = (
            [0.0]
            + [self.sgl]
            + [self.sgcr]
            + list(np.arange(self.sgcr + self.h, 1 - self.sorg - self.swl, self.h))
            + [1 - self.sorg - self.swl]
            + [1 - self.swl]
        )
        sg_list.sort()
        self.table = pd.DataFrame(sg_list, columns=["SG"])
        self.table["sgint"] = list(
            map(int, list(map(round, self.table["SG"] * SWINTEGERS)))
        )
        self.table.drop_duplicates("sgint", inplace=True)

        # Now sg=1-sorg-swl might be accidentally dropped, so make sure we
        # have it by replacing the closest value by 1 - sorg exactly
        sorgindex = (
            (self.table["SG"] - (1 - self.sorg - self.swl)).abs().sort_values().index[0]
        )
        self.table.loc[sorgindex, "SG"] = 1 - self.sorg - self.swl

        sgcrindex = (self.table["SG"] - (self.sgcr)).abs().sort_values().index[0]
        self.table.loc[sgcrindex, "SG"] = self.sgcr

        # Need to conserve sg=0 and sgcr:
        if sgcrindex == 0 and self.sgcr > 0.0:
            # This code is a safeguard againts truncate_zeroness(), which normally
            # prevents this from happening.
            zero_row = pd.DataFrame({"SG": 0}, index=[0])
            self.table = pd.concat([zero_row, self.table], sort=False).reset_index(
                drop=True
            )

        self.table.reset_index(inplace=True)
        self.table = self.table[["SG"]]
        self.table["SL"] = 1 - self.table["SG"]
        if krgendanchor == "sorg":
            # Normalized sg (sgn) is 0 at sgcr, and 1 at 1-swl-sorg
            assert 1 - swl - sgcr - sorg > epsilon
            self.table["SGN"] = (self.table["SG"] - self.sgcr) / (
                1 - self.swl - self.sgcr - self.sorg
            )
        else:
            assert 1 - swl - sgcr > epsilon
            self.table["SGN"] = (self.table["SG"] - self.sgcr) / (
                1 - self.swl - self.sgcr
            )

        # Normalized oil saturation should be 0 at sg=1-swl-sorg, and 1 at sg=sgro
        self.table["SON"] = (self.table["SL"] - self.sorg - self.swl) / (
            1 - self.sorg - self.swl - self.sgro
        )
        self.sgcomment: str = ""
        self.update_sgcomment_and_sorg()
        self.krgcomment = ""
        self.krogcomment = ""
        self.pccomment = ""

        logger.debug(
            "Initialized GasOil with %s saturation points", str(len(self.table))
        )

    def update_sgcomment_and_sorg(self):
        """Recalculate sorg in case it has table data has been manipulated"""
        self.sgcomment = (
            f"-- swirr={self.swirr:g}, sgcr={self.sgcr:g}, swl={self.swl:g}, "
            f"sorg={self.sorg:g}, sgro={self.sgro:g}, "
            f"krgendanchor={self.krgendanchor}\n"
        )
        if "KROG" in self.table.columns:
            self.sorg = (
                1
                - self.swl
                - self.table[np.isclose(self.table["KROG"], 0.0)].min()["SG"]
            )

    def add_fromtable(
        self,
        dframe: pd.DataFrame,
        sgcolname: str = "SG",
        krgcolname: str = "KRG",
        krogcolname: str = "KROG",
        pccolname: str = "PCOG",
        krgcomment: str = "",
        krogcomment: str = "",
        pccomment: str = "",
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
            raise ValueError(
                f"{sgcolname} not found in dataframe, can't read table data"
            )

        for col in [sgcolname, krgcolname, krogcolname, pccolname]:
            # Typecheck/convert all numerical columns:
            if col in dframe and not pd.api.types.is_numeric_dtype(dframe[col]):
                # Try to convert to numeric type
                try:
                    dframe[col] = dframe[col].astype(float)
                    logger.info("Converted column %s to numbers for fromtable()", col)
                except (TypeError, ValueError) as err:
                    raise ValueError(
                        f"Failed to parse column {col} as numbers for add_fromtable()"
                    ) from err

        if dframe[sgcolname].min() > 0.0:
            raise ValueError("sg must start at zero")
        swlfrominput = 1 - dframe[sgcolname].max()
        if abs(swlfrominput - self.swl) > epsilon:
            logger.warning(
                "swl=%f and 1-max(sg)=%f from incoming table do not seem compatible",
                self.swl,
                swlfrominput,
            )
            logger.warning("         Do not trust the result near the endpoint.")

        if 0 < swlfrominput - self.swl < epsilon:
            # Perturb max sg in incoming dataframe when we are this close,
            # or we will get into floating trouble when interpolating.
            dframe.loc[dframe[sgcolname].idxmax(), sgcolname] += swlfrominput - self.swl

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
            self.table["KRG"] = pchip(self.table["SG"], extrapolate=False)
            self.table["KRG"].fillna(method="ffill", inplace=True)
            self.table["KRG"].fillna(method="bfill", inplace=True)
            self.table["KRG"].clip(lower=0.0, upper=1.0, inplace=True)
            self.krgcomment = "-- krg from tabular input" + krgcomment + "\n"
            self.sgcr = self.estimate_sgcr()
        if krogcolname in dframe:
            if not (dframe[krogcolname].diff().dropna() < epsilon).all():
                raise ValueError("Incoming krog not decreasing")
            if dframe[krogcolname].max() > 1.0:
                raise ValueError("krog is above 1 in incoming table")
            if dframe[krogcolname].min() < 0.0:
                raise ValueError("krog is below 0 in incoming table")
            pchip = PchipInterpolator(
                dframe[sgcolname].astype(float), dframe[krogcolname].astype(float)
            )
            self.table["KROG"] = pchip(self.table["SG"], extrapolate=False)
            self.table["KROG"].fillna(method="ffill", inplace=True)
            self.table["KROG"].fillna(method="bfill", inplace=True)
            self.table["KROG"].clip(lower=0.0, upper=1.0, inplace=True)
            self.krogcomment = "-- krog from tabular input" + krogcomment + "\n"
            self.sorg = self.estimate_sorg()

            # For tabulated relperm data, correct knowlegde of sgro
            # is not important to pyscal (it is relevant when add_corey_gas()
            # etc is called). It is estimated here just for convenience.
            sgro_estimate = self.estimate_sgro()
            if not (
                np.isclose(sgro_estimate, 0.0) or np.isclose(sgro_estimate, self.sgcr)
            ):
                logger.warning(
                    "Estimated sgro (%s) from tabulated data was not 0 or sgcr (%s). "
                    "Reset to zero.",
                    str(sgro_estimate),
                    str(self.sgcr),
                )
                self.sgro = 0.0
            else:
                self.sgro = sgro_estimate
        if pccolname in dframe:
            # Incoming dataframe must cover the range:
            if dframe[sgcolname].max() < self.table["SG"].max():
                raise ValueError(
                    f"Too large swl for pcog interpolation, "
                    f"max incoming sg is {dframe[sgcolname].max()} "
                    f"and existing max(SG) is {self.table['SG'].max()}"
                )
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
            self.table["PC"] = pchip(self.table["SG"], extrapolate=False)
            if np.isnan(self.table["PC"]).any() or np.isinf(self.table["PC"]).any():
                raise ValueError("inf/nan in interpolated data, check input")
            self.pccomment = "-- pc from tabular input" + pccomment + "\n"

    def set_endpoints_linearpart_krg(
        self, krgend: float, krgmax: Optional[float] = None
    ):
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
            krgend: krg at sg = 1 - swl - sorg.
            krgmax: krg at Sg = 1 - swl. Default 1.

        """
        self.table.loc[self.table["SG"] <= self.sgcr, "KRG"] = 0

        if self.krgendanchor == "sorg":
            # Linear curve between krgendcanchor and 1-swl if krgend
            # is anchored to sorg
            if not krgmax:
                krgmax = 1
            tmp = pd.DataFrame(self.table[["SG"]])
            tmp["sgendnorm"] = (tmp["SG"] - (1 - (self.sorg + self.swl))) / (self.sorg)
            tmp["KRG"] = (
                tmp["sgendnorm"] * krgmax + (1 - tmp["sgendnorm"]) * krgend
            ).clip(lower=0.0, upper=1.0)
            self.table.loc[
                self.table["SG"] >= (1 - (self.sorg + self.swl + epsilon)), "KRG"
            ] = tmp.loc[tmp["SG"] >= (1 - (self.sorg + self.swl + epsilon)), "KRG"]
        else:
            self.table.loc[
                self.table["SG"] > (1 - (self.swl + epsilon)), "KRG"
            ] = krgend
            if krgmax and krgmax < 1.0 and self.sorg > 0:
                # Only warn if something else than default is in use
                logger.warning("krgmax ignored when not anchoring to sorg")

    def set_endpoints_linearpart_krog(
        self,
        kroend: float,
        kromax: Optional[float] = None,
    ):
        """Set linear parts of krog outside endpoints.

        Linear for sg in [0, sgro], from kromax to kroend, but nonzero
        sgro should only be used in gas-condensate modelling. When sgro
        is zero, kromax will be ignored.

        Zero for sg above 1 - sorg - swl.

        This function is used by add_corey/LET_oil(), and perhaps by other
        utility functions. It should not be necessary for end-users.

        Args:
            kroend: krog at sg=sgro, also if sgro=0
            kromax: krog at sg=0 for sgro > 0
        """
        if kromax is not None:
            if np.isclose(self.sgro, 0) and not np.isclose(kromax, kroend):
                logger.warning("kromax ignored when sgro is zero")
                kromax = kroend
            else:
                assert kroend <= kromax
        else:
            kromax = kroend

        # Special handling of the part close to sg=1, set to zero.
        self.table.loc[
            self.table["SG"] > 1 - self.sorg - self.swl - epsilon, "KROG"
        ] = 0

        # Floating point issues can cause a slight overshoot at sg=0:
        self.table.loc[self.table["KROG"] > kromax, "KROG"] = kromax

        self.table.loc[0, "KROG"] = kromax

    def add_corey_gas(
        self, ng: float = 2.0, krgend: float = 1.0, krgmax: Optional[float] = None
    ):
        """Add krg data through the Corey parametrization

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
        self.table["KRG"] = krgend * self.table["SGN"] ** ng

        self.set_endpoints_linearpart_krg(krgend, krgmax)

        if not krgmax:
            krgmax = 1
        self.krgcomment = (
            f"-- Corey krg, ng={ng:g}, krgend={krgend:g}, krgmax={krgmax:g}\n"
        )

    def add_corey_oil(
        self,
        nog: float = 2,
        kroend: float = 1,
        kromax: Optional[float] = None,
    ):
        """
        Add kro data through the Corey parametrization

        A column named 'kro' will be added to the internal DataFrame,
        replaced if it exists.

        All values above 1 - sorg - swl are set to zero.

        Arguments:
            nog: Corey exponent for oil
            kroend: Value for krog at normalized oil saturation 1
            kromax: Value for sg=0 if sgro > 0.

        Returns:
            None (modifies internal class state)
        """
        assert epsilon < nog < MAX_EXPONENT
        assert 0 < kroend <= 1.0

        self.table["KROG"] = kroend * self.table["SON"] ** nog

        self.set_endpoints_linearpart_krog(kroend, kromax)

        self.krogcomment = f"-- Corey krog, nog={nog:g}, kroend={kroend:g}"
        if kromax is not None:
            self.krogcomment += f", kromax={kromax:g}"

    def add_LET_gas(
        self,
        l: float = 2.0,
        e: float = 2.0,
        t: float = 2.0,
        krgend: float = 1.0,
        krgmax: Optional[float] = None,
    ):
        """
        Add gas relative permability data through the LET parametrization

        A column called 'KRG' will be added, replaced if it does not exist

        If krgendanchor is sorg, the LET curve ends at krgend at
        sg = 1 - swl - sorg, and then linear up to krgmax at
        sg = 1 - swl. If not, it ends at krgend at sg = 1 - swl.

        Arguments:
            l: L parameter in LET
            e: E parameter in LET
            t: T parameter in LET
            krgend: Value of krg at normalized gas saturation 1
            krgmax: Value of krg at gas saturation 1

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

        self.table["KRG"] = (
            krgend
            * self.table["SGN"] ** l
            / ((self.table["SGN"] ** l) + e * (1 - self.table["SGN"]) ** t)
        )
        # This equation is undefined for t a float and sgn=1, set explicitly:
        self.table.loc[np.isclose(self.table["SGN"], 1.0), "KRG"] = krgend

        self.set_endpoints_linearpart_krg(krgend, krgmax)

        if not krgmax:
            krgmax = 1
        self.krgcomment = (
            f"-- LET krg, l={l:g}, e={e:g}, t={t:g}, "
            f"krgend={krgend:g}, krgmax={krgmax:g}\n"
        )

    def add_LET_oil(
        self,
        l: float = 2.0,
        e: float = 2.0,
        t: float = 2.0,
        kroend: float = 1.0,
        kromax: Optional[float] = None,
    ):
        """Add oil (vs gas) relative permeability data through the Corey
        parametrization.

        A column named 'krog' will be added, replaced if it exists.

        All values where sg > 1 - sorg - swl are set to zero.

        Arguments:
            l: L parameter
            e: E parameter
            t: T parameter
            kroend: The value at gas saturation sgcr
            kromax: Value at sg=0 for sgro > 0

        """
        assert epsilon < l < MAX_EXPONENT
        assert epsilon < e < MAX_EXPONENT
        assert epsilon < t < MAX_EXPONENT
        assert 0 < kroend <= 1.0

        self.table["KROG"] = (
            kroend
            * self.table["SON"] ** l
            / ((self.table["SON"] ** l) + e * (1 - self.table["SON"]) ** t)
        )
        # This equation is undefined for t a float and son=1, set explicitly:
        self.table.loc[np.isclose(self.table["SON"], 1.0), "KROG"] = kroend

        self.set_endpoints_linearpart_krog(kroend, kromax)

        self.krogcomment = f"-- LET krog, l={l:g}, e={e:g}, t={t:g}, kroend={kroend:g}"
        if kromax is not None:
            self.krogcomment += f", kromax={kromax:g}"

    def estimate_sgro(self):
        """Estimate sgro of the current krog data

        sgro is estimated by searching for a linear part in kro
        from sg=0. In practice it is impossible to infer sgro = 0,
        since we are limited by h.

        When initializing GasOil, sgro must be either zero or equal
        to sgcr. This function only estimates sgro, and will not guarantee
        that condition.

        If the curve is linear everywhere, sgro will be returned as 1 - swl + h

        Returns:
            float: The estimated sgro
        """
        assert "KROG" in self.table
        assert self.table["KROG"].sum() > 0
        return estimate_diffjumppoint(self.table, xcol="SG", ycol="KROG", side="left")

    def estimate_sorg(self) -> float:
        """Estimate sorg of the current krg or krog data.

        sorg is estimated by searching for a linear part in krg downwards
        from sg=1-swl. In practice it is impossible to infer sorg = 0,
        since we are limited by h, and the last segment from sg=1-swl-h
        to sg=1-swl must always be assumed linear.

        If the curve is linear everywhere, sorg will be returned as sgcr + h

        If krgend is anchored to sorg, krg data is used to infer sorg. If not,
        krg cannot be used for this, and krog is used. sorg might be overestimated
        when krog is used if it very close to zero before reaching sorg.

        Returns:
            The estimated sorg.
        """
        if self.krgendanchor == "sorg":
            assert "KRG" in self.table
            assert self.table["KRG"].sum() > 0
            return self.table["SG"].max() - estimate_diffjumppoint(
                self.table, xcol="SG", ycol="KRG", side="right"
            )
        assert "KROG" in self.table
        assert self.table["KROG"].sum() > 0
        return self.table["SG"].max() - estimate_diffjumppoint(
            self.table, xcol="SG", ycol="KROG", side="right"
        )

    def estimate_sgcr(self) -> float:
        """Estimate sgcr of the current krog data.

        sgcr is the largest gas saturation for which the gas relative
        permeability is approximately zero, where approximately zero is
        roughly equivalent to how many digits are outputted
        by SGOF().

        Returns:
            The estimated sgcr.
        """
        return self.table[self.table["KRG"] < 10 * epsilon]["SG"].max()

    def crosspoint(self) -> float:
        """Locate and return the saturation point where krg = krog

        Accuracy of this crosspoint depends on the resolution chosen
        when initializing the saturation range (it uses linear
        interpolation to solve for the zero)

        Returns:
            The gas saturation where krg == krog, for relperm
            linearly interpolated in gas saturation.
        """
        return crosspoint(self.table, "SG", "KRG", "KROG")

    def selfcheck(self, mode: str = "SGOF") -> bool:
        """Check validities of the data in the table.

        This is to catch errors that are either physically wrong
        or at least causes Eclipse 100 to stop.

        Returns True if no errors are found, False if at least one
        is found.

        If you call SGOF/SLGOF, this function must not return False.

        Args:
            mode: If mode is "SGFN", krog is not required.
        """
        error = False
        if "KRG" not in self.table:
            logger.error("KRG data missing")
            error = True
        if not (self.table["SG"].diff().dropna() > -epsilon).all():
            logger.error("SG data not strictly increasing")
            error = True
        if (
            "KRG" in self.table
            and not (self.table["KRG"].diff().dropna() >= -epsilon).all()
        ):
            logger.error("KRG data not monotonically decreasing")
            error = True

        if mode != "SGFN":
            if "KROG" not in self.table:
                logger.error("KROG data missing")
                error = True
            if (
                "KROG" in self.table
                and not (self.table["KROG"].diff().dropna() <= epsilon).all()
            ):
                logger.error("KROG data not monotonically increasing")
                error = True
        if "KRG" in self.table and not np.isclose(min(self.table["KRG"]), 0.0):
            logger.error("KRG must start at zero")
            error = True
        if "PC" in self.table and self.table["PC"][0] > -epsilon:
            if not (self.table["PC"].diff().dropna() < epsilon).all():
                logger.error("PC data for gas-oil not strictly decreasing")
                error = True
        if "PC" in self.table and np.isinf(self.table["PC"].max()):
            logger.error("PC goes to infinity for gas-oil. ")
            error = True
        if "PC" in self.table.columns and np.isnan(self.table["PC"]).any():
            logger.error("pc data contains NaN")
            error = True

        for col in list(set(["SG", "KRG", "KROG"]) & set(self.table.columns)):
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

    def SGOF(self, header: bool = True, dataincommentrow: bool = True) -> str:
        """
        Produce SGOF input for Eclipse reservoir simulator.

        The columns sg, krg, krog and pc are outputted and
        formatted accordingly.

        Meta-information for the tabulated data are printed
        as Eclipse comments.

        Args:
            header: Whether the SGOF string should be emitted.
                If you have multiple satnums, you should have True only
                for the first (or False for all, and emit the SGOF yourself).
                Defaults to True.
            dataincommentrow: Whether metadata should be printed,
                defaults to True.
        """
        if not self.fast and not self.selfcheck():
            # selfcheck() will log error/warning messages
            return ""
        string = ""
        if "PC" not in self.table:
            self.table["PC"] = 0.0
            self.pccomment = "-- Zero capillary pressure\n"
        if header:
            string += "SGOF\n"
        string += comment_formatter(self.tag)
        string += "-- pyscal: " + str(pyscal.__version__) + "\n"
        if dataincommentrow:
            string += self.sgcomment
            string += self.krgcomment
            string += self.krogcomment
            if not self.fast:
                string += f"-- krg = krog @ sg={self.crosspoint():1.5f}\n"
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
        string += df2str(
            self.table[["SG", "KRG", "KROG", "PC"]],
            monotonicity={
                "KROG": {"sign": -1, "lower": 0, "upper": 1},
                "KRG": {"sign": 1, "lower": 0, "upper": 1},
                "PC": {"sign": 1, "allowzero": True},
            }
            if not self.fast
            else None,
        )
        string += "/\n"
        return string

    def slgof_df(self) -> pd.DataFrame:
        """Slice out an SLGOF table.

        This is used by the SLGOF() function, it is
        extracted as a single function to facilitate testing."""
        if "PC" not in self.table.columns:
            # Only happens when the SLGOF function is skipped (test code)
            self.table["PC"] = 0.0
        slgof = (
            self.table[
                self.table["SG"] <= 1 - self.sorg - self.swl + 1.0 / float(SWINTEGERS)
            ]
            .sort_values("SL")[["SL", "KRG", "KROG", "PC"]]
            .reset_index(drop=True)
        )
        return slgof

    def SLGOF(self, header: bool = True, dataincommentrow: bool = True) -> str:
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
        if "PC" not in self.table:
            self.table["PC"] = 0.0
            self.pccomment = "-- Zero capillary pressure\n"
        if header:
            string += "SLGOF\n"
        string += comment_formatter(self.tag)
        string += "-- pyscal: " + str(pyscal.__version__) + "\n"
        if dataincommentrow:
            string += self.sgcomment
            string += self.krgcomment
            string += self.krogcomment
            string += f"-- krg = krog @ sg={self.crosspoint():1.5f}\n"
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
        string += df2str(
            self.slgof_df(),
            monotonicity={
                "KROG": {"sign": 1, "lower": 0, "upper": 1},
                "KRG": {"sign": -1, "lower": 0, "upper": 1},
                "PC": {"sign": -1, "allowzero": True},
            }
            if not self.fast
            else None,
        )
        string += "/\n"
        return string

    def SGFN(
        self,
        header: bool = True,
        dataincommentrow: bool = True,
        sgcomment: Optional[str] = None,
        crosspointcomment: Optional[str] = None,
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
            sgcomment: Provide the string to include in the comment
                section for describing the saturation endpoints. Used
                by GasWater.
            crosspointcomment: String to be used for crosspoint comment
                string, overrides what this object can provide. Used by GasWater.
                If None, it will be computed, use empty string to avoid.
        """
        if not self.selfcheck(mode="SGFN"):
            # Selfcheck will issue error messages.
            return ""
        string = ""
        if "PC" not in self.table.columns:
            self.table["PC"] = 0.0
            self.pccomment = "-- Zero capillary pressure\n"
        if header:
            string += "SGFN\n"
        string += comment_formatter(self.tag)
        string += "-- pyscal: " + str(pyscal.__version__) + "\n"
        if dataincommentrow:
            if sgcomment is not None:
                string += sgcomment
            else:
                string += self.sgcomment
            string += self.krgcomment
            if crosspointcomment is None:
                if "KROG" in self.table.columns:
                    string += f"-- krg = krog @ sg={self.crosspoint():1.5f}\n"
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
        string += df2str(
            self.table[["SG", "KRG", "PC"]],
            monotonicity={
                "KRG": {"sign": 1, "lower": 0, "upper": 1},
                "PC": {"sign": 1, "allowzero": True},
            }
            if not self.fast
            else None,
        )
        string += "/\n"
        return string

    def GOTABLE(self, header: bool = True, dataincommentrow: bool = True) -> str:
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
        if "PC" not in self.table.columns:
            self.table["PC"] = 0.0
            self.pccomment = "-- Zero capillary pressure\n"
        if header:
            string += "GOTABLE\n"
            string += "SG KRG KROG PC\n"
        string += "! pyscal: " + str(pyscal.__version__) + "\n"
        if dataincommentrow:
            string += self.sgcomment.replace("--", "!")
            string += self.krgcomment.replace("--", "!")
            string += self.krogcomment.replace("--", "!")
            string += f"! krg = krog @ sw={self.crosspoint():1.5f}\n"
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
        string += df2str(
            self.table[["SG", "KRG", "KROG", "PC"]],
            monotonicity={
                "KROG": {"sign": -1, "lower": 0, "upper": 1},
                "KRG": {"sign": 1, "lower": 0, "upper": 1},
                "PC": {"sign": 1, "allowzero": True},
            }
            if self.fast
            else None,
        )
        return string

    def plotkrgkrog(
        self,
        mpl_ax=None,
        color: str = "blue",
        alpha: float = 1.0,
        linewidth: int = 1,
        linestyle: str = "-",
        marker: Optional[str] = None,
        label: Optional[str] = None,
        logyscale: bool = False,
    ):
        """Plot krg and krog

        If mpl_ax is not None, it will be used as a
        matplotlib axis to plot on, if None, a fresh plot
        will be made.
        """
        # pylint: disable=import-outside-toplevel
        # Lazy import of matplotlib for speed reasons
        import matplotlib
        import matplotlib.pyplot as plt

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
            x="SG",
            y="KRG",
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
            x="SG",
            y="KROG",
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
