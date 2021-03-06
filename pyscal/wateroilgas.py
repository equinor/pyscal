"""Container object for one WaterOil and one GasOil object"""

import logging
import numpy as np
import pandas as pd

import pyscal
from pyscal.constants import SWINTEGERS

from pyscal.utils.string import df2str, comment_formatter
from .wateroil import WaterOil
from .gasoil import GasOil

logger = logging.getLogger(__name__)


class WaterOilGas(object):

    """A representation of three-phase properties for oil-water-gas

    Use one object for each satnum.

    One WaterOil and one GasOil object will be created, with
    compatible saturation ranges. Access the class members 'wateroil'
    and 'gasoil' directly to add curves.

    All arguments to the initializer can be defaulted, and all are zero except h.

    Either the gasoil or the wateroil member is allowed to be None
    in case of only two phases present.

    Args:
        swirr (float): Irreducible water saturation for capillary pressure
        swl (float): First water saturation point in outputted tables.
        swcr (float): Critical water saturation, water is immobile below this
        sorw (float): Residual oil saturation
        sgcr (float): Critical gas saturation, gas is immobile below this
        h (float): Saturation intervals in generated tables.
        tag (str): Optional text that will be included as comments.
        fast (bool): Set to True if you prefer speed over robustness. Not recommended,
            pyscal will not guarantee valid output in this mode.
    """

    def __init__(
        self,
        swirr=0,
        swl=0.0,
        swcr=0.0,
        sorw=0.00,
        sorg=0,
        sgcr=0,
        h=0.01,
        tag="",
        fast=False,
    ):
        """Sets up the saturation range for three phases"""
        self.fast = fast
        self.wateroil = WaterOil(
            swirr=swirr, swl=swl, swcr=swcr, sorw=sorw, h=h, tag=tag, fast=fast
        )
        self.gasoil = GasOil(
            swirr=swirr, sgcr=sgcr, sorg=sorg, swl=swl, h=h, tag=tag, fast=fast
        )

    def selfcheck(self):
        """Run selfcheck on both wateroil and gasoil.

        Returns true only if both passes if both are present.

        If only wateroil or gasoil is present (the other
        being None, only the relevant is checked.

        Returns:
            bool
        """
        if self.wateroil is not None and self.gasoil is not None:
            return self.wateroil.selfcheck() and self.gasoil.selfcheck()
        if self.wateroil is not None:
            return self.wateroil.selfcheck()
        if self.gasoil is not None:
            return self.gasoil.selfcheck()
        logger.error("Both wateroil and gasoil are None in WaterOilGas")
        return False

    def SWOF(self, header=True, dataincommentrow=True):
        """Return a SWOF string. Delegated to the wateroil object"""
        if self.wateroil is None:
            logger.error("No WaterOil object in this WaterOilGas object")
            return ""
        if "KRW" not in self.wateroil.table or "KROW" not in self.wateroil.table:
            logger.error("Missing KRW/KROW curves in WaterOilGas object")
            return ""
        return self.wateroil.SWOF(header, dataincommentrow)

    def SGOF(self, header=True, dataincommentrow=True):
        """Return a SGOF string. Delegated to the gasoil object."""
        if self.gasoil is None:
            logger.error("No GasOil object in this WaterOilGas object")
            return ""
        if "KRG" not in self.gasoil.table or "KROG" not in self.gasoil.table:
            logger.error("Missing KRG/KROG curves in WaterOilGas object")
            return ""
        return self.gasoil.SGOF(header, dataincommentrow)

    def SLGOF(self, header=True, dataincommentrow=True):
        """Return a SLGOF string. Delegated to the gasoil object."""
        if self.gasoil is None:
            logger.error("No GasOil object in this WaterOilGas object")
            return ""
        if "KRG" not in self.gasoil.table or "KROG" not in self.gasoil.table:
            logger.error("Missing KRG/KROG in WaterOilGas object")
            return ""
        return self.gasoil.SLGOF(header, dataincommentrow)

    def SGFN(self, header=True, dataincommentrow=True):
        """Return a SGFN string. Delegated to the gasoil object."""
        if self.gasoil is None:
            logger.error("No GasOil object in this WaterOilGas object")
            return ""
        if "KRG" not in self.gasoil.table:
            logger.error("Missing KRG in WaterOilGas object")
            return ""
        return self.gasoil.SGFN(header, dataincommentrow)

    def SWFN(self, header=True, dataincommentrow=True):
        """Return a SWFN string. Delegated to the wateroil object."""
        if self.wateroil is None:
            logger.error("No WaterOil object in this WaterOilGas object")
            return ""
        if "KRW" not in self.wateroil.table:
            logger.error("Missing KRW in WaterOilGas object")
            return ""
        return self.wateroil.SWFN(header, dataincommentrow)

    def SOF3(self, header=True, dataincommentrow=True):
        """Return a SOF3 string, combining data from the wateroil and
        gasoil objects.

        So - the oil saturation ranges from 0 to 1-swl. The saturation points
        from the WaterOil object is used to generate these
        """
        if "KROW" not in self.wateroil.table or "KROG" not in self.gasoil.table:
            logger.error("Both WaterOil and GasOil krow/krog is needed for SOF3")
            return ""
        self.threephaseconsistency()

        # Copy of the wateroil data:
        table = pd.DataFrame(self.wateroil.table[["SW", "KROW"]])
        table["SO"] = 1 - table["SW"]

        # Copy of the gasoil data:
        gastable = pd.DataFrame(self.gasoil.table[["SG", "KROG"]])
        gastable["SO"] = 1 - gastable["SG"] - self.wateroil.swl

        # Merge WaterOil and GasOil on oil saturation, interpolate for
        # missing data (potentially different sg- and sw-grids)
        sof3table = (
            pd.concat([table, gastable], sort=True)
            .set_index("SO")
            .sort_index()
            .interpolate(method="slinear")
            .fillna(method="ffill")
            .fillna(method="bfill")
            .reset_index()
        )
        sof3table["soint"] = list(
            map(int, list(map(round, sof3table["SO"] * SWINTEGERS)))
        )
        sof3table.drop_duplicates("soint", inplace=True)

        # The 'so' column has been calculated from floating point numbers
        # and the zero value easily becomes a negative zero, circumvent this:
        zerorow = np.isclose(sof3table["SO"], 0.0)
        sof3table.loc[zerorow, "SO"] = abs(sof3table.loc[zerorow, "SO"])

        string = ""
        if header:
            string += "SOF3\n"
        wo_tag = comment_formatter(self.wateroil.tag)
        go_tag = comment_formatter(self.gasoil.tag)
        if wo_tag != go_tag:
            string += wo_tag
            string += go_tag
        else:
            # Only print once if they are equal
            string += wo_tag
        string += "-- pyscal: " + str(pyscal.__version__) + "\n"
        if dataincommentrow:
            string += self.wateroil.swcomment
            string += self.gasoil.sgcomment
            string += self.wateroil.krowcomment
            string += self.gasoil.krogcomment

        width = 10
        string += (
            "-- "
            + "SO".ljust(width - 3)
            + "KROW".ljust(width)
            + "KROG".ljust(width)
            + "\n"
        )
        string += df2str(sof3table[["SO", "KROW", "KROG"]])
        string += "/\n"
        return string

    @property
    def swirr(self):
        """Get the swirr used for the WaterOil object"""
        return self.wateroil.swirr

    @property
    def swl(self):
        """Get the swl used for the WaterOil object"""
        return self.wateroil.swl

    @property
    def sorg(self):
        """Get the sorg used in the GasOil object"""
        return self.gasoil.sorg

    @property
    def sorw(self):
        """Get the sorw used for the WaterOil object"""
        return self.wateroil.sorw

    @property
    def tag(self):
        """Get the tag, a combination of the tags in WaterOil
        and GasOil only if they are different"""
        if self.wateroil.tag == self.gasoil.tag:
            return self.wateroil.tag
        return self.wateroil.tag + " " + self.gasoil.tag

    def threephaseconsistency(self):
        """Perform consistency checks on produced curves, similar
        to what Eclipse does at startup.

        If any (likely) errors are found, these are printed
        as warnings and this function returns False.

        "Errors" from this function should be treated as
        warnings, as false positives from this function should
        not have breaking consequences.

        Args:
            None, examines self.

        Returns:
            bool - True if this function approves the data
        """

        # Eclipse errors:

        # 1: Error in saturation table number 1 at the maximum oil
        # saturation (0.9) krow and krog should both be equal the oil
        # relative permeability for a system with oil and connate
        # water only - but in this case they are different (krow=1.0
        # and krog=0.93)

        if self.wateroil is None or self.gasoil is None:
            # Nothing here to check if we only have two phases
            return True

        wog_is_ok = True
        if not np.isclose(
            self.wateroil.table["KROW"].max(), self.gasoil.table["KROG"].max()
        ):
            logger.warning(
                "Eclipse will fail, max(KROW)=%g is not equal to max(KROG)=%g",
                self.wateroil.table["KROW"].max(),
                self.gasoil.table["KROG"].max(),
            )
            wog_is_ok = False

        # 2: Inconsistent end points in saturation table 1 the maximum
        # gas saturation (0.91) plus the connate water saturation
        # (0.10) must not exceed 1.0
        if self.gasoil.table["SG"].max() + self.wateroil.table["SW"].min() > 1.0:
            logger.warning("Eclipse will fail, max(SG) + swl > 1.0")
            wog_is_ok = False

        # 3: Warning: Consistency problem for gas phase endpoint (krgr > krg)
        # in grid cell (i, j, k) for saturation end-points krgr=1.0
        # krg = 0.49.

        # 4: Warning: Consistency problem for oil phase endpoint (sgu > 1-swl)
        # in grid cell (i, j, k) for saturation endpoints sgu=0.78,
        # swl=0.45, (1-swl) = 0.55

        return wog_is_ok
