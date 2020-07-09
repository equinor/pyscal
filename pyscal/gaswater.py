"""Object to represent GasWater, implemented as a Container
object for one WaterOil and one GasOil object"""
from __future__ import division, absolute_import
from __future__ import print_function

import logging

import pandas as pd

from .wateroil import WaterOil
from .gasoil import GasOil
from . import utils

logging.basicConfig()
logger = logging.getLogger(__name__)


def is_documented_by(original):
    """Decorator to avoid duplicating function docstrings"""

    def wrapper(target):
        target.__doc__ = original.__doc__
        return target

    return wrapper


class GasWater(object):
    """A representation of two-phase properties for gas-water

    Internally, this class handles gas-water by using one WaterOil
    object and one GasOil object, with dummy parameters for oil.

    Args:
        swirr (float): Irreducible water saturation for capillary pressure
        swl (float): First water saturation point in outputted tables.
        swcr (float): Critical water saturation, water is immobile below this
        sgrw (float): Residual gas saturation after water flooding.
        sgcr (float): Critical gas saturation, gas is immobile below this
        h (float): Saturation intervals in generated tables.
        tag (str): Optional text that will be included as comments.
        fast (bool): Set to True if you prefer speed over robustness. Not recommended,
            pyscal will not guarantee valid output in this mode.
   """

    def __init__(
        self, swirr=0, swl=0.0, swcr=0.0, sgrw=0.0, sgcr=0, h=0.01, tag="", fast=False,
    ):
        """Sets up the saturation range for a GasWater object,
        by initializing one WaterOil and one GasOil object, with
        endpoints set to fit with the GasWater proxy object."""
        self.fast = fast

        self.wateroil = WaterOil(
            swirr=swirr,
            swl=swl,
            swcr=swcr,
            sorw=sgrw,
            h=h,
            tag=tag,
            fast=fast,
            _sgcr=sgcr,
        )

        self.sgcr = sgcr
        self.sgrw = sgrw
        # (remaining parameters are implemented as property functions)

        self.gasoil = GasOil(
            swirr=swirr,  # Reserved for use in capillary pressure
            sgcr=sgcr,
            sorg=0,  # Irrelevant for GasWater
            swl=swl,
            h=h,
            tag=tag,
            fast=fast,
        )

        # Dummy oil curves, just to avoid objects
        # being invalid.
        self.wateroil.add_corey_oil()
        self.gasoil.add_corey_oil()

    def selfcheck(self):
        """Run selfcheck on the data.

        Performs tests if necessary data is ready in the object for
        printing Eclipse include files, and checks some numerical
        properties (direction and monotonocity)

        Returns:
            bool
        """
        if self.wateroil is not None and self.gasoil is not None:
            return self.wateroil.selfcheck() and self.gasoil.selfcheck()
        logger.error("None objects in GasWater (bug)")
        return False

    def add_corey_water(self, nw=2, krwend=1, krwmax=None):
        """Add krw data through the Corey parametrization

        A column named 'krw' will be added. If it exists, it will
        be replaced.

        The Corey model applies for sw < 1 - sgrw. For higher
        water saturations, krw is linear between krwend and krwmax.

        krwmax will be ignored if sgrw is close to zero

        Args:
            nw (float): Corey parameter for water.
            krwend (float): value of krw at 1 - sorw.
            krwmax (float): maximal value at Sw=1. Default 1
        """
        self.wateroil.add_corey_water(nw, krwend, krwmax)

    def add_corey_gas(self, ng=2, krgend=1):
        """Add krg data through the Corey parametrization

        A column named 'krg' will be added. If it exists, it will
        be replaced.

        Args:
            ng (float): Corey parameter for gas
            krgend (float): value of krg at swl.
        """
        self.gasoil.add_corey_gas(ng, krgend, krgmax=None)

    def add_LET_water(self, l=2, e=2, t=2, krwend=1, krwmax=None):
        """Add krw data through LET parametrization

        The LET model applies for sw < 1 - sgrw. For higher
        water saturations, krw is linear between krwend and krwmax.

        krwmax will be ignored if sorw is close to zero.

        Args:
            l (float): LET parameter
            e (float): LET parameter
            t (float): LET parameter
            krwend (float): value of krw at 1 - sorw
            krwmax (float): maximal value at Sw=1. Default 1
        """
        self.wateroil.add_LET_water(l, e, t, krwend, krwmax)

    def add_LET_gas(self, l=2, e=2, t=2, krgend=1):
        """Add krg data through the LET parametrization

        A column named 'krg' will be added. If it exists, it will
        be replaced.

        Args:
            l (float): LET parameter
            e (float): LET parameter
            t (float): LET parameter
            krgend (float): value of krg at swl
        """
        self.gasoil.add_LET_gas(l, e, t, krgend, krgmax=None)

    @is_documented_by(WaterOil.add_simple_J)
    def add_simple_J(self, a=5, b=-1.5, poro_ref=0.25, perm_ref=100, drho=300, g=9.81):
        self.wateroil.add_simple_J(a, b, poro_ref, perm_ref, drho, g)

    @is_documented_by(WaterOil.add_simple_J_petro)
    def add_simple_J_petro(self, a, b, poro_ref=0.25, perm_ref=100, drho=300, g=9.81):
        self.wateroil.add_simple_J_petro(a, b, poro_ref, perm_ref, drho, g)

    def SWFN(self, header=True, dataincommentrow=True):
        """Produce SWFN input to Eclipse

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
        """
        if self.fast:
            crosspointcomment = ""
        else:
            crosspoint_value = self.crosspoint()
            if crosspoint_value is not None:
                crosspointcomment = "-- krw = krg @ sw=%1.5f\n" % crosspoint_value
            else:
                crosspointcomment = ""
        return self.wateroil.SWFN(
            header,
            dataincommentrow,
            swcomment=self.swcomment,
            crosspointcomment=crosspointcomment,
        )

    def SGFN(self, header=True, dataincommentrow=True):
        """Produce SGFN input for Eclipse reservoir simulator.

        The columns sg and krg are outputted and formatted accordingly.

        Meta-information for the tabulated data are printed as Eclipse comments.

        Args:
            header: boolean for whether the SGFN string should be emitted.
                If you have multiple satnums, you should have True only
                for the first (or False for all, and emit the SGFN yourself).
                Defaults to True.
            dataincommentrow: boolean for wheter metadata should be printed,
                defaults to True.
        """
        if self.fast:
            crosspointcomment = ""
        else:
            crosspoint_value = self.crosspoint()
            if crosspoint_value is not None:
                crosspointcomment = "-- krw = krg @ sw=%1.5f\n" % crosspoint_value
            else:
                crosspointcomment = ""
        return self.gasoil.SGFN(
            header,
            dataincommentrow,
            sgcomment=self.sgcomment,
            crosspointcomment=crosspointcomment,
        )

    def crosspoint(self):
        """Calculate the sw value where krg == krw.

        Accuracy of this crosspoint depends on the resolution chosen
        when initializing the saturation range (it uses linear
        interpolation to solve for the zero)

        Returns:
            float: the gas saturation where krw == krg, for relperm
                linearly interpolated in water saturation.
        """
        if not {"sw", "krw"}.issubset(self.wateroil.table.columns):
            logger.warning("Can't compute crosspoint when krw is not present")
            return None
        if not {"sl", "krg"}.issubset(self.gasoil.table.columns):
            logger.warning("Can't compute crosspoint when krg is not present")
            return None
        dframe = pd.concat(
            [self.wateroil.table[["sw", "krw"]], self.gasoil.table[["sl", "krg"]]],
            sort=False,
        )
        # The  "sl" column in the GasOil object corresponds exactly to "sw" in WaterOil
        # but since they are floating point, we do not want to "merge" dataframes on it,
        # rather concatenate and let linear interpolation fill in values.
        dframe["sw"].fillna(value=0, inplace=True)
        dframe["sl"].fillna(value=0, inplace=True)
        dframe["sat"] = dframe["sl"] + dframe["sw"]
        dframe = (
            dframe.set_index("sat")
            .sort_index()
            .interpolate(method="slinear")
            .dropna()
            .reset_index()
        )
        return utils.crosspoint(dframe, "sat", "krw", "krg")

    def plotkrwkrg(
        self,
        mpl_ax=None,
        color="blue",
        alpha=1,
        linewidth=1,
        linestyle="-",
        marker=None,
        label="",
        logyscale=False,
    ):
        """Plot krw and krg

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
        self.wateroil.table.plot(
            ax=useax,
            x="sw",
            y="krw",
            c=color,
            alpha=alpha,
            legend=None,
            label=label,
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
        )
        self.gasoil.table.plot(
            ax=useax,
            x="sl",
            y="krg",
            c=color,
            alpha=alpha,
            label=None,
            legend=None,
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
        )
        plt.xlabel("sw")
        if mpl_ax is None:
            plt.show()

    @property
    def swirr(self):
        return self.wateroil.swirr

    @property
    def swl(self):
        return self.wateroil.swl

    @property
    def swcr(self):
        return self.wateroil.swcr

    @property
    def swcomment(self):
        return self.wateroil.swcomment.replace("sorw", "sgrw")

    @property
    def sgcomment(self):
        return (
            self.gasoil.sgcomment.replace("sorg", "sgrw")
            .replace(", krgendanchor=sgrw", "")
            .replace(", krgendanchor=", "")
        )

    @property
    def krwcomment(self):
        return self.wateroil.krwcomment

    @property
    def krgcomment(self):
        return self.gasoil.krgcomment

    @property
    def tag(self):
        if self.wateroil.tag == self.gasoil.tag:
            return self.wateroil.tag
        else:
            raise ValueError("Internal tag-inconsistency in GasWater")
