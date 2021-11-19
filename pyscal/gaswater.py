"""Object to represent GasWater, implemented as a Container
object for one WaterOil and one GasOil object"""

from typing import Optional

import pandas as pd

from pyscal import getLogger_pyscal
from pyscal.utils.relperm import crosspoint

from .gasoil import GasOil
from .wateroil import WaterOil

logger = getLogger_pyscal(__name__)


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
        swirr: Irreducible water saturation for capillary pressure
        swl: First water saturation point in outputted tables.
        sgl: Minimum gas saturation in a gas paleozone.
        swcr: Critical water saturation, water is immobile below this
        sgrw: Residual gas saturation after water flooding.
        sgcr: Critical gas saturation, gas is immobile below this
        h: Saturation intervals in generated tables.
        tag: Optional text that will be included as comments.
        fast: Set to True if you prefer speed over robustness. Not recommended,
            pyscal will not guarantee valid output in this mode.
    """

    def __init__(
        self,
        swirr: float = 0.0,
        swl: float = 0.0,
        sgl: float = 0.0,
        swcr: float = 0.0,
        sgrw: float = 0.0,
        sgcr: float = 0.0,
        h: Optional[float] = None,
        tag: str = "",
        fast: bool = False,
    ) -> None:
        """Sets up the saturation range for a GasWater object,
        by initializing one WaterOil and one GasOil object, with
        endpoints set to fit with the GasWater proxy object."""
        self.fast: bool = fast

        if h is None:
            h = 0.01

        self.wateroil: WaterOil = WaterOil(
            swirr=swirr,
            swl=swl,
            swcr=swcr,
            sorw=sgrw,
            h=h,
            tag=tag,
            fast=fast,
            _sgcr=sgcr,
            _sgl=sgl,
        )

        self.sgcr: float = sgcr
        self.sgrw: float = sgrw
        self.sgl: float = sgl
        # (remaining parameters are implemented as property functions)

        self.gasoil: GasOil = GasOil(
            swirr=swirr,  # Reserved for use in capillary pressure
            sgcr=sgcr,
            sorg=0,  # Irrelevant for GasWater
            swl=swl,
            h=h,
            tag=tag,
            fast=fast,
            _sgl=sgl,
        )

        # Dummy oil curves, just to avoid objects being invalid.
        self.wateroil.add_corey_oil()
        self.gasoil.add_corey_oil()

    def selfcheck(self) -> bool:
        """Run selfcheck on the data.

        Performs tests if necessary data is ready in the object for
        printing Eclipse include files, and checks some numerical
        properties (direction and monotonicity)
        """
        if self.wateroil is not None and self.gasoil is not None:
            return self.wateroil.selfcheck() and self.gasoil.selfcheck()
        logger.error("None objects in GasWater (bug)")
        return False

    def add_corey_water(
        self, nw: float = 2.0, krwend: float = 1.0, krwmax: Optional[float] = None
    ) -> None:
        """Add krw data through the Corey parametrization

        A column named 'krw' will be added. If it exists, it will
        be replaced.

        The Corey model applies for sw < 1 - sgrw. For higher
        water saturations, krw is linear between krwend and krwmax.

        krwmax will be ignored if sgrw is close to zero

        Args:
            nw: Corey parameter for water.
            krwend: value of krw at 1 - sgcr.
            krwmax: maximal value at Sw=1. Default 1
        """
        self.wateroil.add_corey_water(nw, krwend, krwmax)

    def add_corey_gas(self, ng: float = 2.0, krgend: float = 1.0) -> None:
        """Add krg data through the Corey parametrization

        A column named 'krg' will be added. If it exists, it will
        be replaced.

        Args:
            ng: Corey parameter for gas
            krgend: value of krg at swl.
        """
        self.gasoil.add_corey_gas(ng, krgend, krgmax=None)

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
        self.wateroil.add_LET_water(l, e, t, krwend, krwmax)

    def add_LET_gas(
        self, l: float = 2.0, e: float = 2.0, t: float = 2.0, krgend: float = 1.0
    ) -> None:
        """Add krg data through the LET parametrization

        A column named 'krg' will be added. If it exists, it will
        be replaced.

        Args:
            l: LET parameter
            e: LET parameter
            t: LET parameter
            krgend: value of krg at swl
        """
        self.gasoil.add_LET_gas(l, e, t, krgend, krgmax=None)

    @is_documented_by(WaterOil.add_simple_J)
    def add_simple_J(
        self,
        a: float = 5.0,
        b: float = -1.5,
        poro_ref: float = 0.25,
        perm_ref: float = 100.0,
        drho: float = 300.0,
        g: float = 9.81,
    ) -> None:
        """Add a simple J-function, handed over to the WaterOil object"""
        self.wateroil.add_simple_J(a, b, poro_ref, perm_ref, drho, g)

    @is_documented_by(WaterOil.add_simple_J_petro)
    def add_simple_J_petro(
        self,
        a: float,
        b: float,
        poro_ref: float = 0.25,
        perm_ref: float = 100.0,
        drho: float = 300.0,
        g: float = 9.81,
    ) -> None:
        """Add a simple petrophysical variant of the J-function, handed
        over to the WaterOil object"""
        self.wateroil.add_simple_J_petro(a, b, poro_ref, perm_ref, drho, g)

    def SWFN(self, header: bool = True, dataincommentrow: bool = True):
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
                crosspointcomment = f"-- krw = krg @ sw={crosspoint_value:1.5f}\n"
            else:
                crosspointcomment = ""
        return self.wateroil.SWFN(
            header,
            dataincommentrow,
            swcomment=self.swcomment,
            crosspointcomment=crosspointcomment,
        )

    def SGFN(self, header: bool = True, dataincommentrow: bool = True):
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
                crosspointcomment = f"-- krw = krg @ sw={crosspoint_value:1.5f}\n"
            else:
                crosspointcomment = ""
        return self.gasoil.SGFN(
            header,
            dataincommentrow,
            sgcomment=self.sgcomment,
            crosspointcomment=crosspointcomment,
        )

    def crosspoint(self) -> Optional[float]:
        """Calculate the sw value where krg == krw.

        Accuracy of this crosspoint depends on the resolution chosen
        when initializing the saturation range (it uses linear
        interpolation to solve for the zero)

        Returns:
            The gas saturation where krw == krg, for relperm linearly
            interpolated in water saturation.
        """
        if not {"SW", "KRW"}.issubset(self.wateroil.table.columns):
            logger.warning("Can't compute crosspoint when KRW is not present")
            return None
        if not {"SL", "KRG"}.issubset(self.gasoil.table.columns):
            logger.warning("Can't compute crosspoint when KRG is not present")
            return None
        dframe = pd.concat(
            [self.wateroil.table[["SW", "KRW"]], self.gasoil.table[["SL", "KRG"]]],
            sort=False,
        )
        # The  "SL" column in the GasOil object corresponds exactly to "SW" in WaterOil
        # but since they are floating point, we do not want to "merge" dataframes on it,
        # rather concatenate and let linear interpolation fill in values.
        dframe["SW"].fillna(value=0, inplace=True)
        dframe["SL"].fillna(value=0, inplace=True)
        dframe["sat"] = dframe["SL"] + dframe["SW"]
        dframe = (
            dframe.set_index("sat")
            .sort_index()
            .interpolate(method="slinear")
            .dropna()
            .reset_index()
        )
        return crosspoint(dframe, "sat", "KRW", "KRG")

    def plotkrwkrg(
        self,
        mpl_ax=None,
        color: str = "blue",
        alpha: float = 1,
        linewidth: int = 1,
        linestyle: str = "-",
        marker: Optional[str] = None,
        label: str = "",
        logyscale: bool = False,
    ):
        """Plot krw and krg

        If the argument 'mpl_ax' is not supplied, a new plot
        window will be made. If supplied, it will draw on
        the specified axis."""

        # pylint: disable=import-outside-toplevel
        # Lazy import of matplotlib for speed reasons.
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
        self.wateroil.table.plot(
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
        self.gasoil.table.plot(
            ax=useax,
            x="SL",
            y="KRG",
            c=color,
            alpha=alpha,
            label=None,
            legend=None,
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
        )
        plt.xlabel("SW")
        if mpl_ax is None:
            plt.show()

    @property
    def swirr(self) -> float:
        """Get the swirr, irreducible water saturation used for pc-init"""
        return self.wateroil.swirr

    @property
    def swl(self) -> float:
        """Get the swl"""
        return self.wateroil.swl

    @property
    def swcr(self) -> float:
        """Get the swcr"""
        return self.wateroil.swcr

    @property
    def swcomment(self) -> str:
        """Get a string representation of the endpoints used for water"""
        return self.wateroil.swcomment.replace("sorw", "sgrw")

    @property
    def sgcomment(self) -> str:
        """Get a string representation of the endpoints used for gas"""
        return (
            self.gasoil.sgcomment.replace("sorg", "sgrw")
            .replace(", krgendanchor=sgrw", "")
            .replace(", krgendanchor=", "")
        )

    @property
    def krwcomment(self) -> str:
        """Get a string representation describing krw"""
        return self.wateroil.krwcomment

    @property
    def krgcomment(self) -> str:
        """Get a string representation describing krg"""
        return self.gasoil.krgcomment

    @property
    def tag(self) -> str:
        """Get the user configured tag"""
        if self.wateroil.tag == self.gasoil.tag:
            return self.wateroil.tag
        raise ValueError("Internal tag-inconsistency in GasWater")
