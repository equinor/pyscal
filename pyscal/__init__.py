"""A module for creating relative permeability input curves for
Eclipse and Nexus.
"""

from __future__ import division, absolute_import
from __future__ import print_function


from .utils import interpolator  # noqa
from .wateroil import WaterOil  # noqa
from .wateroilgas import WaterOilGas  # noqa
from .gasoil import GasOil  # noqa
from .scalrecommendation import SCALrecommendation  # noqa
from .factory import PyscalFactory  # noqa


from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
