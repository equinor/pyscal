# -*- coding: utf-8 -*-
"""A module for creating relative permeability input curves for
Eclipse and Nexus.
"""

from __future__ import division, absolute_import
from __future__ import print_function

from .utils import interpolator
from .wateroil import WaterOil
from .wateroilgas import WaterOilGas
from .gasoil import GasOil
from .scalrecommendation import SCALrecommendation

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
