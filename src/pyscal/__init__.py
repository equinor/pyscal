"""pyscal"""

from .__version__ import __version__
from .factory import PyscalFactory
from .gasoil import GasOil
from .gaswater import GasWater
from .pyscal_logger import getLogger_pyscal
from .pyscallist import PyscalList
from .scalrecommendation import SCALrecommendation
from .wateroil import WaterOil
from .wateroilgas import WaterOilGas

__all__ = [
    "GasOil",
    "GasWater",
    "PyscalFactory",
    "PyscalList",
    "SCALrecommendation",
    "WaterOil",
    "WaterOilGas",
    "__version__",
    "getLogger_pyscal",
]
