"""pyscal"""


# The order of imports must be conserved to avoid circular imports:
from .wateroil import WaterOil  # noqa
from .wateroilgas import WaterOilGas  # noqa
from .gasoil import GasOil  # noqa
from .gaswater import GasWater  # noqa
from .scalrecommendation import SCALrecommendation  # noqa
from .pyscallist import PyscalList  # noqa
from .factory import PyscalFactory  # noqa

try:
    from .version import version

    __version__ = version
except ImportError:
    __version__ = "0.0.0"
