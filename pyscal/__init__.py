"""pyscal"""

import logging
import sys
from typing import Dict, Optional, Union

try:
    from .version import version

    __version__ = version
except ImportError:
    __version__ = "0.0.0"


def getLogger_pyscal(
    module_name: str = "pyscal", args_dict: Optional[Dict[str, Union[str, bool]]] = None
) -> logging.Logger:
    # pylint: disable=invalid-name
    """Provide a custom logger for pyscal
    Logging output is by default split by logging levels (split between WARNING and
    ERROR) to stdout and stderr, each log occurs in only one of the streams.
    Args:
        module_name: A suggested name for the logger, usually __name__ should be supplied
        args_dict: Dictionary with contents from the argparse namespace object containing
        only keys "output", "verbose" and "debug".
    """
    logger = logging.getLogger(module_name)
    if len(logger.handlers) != 0:
        return logger

    if args_dict is None:
        args_dict = {}
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    if args_dict.get("output", None) == "-":
        # If main output is to stdout, we must send all logs to stderr:
        default_handler = logging.StreamHandler(sys.stderr)
        default_handler.setFormatter(formatter)
        logger.addHandler(default_handler)
    else:
        # Split log messages to either stdout or stderr based on log level:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.addFilter(lambda record: record.levelno < logging.ERROR)
        stdout_handler.setFormatter(formatter)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.addFilter(lambda record: record.levelno >= logging.ERROR)
        stderr_handler.setFormatter(formatter)

        logger.addHandler(stdout_handler)
        logger.addHandler(stderr_handler)

    # --debug overrides --verbose
    if args_dict.get("debug", False):
        logger.setLevel(logging.DEBUG)
    elif args_dict.get("verbose", False):
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    if module_name == "pyscal.pyscalcli":
        all_modules = [
            "factory",
            "gasoil",
            "gaswater",
            "pyscallist",
            "scalrecommendation",
            "wateroil",
            "wateroilgas",
        ]
        for module in all_modules:
            module_logger = logging.getLogger("pyscal." + module)
            module_logger.handlers = []
            for handler in logger.handlers:
                module_logger.addHandler(handler)
            module_logger.setLevel(logger.level)

    return logger


# The order of imports must be conserved to avoid circular imports:
from .wateroil import WaterOil  # noqa
from .wateroilgas import WaterOilGas  # noqa
from .gasoil import GasOil  # noqa
from .gaswater import GasWater  # noqa
from .scalrecommendation import SCALrecommendation  # noqa
from .pyscallist import PyscalList  # noqa
from .factory import PyscalFactory  # noqa
