"""Utility functions for creating strings from pyscal"""

import logging
from typing import Dict, Optional

import pandas as pd

from .monotonicity import MonotonicitySpec, modify_dframe_monotonicity

logger = logging.getLogger(__name__)


def df2str(
    dframe: pd.DataFrame,
    digits: int = 7,
    roundlevel: int = 9,
    header: bool = False,
    monotonicity: Optional[Dict[str, MonotonicitySpec]] = None,
) -> str:
    """
    Make a string representation of a dataframe with
    proper rounding.

    This is used to print the tables in the SWOF/SGOF include files,
    explicit rounding is necessary to avoid monotonicity errors
    from truncation. Examples in test code.

    Capillary pressure must be strictly monotone if nonzero, and if a
    column name is provided, the string representation of that column is
    ensured to be strictly monotone decreasing

    Args:
        dframe: A dataframe to print, all columns
            are included
        digits: Number of digits used in floating point format f.ex ".7f"
            It is not recommended to deviate from the default 7 uncritically
            for pyscal output, other code have to be tuned to ensure
            numerical robustness to the deviation.
        roundlevel: To how many digits should we round prior to print.
            Recommended to be > digits + 1, see test code.
        header: If the dataframe column header should be included
        monotonicity: Column names in dframe are the keys, pointing
            to a specification for monotonicity to be enforced.
    """
    float_format = "%1." + str(digits) + "f"

    if monotonicity is not None:
        dframe = modify_dframe_monotonicity(dframe, monotonicity, digits)

    return dframe.round(roundlevel).to_csv(
        sep=" ", float_format=float_format, header=header, index=False
    )


def comment_formatter(multiline: str, prefix: str = "-- "):
    """Prepends comment characters to every line in input

    Args:
        multiline: String that can contain newlines
        prefix: Comment characters to prepend every line with
            Default is the Eclipse comment syntax '-- '

    Returns:
        string with newlines preserved, and where each line
        starts with the given prefix. Always ends with a newline.
    """
    if multiline is None or not multiline.strip():
        # Ensure we indicate that there is placeholder for something.
        return "-- \n"
    return (
        "\n".join([prefix + line.strip() for line in multiline.splitlines()]).strip()
        + "\n"
    )
