"""Command line tool for pyscal"""

import argparse
import sys
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd

from pyscal import (
    GasWater,
    SCALrecommendation,
    WaterOilGas,
    __version__,
    getLogger_pyscal,
)

from .factory import PyscalFactory

EPILOG = """
The parameter file should contain a table with at least the column
SATNUM, containing only consecutive integers starting at 1. Each row
provides the data for the corresponding SATNUM. Comments are put in a
column called TAG or COMMENT. Column headers are case insensitive.

Saturation endpoints are put in columns 'swirr', 'swl', 'swcr', 'sorw',
'sgcr' and 'sorg'. Relative permeability endpoints are put in columns
'krwend', 'krwmax', 'krowend', 'krogend', 'krgend' and 'krgmax'.
These columns are optional and are defaulted to 0 or 1.

Corey or LET parametrization are based on presence of the columns
'Nw', 'Now', 'Nog', 'Ng', 'Lw', 'Ew', 'Tw', 'Low', 'Eow', 'Tow',
'Log', 'Eog', 'Tog', 'Lg', 'Eg', 'Tg'.

Simple J-function for capillary pressure ("RMS" version) is used if the columns
'a', 'b', 'poro_ref', 'perm_ref' and 'drho' are found. If you provide
'a_petro', or 'b_petro', the petrophysical formulation of the simple J-function
is used. Check API for exact formulas. Normalized J-function is used if 'a',
'b', 'poro', 'perm' and 'sigma_costau' is provided.

For SCAL recommendations, there should be exactly three rows for each SATNUM,
tagged with the strings 'low', 'base' and 'high' in the column 'CASE'

When interpolating in a SCAL recommendation, 'int_param_wo' is the main parameter
that is used for water-oil, gas-oil and gas-water, and for all SATNUMs if nothing
more is provided. Provide int_param_go in addition if separate interpolation
for WaterOil and GasOil is needed, and specify multiple floats pr. parameter
if individual interpolation for each SATNUM is needed.
"""


def get_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for the command line script.

    Returns:
        argparse.Parser
    """
    parser = argparse.ArgumentParser(
        prog="pyscal",
        description=(
            "pyscal (" + __version__ + ") is a tool to create Eclipse include "
            "files for relative permeability input from tabulated parameters."
        ),
        epilog=EPILOG,
    )
    parser.add_argument(
        "parametertable",
        help=(
            "CSV or XLSX file with Corey or LET parameters for relperms. "
            "One SATNUM pr row."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print informational messages while processing input",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version " + __version__ + ")",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="relperm.inc",
        help="Name of Eclipse include file to produce",
    )
    parser.add_argument(
        "--delta_s",
        default=None,
        type=float,
        help="Saturation table step-length for sw/sg. Default 0.01",
    )
    parser.add_argument(
        "--int_param_wo",
        default=None,
        type=float,
        help=(
            "Interpolation parameter for water-oil, needed if the parametertable "
            "contains pess/low, base and opt/high for each SATNUM. "
            "The parameter will be used for all SATNUMs and must be "
            "between -1 and 1. Also used for GasWater."
        ),
    )
    parser.add_argument(
        "--int_param_go",
        default=None,
        type=float,
        help=(
            "Interpolation parameter for gas-oil. "
            "If not provided, the water-oil interpolation parameter will be used "
            "as default. Do not use for GasWater."
        ),
    )
    parser.add_argument(
        "--sheet_name",
        type=str,
        default=None,
        help="Sheet name if reading XLSX file. Defaults to first sheet",
    )
    parser.add_argument(
        "--slgof",
        action="store_true",
        default=False,
        help="If using family 1 keywords, use SLGOF instead of SGOF",
    )
    parser.add_argument(
        "--family2",
        action="store_true",
        default=False,
        help=(
            "Output family 2 keywords, SWFN, SGFN and SOF3/SOF2. "
            "Family 1 (SWOF + SGOF) is written if this is not set.  "
            "Implicit for gas-water input."
        ),
    )
    return parser


def main() -> None:
    """Endpoint for pyscals command line utility.

    Translates from argparse API to Pyscal's Python API"""
    parser = get_parser()
    args = parser.parse_args()

    try:
        pyscal_main(
            parametertable=args.parametertable,
            verbose=args.verbose,
            debug=args.debug,
            output=args.output,
            delta_s=args.delta_s,
            int_param_wo=args.int_param_wo,
            int_param_go=args.int_param_go,
            sheet_name=args.sheet_name,
            slgof=args.slgof,
            family2=args.family2,
        )
    except (OSError, ValueError) as err:
        print("".join(traceback.format_tb(err.__traceback__)))
        sys.exit(str(err))


def pyscal_main(
    parametertable: str,
    verbose: bool = False,
    debug: bool = False,
    output: str = "relperm.inc",
    delta_s: Optional[float] = None,
    int_param_wo: Optional[float] = None,
    int_param_go: Optional[float] = None,
    sheet_name: Optional[str] = None,
    slgof: bool = False,
    family2: bool = False,
) -> None:
    """A "main()" method not relying on argparse. This can be used
    for testing, and also by an ERT forward model, e.g.
    in semeio (github.com/equinor/semeio)

    Args:
        parametertable: Filename (CSV or XLSX) to load
        verbose: verbose or not
        debug: debug mode or not
        output: Output filename
        delta_s: Saturation step-length
        int_param_wo: Interpolation parameter for wateroil
        int_param_go: Interpolation parameter for gasoil
        sheet_name: Which sheet in XLSX file
        slgof: Use SLGOF
        family2: Dump family 2 keywords
    """

    logger = getLogger_pyscal(
        __name__, {"debug": debug, "verbose": verbose, "output": output}
    )

    parametertable = PyscalFactory.load_relperm_df(
        parametertable, sheet_name=sheet_name
    )

    assert isinstance(parametertable, pd.DataFrame)
    logger.debug("Input data:\n%s", parametertable.to_string(index=False))

    if int_param_go is not None and int_param_wo is None:
        raise ValueError("Don't use int_param_go alone, only int_param_wo")
    if isinstance(int_param_wo, list) or isinstance(int_param_go, list):
        raise TypeError(
            "SATNUM specific interpolation parameters are not supported in pyscalcli"
        )
    if int_param_wo is not None and "CASE" not in parametertable:
        raise ValueError(
            "Interpolation parameter provided but no CASE column in input data"
        )
    if "SATNUM" not in parametertable:
        raise ValueError("There is no column called SATNUM in the input data")

    if "CASE" in parametertable:
        # Then we should do interpolation
        if int_param_wo is None:
            raise ValueError("No interpolation parameters provided")
        scalrec_list = PyscalFactory.create_scal_recommendation_list(
            parametertable, h=delta_s
        )
        assert isinstance(scalrec_list[1], SCALrecommendation)
        if scalrec_list[1].type == WaterOilGas:
            logger.info(
                "Interpolating, wateroil=%s, gasoil=%s",
                str(int_param_wo),
                str(int_param_go),
            )
            wog_list = scalrec_list.interpolate(int_param_wo, int_param_go, h=delta_s)
        elif scalrec_list[1].type == GasWater:
            logger.info(
                "Interpolating, gaswater=%s",
                str(int_param_wo),
            )
            wog_list = scalrec_list.interpolate(int_param_wo, None, h=delta_s)
    else:
        wog_list = PyscalFactory.create_pyscal_list(
            parametertable, h=delta_s
        )  # can be both water-oil, water-oil-gas, or gas-water

    if family2 or wog_list.pyscaltype == GasWater:
        family = 2
    else:
        family = 1

    if output == "-":
        print(wog_list.build_eclipse_data(family=family, slgof=slgof))
    else:
        if not Path(output).parent.exists():
            logger.warning(
                "Implicit directory creation is deprecated.\n"
                "Please create the output directory prior to calling pyscal."
            )
            Path(output).parent.mkdir(exist_ok=True, parents=True)
        Path(output).write_text(
            wog_list.build_eclipse_data(family=family, slgof=slgof), encoding="utf-8"
        )
        print("Written to " + output)
