"""Command line tool for pyscal"""

import sys
import argparse

import logging

from pyscal import WaterOilGas, GasWater, PyscalFactory

from pyscal import __version__

logging.basicConfig()
logger = logging.getLogger(__name__)

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

Simple J-function for capillary pressure is used if the columns
'a', 'b', 'poro_ref', 'perm_ref' and 'drho' are found. If you provide
'a_petro', or 'b_petro', the petrophysical formulation of the simple
J-function is used. Check API for exact formulas. Normalized J-function
is used if 'a', 'b', 'poro', 'perm' and 'sigma_costau' is provided.

For SCAL recommendations, there should be exactly three rows for each SATNUM,
tagged with the strings 'low', 'base' and 'high' in the column 'CASE'

When interpolating in a SCAL recommendation, 'int_param_wo' is the main parameter
that is used for water-oil, gas-oil and gas-water, and for all SATNUMs if nothing
more is provided. Provide int_param_go in addition if separate interpolation
for WaterOil and GasOil is needed, and specify multiple floats pr. parameter
if individual interpolation for each SATNUM is needed.
"""


def get_parser():
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
        "--debug", action="store_true", help="Print debug information",
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
        nargs="+",
        default=None,
        type=float,
        help=(
            "Interpolation parameters for water-oil, if the parametertable contains "
            "low, base and high for each SATNUM. "
            "Either one number used for all SATNUM, or a sequence of "
            "length equal to the max SATNUM. Numbers between -1 and 1. "
            "Also used for GasWater."
        ),
    )
    parser.add_argument(
        "--int_param_go",
        nargs="+",
        default=None,
        type=float,
        help=(
            "Interpolation parameters for gas-oil, if the parametertable contains "
            "low, base and high for each SATNUM. "
            "Either one number used for all SATNUM, or a sequence "
            "of length equal to the max SATNUM. Numbers between -1 and 1. "
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


def main():
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
    except ValueError:
        # If ValueErrors, error messages have already been printed
        sys.exit(1)


def pyscal_main(
    parametertable,
    verbose=False,
    debug=False,
    output="relperm.inc",
    delta_s=None,
    int_param_wo=None,
    int_param_go=None,
    sheet_name=None,
    slgof=False,
    family2=False,
):
    """A "main()" method not relying on argparse. This can be used
    for testing, and also by an ERT forward model, e.g.
    in semeio (github.com/equinor/semeio)

    Args:
        parametertable (string): Filename (CSV or XLSX) to load
        verbose (bool): verbose or not
        output (string): Output filename
        delta_s (float): Saturation step-length
        int_param_wo (list): Interpolation params for wateroil
        int_param_go (list): Interpolation params for gasoil
        sheet_name (string): Which sheet in XLSX file
        slgof (bool): Use SLGOF
        family2 (bool): Dump family 2 keywords
    """

    def set_logger_levels(loglevel):
        """Set log levels for all modules imported by this script"""
        logger.setLevel(loglevel)
        logging.getLogger("pyscal.factory").setLevel(loglevel)
        logging.getLogger("pyscal.wateroil").setLevel(loglevel)
        logging.getLogger("pyscal.wateroilgas").setLevel(loglevel)
        logging.getLogger("pyscal.gasoil").setLevel(loglevel)
        logging.getLogger("pyscal.utils").setLevel(loglevel)
        logging.getLogger("pyscal.pyscallist").setLevel(loglevel)

    if verbose:
        set_logger_levels(logging.INFO)
    if debug:
        set_logger_levels(logging.DEBUG)

    scalinput_df = PyscalFactory.load_relperm_df(parametertable, sheet_name=sheet_name)

    logger.debug("Input data:\n%s", scalinput_df.to_string(index=False))

    if int_param_go is not None and int_param_wo is None:
        logger.error("Don't use int_param_go alone, only int_param_wo")
        raise ValueError
    if "SATNUM" not in scalinput_df:
        logger.error("There is no column called SATNUM in the input data")
        raise ValueError
    if "CASE" in scalinput_df:
        # Then we should do interpolation
        if int_param_wo is None:
            logger.error("No interpolation parameters provided")
            raise ValueError
        scalrec_list = PyscalFactory.create_scal_recommendation_list(
            scalinput_df, h=delta_s
        )
        if scalrec_list[1].type == WaterOilGas:
            logger.info(
                "Interpolating, wateroil=%s, gasoil=%s",
                str(int_param_wo),
                str(int_param_go),
            )
            wog_list = scalrec_list.interpolate(int_param_wo, int_param_go, h=delta_s)
        elif scalrec_list[1].type == GasWater:
            logger.info(
                "Interpolating, gaswater=%s", str(int_param_wo),
            )
            wog_list = scalrec_list.interpolate(int_param_wo, None, h=delta_s)
    else:
        wog_list = PyscalFactory.create_pyscal_list(
            scalinput_df, h=delta_s
        )  # can be both water-oil, water-oil-gas, or gas-water

    if (
        int_param_wo is not None or int_param_go is not None
    ) and "CASE" not in scalinput_df:
        logger.error(
            "Interpolation parameter provided but no CASE column in input data"
        )
        raise ValueError

    if family2 or wog_list.pyscaltype == GasWater:
        logger.info("Generating family 2 keywords.")
        if output == "-":
            print(wog_list.dump_family_2())
        else:
            wog_list.dump_family_2(filename=output)
            print("Written to " + output)
    else:
        logger.info("Generating family 1 keywords.")
        if output == "-":
            print(wog_list.dump_family_1(slgof=slgof))
        else:
            wog_list.dump_family_1(filename=output, slgof=slgof)
            print("Written to " + output)


if __name__ == "__main__":
    main()
