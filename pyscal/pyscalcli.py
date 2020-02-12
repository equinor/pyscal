"""Command line tool for pyscal"""

import sys
import argparse

import logging

from pyscal import PyscalFactory

logging.basicConfig()
logger = logging.getLogger(__name__)

EPILOG = """
The parameter file should contain a table with at least the column
SATNUM, containing only consecutive integers starting at 1. Each row
provides the data for the corresponding SATNUM. Comments are put in a
column called TAG or COMMENT. Column headers are case insensitive.

Saturation endpoints are put in columns 'swirr', 'swl', 'swcr', 'sorw',
'sgcr' and 'sorg'. Relative permeability endpoints are put in columns
'krwend', 'krwmax', 'krowend', 'krogend', 'kromax', 'krgend' and 'krgmax'.
These columns are optional and are defaulted to 0 or 1.

Corey or LET parametrization are based on presence of the columns
'Nw', 'Now', 'Nog', 'Ng', 'Lw', 'Ew', 'Tw', 'Low', 'Eow', 'Tow',
'Log', 'Eog', 'Tog', 'Lg', 'Eg', 'Tg'.

Simple J-function for capillary pressure is used if the columns
'a', 'b', 'poro_ref', 'perm_ref' and 'drho' are found.

For SCAL recommendations, there should be exactly three rows for each SATNUM,
tagged with the strings 'low', 'base' and 'high' in the column 'CASE'

When interpolating in a SCAL recommendation, 'int_param_wo' is the main parameter
that is used for both water-oil and gas-oil, and for all SATNUMs if nothing
more is provided. Provide int_param_go in addition if separate interpolation
for WaterOil and GasOil is needed, and specify multiple floats pr. parameter
if individual interpolation for each SATNUM is needed.
"""


def get_parser():
    """Construct the argparse parser for the command line script.

    Returns:
        argparse.Parser
    """
    parser = argparse.ArgumentParser(prog="pyscal", epilog=EPILOG)
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
            "Either one number used for all SATMUM, or a sequence of "
            "length equal to the max SATNUM. Numbers between -1 and 1"
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
            "Either one number used for all SATMUM, or a sequence "
            "of length equal to the max SATNUM. Numbers between -1 and 1"
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
            "Family 1 (SWOF + SGOF) is written if this is not set"
        ),
    )
    return parser


def main():
    """Endpoint for pyscals command line utility"""
    parser = get_parser()
    args = parser.parse_args()

    if args.verbose:
        # Fixme: Logging level is not inherited in called modules.
        logger.setLevel(logging.INFO)

    if args.sheet_name:
        logger.info(
            "Loading data from %s and sheetname %s",
            args.parametertable,
            args.sheet_name,
        )
    else:
        logger.info("Loading data from %s", args.parametertable)
    try:
        scalinput_df = PyscalFactory.load_relperm_df(
            args.parametertable, sheet_name=args.sheet_name
        )
    except ValueError:
        # If ValueErrors are raised by pyscal-code, error messages
        # have already been printed. Lets just exit with error code.
        sys.exit(1)

    logger.info("Input data:\n%s", scalinput_df.to_string(index=False))

    if args.int_param_go is not None and args.int_param_wo is None:
        logger.error("Don't use int_param_go alone, only int_param_wo")
        sys.exit(1)
    if "SATNUM" not in scalinput_df:
        logger.error("There is no column called SATNUM in the input data")
        sys.exit(1)
    if "CASE" in scalinput_df:
        # Then we should do interpolation
        if args.int_param_wo is None:
            logger.error("No interpolation parameters provided")
            sys.exit(1)
        try:
            scalrec_list = PyscalFactory.create_scal_recommendation_list(
                scalinput_df, h=args.delta_s
            )
        except ValueError:
            sys.exit(1)
        logger.info(
            "Interpolating, wateroil=%s, gasoil=%s",
            str(args.int_param_wo),
            str(args.int_param_go),
        )
        try:
            wog_list = scalrec_list.interpolate(
                args.int_param_wo, args.int_param_go, h=args.delta_s
            )
        except ValueError:
            sys.exit(1)
    else:
        try:
            wog_list = PyscalFactory.create_pyscal_list(
                scalinput_df, h=args.delta_s
            )  # can be both water-oil, water-oil-gas, or gas-water
        except ValueError:
            sys.exit(1)

    if (
        args.int_param_wo is not None or args.int_param_go is not None
    ) and "CASE" not in scalinput_df:
        logger.error(
            "Interpolation parameter provided but no CASE column in input data"
        )
        sys.exit(1)
    if not args.family2:
        logger.info("Generating family 1 keywords.")
        if args.output == "-":
            print(wog_list.dump_family_1(slgof=args.slgof))
        else:
            wog_list.dump_family_1(filename=args.output, slgof=args.slgof)
            print("Written to " + args.output)
    else:
        logger.info("Generating family 2 keywords")
        if args.output == "-":
            print(wog_list.dump_family_2())
        else:
            wog_list.dump_family_2(filename=args.output)
            print("Written to " + args.output)


if __name__ == "__main__":
    main()
