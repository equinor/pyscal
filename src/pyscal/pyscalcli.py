"""Command line tool for pyscal"""

import argparse
import io
import sys
import traceback
from pathlib import Path

import pandas as pd

from pyscal import (
    GasWater,
    SCALrecommendation,
    WaterOilGas,
    __version__,
    getLogger_pyscal,
    plotting,
)
from pyscal.factory import (
    create_pyscal_list,
    create_scal_recommendation_list,
    load_relperm_df,
)
from pyscal.gasoil import GasOil
from pyscal.pyscallist import PyscalList
from pyscal.wateroil import WaterOil

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

Primary drainage and imbibition capillary pressure with LET parametrization are
based on presence of the columnns 'Lp', 'Ep', 'Tp','Lt', 'Et', 'Tt', 'Pcmax', 'Pct' and
'Ls', 'Es', 'Ts', 'Lf', 'Ef', 'Tf', 'Pcmax', 'Pct', 'Pcmin' respectively.

Capillary pressure from the Skjæveland correlation is based on the on the presence
of the columns 'cw', 'co', 'aw', 'ao'.

For SCAL recommendations, there should be exactly three rows for each SATNUM,
tagged with the strings 'low', 'base' and 'high' in the column 'CASE'

When interpolating in a SCAL recommendation, 'int_param_wo' is the main parameter
that is used for water-oil, gas-oil and gas-water, and for all SATNUMs if nothing
more is provided. Provide int_param_go in addition if separate interpolation
for WaterOil and GasOil is needed.
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
        "--export_params",
        default="pyscal-params.csv",
        help="Export curve parameters to a CSV file (default: %(default)s)",
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

    # Plotting arguments
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help=("Make and save relative permeability figures."),
    )
    parser.add_argument(
        "--plot_pc",
        action="store_true",
        default=False,
        help=("Make and save capillary pressure figures."),
    )
    parser.add_argument(
        "--plot_semilog",
        action="store_true",
        default=False,
        help=(
            "Plot relative permeability figures with log y-axis."
            "Run both with and without this flag to plot"
            "both linear and semi-log relperm plots."
        ),
    )
    parser.add_argument(
        "--plot_outdir",
        default="./",
        help="Directory where the plot output figures will be saved.",
    )

    return parser


def main() -> None:
    """Endpoint for pyscals command line utility.

    Translates from argparse API to Pyscal's Python API"""
    parser = get_parser()
    args = parser.parse_args()

    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="UTF-8")
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="UTF-8")

    try:
        pyscal_main(
            parametertable=args.parametertable,
            verbose=args.verbose,
            debug=args.debug,
            output=args.output,
            export_params=args.export_params,
            delta_s=args.delta_s,
            int_param_wo=args.int_param_wo,
            int_param_go=args.int_param_go,
            sheet_name=args.sheet_name,
            slgof=args.slgof,
            family2=args.family2,
            plot=args.plot,
            plot_pc=args.plot_pc,
            plot_semilog=args.plot_semilog,
            plot_outdir=args.plot_outdir,
        )
    except (OSError, ValueError) as err:
        print("".join(traceback.format_tb(err.__traceback__)))
        sys.exit(str(err))


def pyscal_main(
    parametertable: str,
    verbose: bool = False,
    debug: bool = False,
    output: str = "relperm.inc",
    delta_s: float | None = None,
    int_param_wo: float | None = None,
    int_param_go: float | None = None,
    sheet_name: str | None = None,
    slgof: bool = False,
    family2: bool = False,
    plot: bool = False,
    plot_pc: bool = False,
    plot_semilog: bool = False,
    plot_outdir: str = "./",
    *,
    export_params: str = "pyscal-params.csv",
) -> None:
    """A "main()" method not relying on argparse. This can be used
    for testing, and also by an ERT forward model, e.g.
    in semeio (github.com/equinor/semeio)

    Args:
        parametertable: Filename (CSV or XLSX) to load
        verbose: verbose or not
        debug: debug mode or not
        output: Output filename
        export_params: CSV filename for curve parameters
        delta_s: Saturation step-length
        int_param_wo: Interpolation parameter for wateroil
        int_param_go: Interpolation parameter for gasoil
        sheet_name: Which sheet in XLSX file
        slgof: Use SLGOF
        family2: Dump family 2 keywords
        plot: Plot relative permeability figures and save
        plot_pc: Plot capillary pressure curves in addition to relperm curves
        plot_semilog: Plot relative permeability figures with log y-axis

    """

    logger = getLogger_pyscal(
        __name__, {"debug": debug, "verbose": verbose, "output": output}
    )

    parametertable_df = load_relperm_df(parametertable, sheet_name=sheet_name)

    assert isinstance(parametertable_df, pd.DataFrame)
    logger.debug("Input data:\n%s", parametertable_df.to_string(index=False))

    if int_param_go is not None and int_param_wo is None:
        raise ValueError("Don't use int_param_go alone, only int_param_wo")
    if isinstance(int_param_wo, list) or isinstance(int_param_go, list):
        raise TypeError(
            "SATNUM specific interpolation parameters are not supported in pyscalcli"
        )
    if int_param_wo is not None and "CASE" not in parametertable_df:
        raise ValueError(
            "Interpolation parameter provided but no CASE column in input data"
        )
    if "SATNUM" not in parametertable_df:
        raise ValueError("There is no column called SATNUM in the input data")

    if "CASE" in parametertable_df:
        # Then we should do interpolation
        if int_param_wo is None:
            raise ValueError("No interpolation parameters provided")
        scalrec_list = create_scal_recommendation_list(parametertable_df, h=delta_s)
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
        wog_list = create_pyscal_list(
            parametertable_df, h=delta_s
        )  # can be both water-oil, water-oil-gas, or gas-water

    family = 2 if family2 or wog_list.pyscaltype == GasWater else 1

    export_path = Path(export_params)
    if not export_path.parent.exists():
        raise OSError(f"Output directory not found '{export_path.parent}'")
    protected_paths = [Path(parametertable)]
    if output != "-":
        protected_paths.append(Path(output))
    for protected_path in protected_paths:
        same_path = export_path.resolve() == protected_path.resolve()
        same_file = (
            export_path.exists()
            and protected_path.exists()
            and export_path.samefile(protected_path)
        )
        if same_path or same_file:
            raise ValueError(
                "Parameter export path must differ from input and output paths"
            )

    if output == "-":
        print(wog_list.build_eclipse_data(family=family, slgof=slgof))
    else:
        if not Path(output).parent.exists():
            raise OSError(f"Output directory not found '{Path(output).parent}'")
        with open(output, "w", newline="\n", encoding="utf-8") as fh:
            fh.write(wog_list.build_eclipse_data(family=family, slgof=slgof))
        print("Written to " + output)

    export_int_param_go = int_param_go
    if wog_list.pyscaltype == GasWater:
        export_int_param_go = None
    elif wog_list.pyscaltype == WaterOilGas and export_int_param_go is None:
        export_int_param_go = int_param_wo
    _params_df(wog_list, int_param_wo, export_int_param_go).to_csv(
        export_path, index=False
    )
    print(
        "Written parameters to " + str(export_path),
        file=sys.stderr if output == "-" else sys.stdout,
    )

    if plot:
        plotting.plotter(wog_list, plot_pc, plot_semilog, plot_outdir)


def _params_df(
    wog_list: PyscalList,
    int_param_wo: float | None = None,
    int_param_go: float | None = None,
) -> pd.DataFrame:
    """Return structured header parameters in one row per SATNUM."""
    rows = []

    def without_interpolation_prefix(tag: str | None) -> str:
        if tag is None:
            return ""
        if tag.startswith("SCAL recommendation interpolation to "):
            return tag.partition("\n")[2]
        return tag

    for satnum, obj in enumerate(wog_list.pyscal_list, start=1):
        if (
            isinstance(obj, WaterOilGas)
            and obj.wateroil is not None
            and obj.gasoil is not None
        ):
            wateroil_tag = without_interpolation_prefix(obj.wateroil.tag)
            gasoil_tag = without_interpolation_prefix(obj.gasoil.tag)
            tag = (
                wateroil_tag
                if wateroil_tag == gasoil_tag
                else wateroil_tag + " " + gasoil_tag
            )
        else:
            tag = without_interpolation_prefix(obj.tag)

        gas_water = isinstance(obj, GasWater)
        if isinstance(obj, (GasWater, WaterOilGas)):
            wateroil, gasoil = obj.wateroil, obj.gasoil
        elif isinstance(obj, WaterOil):
            wateroil, gasoil = obj, None
        elif isinstance(obj, GasOil):
            wateroil, gasoil = None, obj
        else:
            raise TypeError(f"Unexpected pyscal object: {type(obj)}")

        row: dict[str, object] = {
            "SATNUM": satnum,
            "TAG": tag,
            "int_param_wo": int_param_wo,
            "int_param_go": int_param_go if gasoil is not None else None,
        }
        if wateroil is not None:
            row.update(
                swirr=wateroil.swirr,
                swl=wateroil.swl,
                swcr=wateroil.swcr,
                **wateroil.krw_params,
                **{f"pcow_{key}": value for key, value in wateroil.pc_params.items()},
            )
            row["sgrw" if gas_water else "sorw"] = wateroil.sorw
            if not gas_water:
                row.update(wateroil.krow_params)
                if wateroil.socr > wateroil.sorw:
                    row["socr"] = wateroil.socr
            if (
                not obj.fast
                and {"KRW", "KROW"}.issubset(wateroil.table.columns)
                and not gas_water
            ):
                row["krw_krow_crosspoint"] = round(wateroil.crosspoint(), 5)
        if gasoil is not None:
            row.update(
                sgcr=gasoil.sgcr,
                sgro=gasoil.sgro,
                **gasoil.krg_params,
                **{f"pcog_{key}": value for key, value in gasoil.pc_params.items()},
            )
            if wateroil is None:
                row.update(swirr=gasoil.swirr, swl=gasoil.swl)
            if not gas_water:
                row.update(
                    sorg=gasoil.sorg,
                    krgendanchor=gasoil.krgendanchor,
                    **gasoil.krog_params,
                )
            if (
                not obj.fast
                and {"KRG", "KROG"}.issubset(gasoil.table.columns)
                and not gas_water
            ):
                row["krg_krog_crosspoint"] = round(gasoil.crosspoint(), 5)
        if (
            gas_water
            and isinstance(obj, GasWater)
            and wateroil is not None
            and gasoil is not None
        ):
            row["sgcr"] = gasoil.sgcr
            row["krgmax"] = gasoil.krg_params.get("krgmax")
            if (
                not obj.fast
                and {"KRW"}.issubset(wateroil.table)
                and {"KRG"}.issubset(gasoil.table)
            ):
                crosspoint = obj.crosspoint()
                if crosspoint is not None:
                    row["krw_krg_crosspoint"] = round(crosspoint, 5)
        if (
            wateroil is not None
            and gasoil is not None
            and (wateroil.swirr != gasoil.swirr or wateroil.swl != gasoil.swl)
        ):
            row.pop("swirr", None)
            row.pop("swl", None)
            row.update(
                swirr_wo=wateroil.swirr,
                swl_wo=wateroil.swl,
                swirr_go=gasoil.swirr,
                swl_go=gasoil.swl,
            )
        rows.append(row)

    return pd.DataFrame(rows)
