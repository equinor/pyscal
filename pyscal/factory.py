"""Factory functions for creating the pyscal objects"""

import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union

import numpy as np
import openpyxl
import pandas as pd
import xlrd

from pyscal import getLogger_pyscal
from pyscal.utils import capillarypressure

from .gasoil import GasOil
from .gaswater import GasWater
from .pyscallist import PyscalList
from .scalrecommendation import SCALrecommendation
from .wateroil import WaterOil
from .wateroilgas import WaterOilGas

logger = getLogger_pyscal(__name__)


def slicedict(dct: dict, keys: Iterable):
    """Slice a dictionary for a set of keys.
    Keys not existing will be ignored.
    """
    return dict((k, dct[k]) for k in keys if k in dct)


# Sets of dictionary keys, presence of these
# in incoming dictionary determines the codepaths. These must
# again match the API of the functions downstream in e.g. WaterOil, except
# for LET parameters, where the API is simplified to 'l', 'e' and 't'.
# We are case *insensitive* in this factory class, so everything here should
# be lower case
WO_INIT = ["swirr", "swl", "swcr", "sorw", "socr", "sgrw", "h", "tag"]
WO_COREY_WATER = ["nw"]
WO_WATER_ENDPOINTS = ["krwmax", "krwend"]
WO_COREY_OIL = ["now"]
WO_LET_WATER = ["lw", "ew", "tw"]  # Will translated to l, e and t in code below.
WO_LET_OIL = ["low", "eow", "tow"]
WO_LET_OIL_ALT = ["lo", "eo", "to"]  # Alternative parameter names.
WO_OIL_ENDPOINTS = ["kroend"]
WO_SIMPLE_J = ["a", "b", "poro_ref", "perm_ref", "drho"]  # "g" is optional
WO_SWLHEIGHT = ["swlheight"]
WO_SWCR_ADD = ["swcr_add"]  # Relevant when swlheight is in use.
WO_SWL_FROM_HEIGHT = WO_SWLHEIGHT + ["swirr", "a", "b", "poro_ref", "perm_ref"]
WO_NORM_J = ["a", "b", "poro", "perm", "sigma_costau"]
# 'a' in WO_NORM_J is the same as a_petro, but should possibly kept as is.
WO_SIMPLE_J_PETRO = [
    "a_petro",
    "b_petro",
    "poro_ref",
    "perm_ref",
    "drho",
]  # "g" is optional

GO_INIT = ["swirr", "sgcr", "sorg", "sgro", "swl", "krgendanchor", "h", "tag"]
GO_COREY_GAS = ["ng"]
GO_GAS_ENDPOINTS = ["krgend", "krgmax"]
GO_COREY_OIL = ["nog"]
GO_OIL_ENDPOINTS = [
    "kroend",
    "kromax",
]

GO_LET_GAS = ["lg", "eg", "tg"]
GO_LET_OIL = ["log", "eog", "tog"]

GW_INIT = ["swirr", "swl", "sgl", "swcr", "sgrw", "sgcr", "h", "tag"]
GW_COREY_WATER = ["nw"]
GW_COREY_GAS = ["ng"]
GW_WATER_ENDPOINTS = ["krwmax", "krwend"]
GW_LET_WATER = ["lw", "ew", "tw"]
GW_LET_GAS = ["lg", "eg", "tg"]
GW_GAS_ENDPOINTS = ["krgmax", "krgend"]
GW_SIMPLE_J = ["a", "b", "poro_ref", "perm_ref", "drho"]  # "g" is optional
GW_NORM_J = ["a", "b", "poro", "perm", "sigma_costau"]
# 'a' in WO_NORM_J is the same as a_petro, but should possibly kept as is.
GW_SIMPLE_J_PETRO = [
    "a_petro",
    "b_petro",
    "poro_ref",
    "perm_ref",
    "drho",
]  # "g" is optional

WOG_INIT = ["swirr", "swl", "swcr", "sorw", "socr", "sorg", "sgcr", "h", "tag"]


class PyscalFactory(object):
    """Class for implementing the factory pattern for Pyscal objects

    The factory functions herein can take multiple parameter sets,
    determine what kind of parametrization to be used, and set up
    the full objects based on these parameters, instead of
    explicitly having to call the API for each task.

    Example::

      wo = WaterOil(sorw=0.05)
      wo.add_corey_water(nw=3)
      wo.add_corey_oil(now=2)
      # is equivalent to:
      wo = factory.create_water_oil(dict(sorw=0.05, nw=3, now=2))

    Parameter names to factory functions are case *insensitive*, while
    the add_*() parameters are not. This is because the add_*() parameters
    are meant as a Python API, while the factory class is there to aid
    users when input is written in a different context, like an Excel
    spreadsheet.
    """

    @staticmethod
    def create_water_oil(
        params: Optional[Dict[str, float]] = None, fast: bool = False
    ) -> WaterOil:
        """Create a WaterOil object from a dictionary of parameters.

        Parameterization (Corey/LET) is inferred from presence
        of certain parameters in the dictionary.

        Don't rely on behaviour of you supply both Corey and LET at
        the same time.

        Parameter names in the dictionary are case insensitive. You
        can use Swirr, swirr, sWirR, swiRR etc.

        NB: the add_LET_* methods have the names 'l', 'e' and 't'
        in their signatures, which is not precise enough in this
        context, so we require e.g. 'Lw' and 'Low' (which both will be
        translated to 'l')

        Recognized parameters:
          swirr, swl, swcr, sorw, socr, sgrw, h, tag, nw, now, krwmax, krwend,
          lw, ew, tw, low, eow, tow, lo, eo, to, kroend,
          a, a_petro, b, b_petro, poro_ref, perm_ref, drho,
          a, b, poro, perm, sigma_costau

        Args:
            params: Dictionary with parameters describing the WaterOil object.
            fast: If fast-mode should be set for constructed object.
        """
        if not params:
            params = {}
        if not isinstance(params, dict):
            raise TypeError("Parameter to create_water_oil must be a dictionary")

        check_deprecated(params)

        sufficient_water_oil_params(params, failhard=True)

        # For case insensitiveness, all keys are converted to lower case:
        params = {key.lower(): value for (key, value) in params.items()}

        # Allowing sending in NaN values, delete those keys.
        params = filter_nan_from_dict(params)

        usedparams: Set[str] = set()

        # Check if we should initialize swl from a swlheight parameter:
        if set(WO_SWL_FROM_HEIGHT).issubset(params):
            if "swl" in params:
                raise ValueError(
                    "Do not provide both swl and swlheight at the same time"
                )
            if params["swlheight"] <= 0:
                raise ValueError("swlheight must be larger than zero")
            params_swl_from_height = slicedict(params, WO_SWL_FROM_HEIGHT)
            params["swl"] = capillarypressure.swl_from_height_simpleJ(
                **params_swl_from_height
            )
            logger.debug("Computed swl from swlwheight to %s", str(params["swl"]))
            if "swcr" in params and params["swcr"] < params["swl"]:
                raise ValueError(
                    f'Provided swcr={params["swcr"]} is lower than '
                    f'computed swl={params["swl"]}'
                )
        elif set(WO_SWLHEIGHT).issubset(params):
            raise ValueError(
                (
                    "Can't initialize from SWLHEIGHT without sufficient "
                    f"simple-J parameters, needs all of: {WO_SWL_FROM_HEIGHT}"
                )
            )

        # Should we have a swcr relative to swl?
        if set(WO_SWCR_ADD).issubset(params):
            if "swl" not in params:
                raise ValueError(
                    (
                        "If swcr should be relative to swl, "
                        "both swcr_add and swl must be provided"
                    )
                )
            if "swcr" in params:
                raise ValueError(
                    "Do not provide both swcr and swcr_add at the same time"
                )
            params["swcr"] = params["swl"] + params[WO_SWCR_ADD[0]]

        # No requirements to the base objects, defaults are ok.
        wateroil = WaterOil(
            **PyscalFactory.alias_sgrw(slicedict(params, WO_INIT)), fast=fast
        )
        usedparams = usedparams.union(set(slicedict(params, WO_INIT).keys()))
        logger.debug(
            "Initialized WaterOil object from parameters %s", str(list(usedparams))
        )

        # Water curve
        params_corey_water = slicedict(params, WO_COREY_WATER + WO_WATER_ENDPOINTS)
        params_let_water = slicedict(params, WO_LET_WATER + WO_WATER_ENDPOINTS)
        if set(WO_COREY_WATER).issubset(set(params_corey_water)):
            wateroil.add_corey_water(**params_corey_water)
            logger.debug(
                "Added Corey water to WaterOil object from parameters %s",
                str(params_corey_water.keys()),
            )
        elif set(WO_LET_WATER).issubset(set(params_let_water)):
            params_let_water["l"] = params_let_water.pop("lw")
            params_let_water["e"] = params_let_water.pop("ew")
            params_let_water["t"] = params_let_water.pop("tw")
            wateroil.add_LET_water(**params_let_water)
            logger.debug(
                "Added LET water to WaterOil object from parameters %s",
                str(params_let_water.keys()),
            )

        # Oil curve:
        params_corey_oil = slicedict(params, WO_COREY_OIL + WO_OIL_ENDPOINTS)
        params_let_oil = slicedict(
            params, WO_LET_OIL + WO_LET_OIL_ALT + WO_OIL_ENDPOINTS
        )
        if set(WO_COREY_OIL).issubset(set(params_corey_oil)):
            wateroil.add_corey_oil(**params_corey_oil)
            logger.debug(
                "Added Corey water to WaterOil object from parameters %s",
                str(params_corey_oil.keys()),
            )
        elif set(WO_LET_OIL).issubset(set(params_let_oil)):
            params_let_oil["l"] = params_let_oil.pop("low")
            params_let_oil["e"] = params_let_oil.pop("eow")
            params_let_oil["t"] = params_let_oil.pop("tow")
            wateroil.add_LET_oil(**params_let_oil)
            logger.debug(
                "Added LET water to WaterOil object from parameters %s",
                str(params_let_oil.keys()),
            )
        elif set(WO_LET_OIL_ALT).issubset(set(params_let_oil)):
            params_let_oil["l"] = params_let_oil.pop("lo")
            params_let_oil["e"] = params_let_oil.pop("eo")
            params_let_oil["t"] = params_let_oil.pop("to")
            wateroil.add_LET_oil(**params_let_oil)
            logger.debug(
                "Added LET water to WaterOil object from parameters %s",
                str(params_let_oil.keys()),
            )

        # Capillary pressure:
        params_simple_j = slicedict(params, WO_SIMPLE_J + ["g"])
        params_norm_j = slicedict(params, WO_NORM_J)
        params_simple_j_petro = slicedict(params, WO_SIMPLE_J_PETRO + ["g"])
        if set(WO_SIMPLE_J).issubset(set(params_simple_j)):
            wateroil.add_simple_J(**params_simple_j)
        elif set(WO_SIMPLE_J_PETRO).issubset(set(params_simple_j_petro)):
            params_simple_j_petro["a"] = params_simple_j_petro.pop("a_petro")
            params_simple_j_petro["b"] = params_simple_j_petro.pop("b_petro")
            wateroil.add_simple_J_petro(**params_simple_j_petro)
        elif set(WO_NORM_J).issubset(set(params_norm_j)):
            wateroil.add_normalized_J(**params_norm_j)
        else:
            logger.info(
                (
                    "Missing or ambiguous parameters for capillary pressure in "
                    "WaterOil object. Using zero."
                )
            )
        if not wateroil.selfcheck():
            raise ValueError(
                ("Incomplete WaterOil object, some parameters missing to factory")
            )
        return wateroil

    @staticmethod
    def create_gas_oil(
        params: Optional[Dict[str, float]] = None, fast: bool = False
    ) -> GasOil:
        """Create a GasOil object from a dictionary of parameters.

        Parameterization (Corey/LET) is inferred from presence
        of certain parameters in the dictionary.

        Don't rely on behaviour of you supply both Corey and LET at
        the same time.

        NB: the add_LET_* methods have the names 'l', 'e' and 't'
        in their signatures, which is not precise enough in this
        context, so we require e.g. 'Lg' and 'Log' (which both will be
        translated to 'l').

        Recognized parameters:
          swirr, sgcr, sorg, swl, krgendanchor, h, tag,
          ng, krgend, krgmax, nog, kroend, kromax
          lg, eg, tg, log, eog, tog

        Args:
            params: Dictionary with parameters describing the GasOil object.
            fast: If fast-mode should be set for constructed object.
        """
        if not params:
            params = {}
        if not isinstance(params, dict):
            raise TypeError("Parameter to create_gas_oil must be a dictionary")

        check_deprecated(params)

        sufficient_gas_oil_params(params, failhard=True)

        # For case insensitiveness, all keys are converted to lower case:
        params = {key.lower(): value for (key, value) in params.items()}

        # Allowing sending in NaN values, delete those keys.
        params = filter_nan_from_dict(params)

        usedparams: Set[str] = set()
        # No requirements to the base objects, defaults are ok.
        gasoil = GasOil(**slicedict(params, GO_INIT), fast=fast)
        usedparams = usedparams.union(set(slicedict(params, GO_INIT).keys()))
        logger.debug(
            "Initialized GasOil object from parameters %s", str(list(usedparams))
        )

        # Gas curve
        params_corey_gas = slicedict(params, GO_COREY_GAS + GO_GAS_ENDPOINTS)
        params_let_gas = slicedict(params, GO_LET_GAS + GO_GAS_ENDPOINTS)
        if set(GO_COREY_GAS).issubset(set(params_corey_gas)):
            gasoil.add_corey_gas(**params_corey_gas)
            logger.debug(
                "Added Corey gas to GasOil object from parameters %s",
                str(params_corey_gas.keys()),
            )
        elif set(GO_LET_GAS).issubset(set(params_let_gas)):
            params_let_gas["l"] = params_let_gas.pop("lg")
            params_let_gas["e"] = params_let_gas.pop("eg")
            params_let_gas["t"] = params_let_gas.pop("tg")
            gasoil.add_LET_gas(**params_let_gas)
            logger.debug(
                "Added LET gas to GasOil object from parameters %s",
                str(params_let_gas.keys()),
            )
        else:
            logger.warning(
                "Missing or ambiguous parameters for gas curve in GasOil object"
            )

        # Oil curve:
        params_corey_oil = slicedict(params, GO_COREY_OIL + GO_OIL_ENDPOINTS)
        params_let_oil = slicedict(params, GO_LET_OIL + GO_OIL_ENDPOINTS)
        if set(GO_COREY_OIL).issubset(set(params_corey_oil)):
            gasoil.add_corey_oil(**params_corey_oil)
            logger.debug(
                "Added Corey gas to GasOil object from parameters %s",
                str(params_corey_oil.keys()),
            )
        elif set(GO_LET_OIL).issubset(set(params_let_oil)):
            params_let_oil["l"] = params_let_oil.pop("log")
            params_let_oil["e"] = params_let_oil.pop("eog")
            params_let_oil["t"] = params_let_oil.pop("tog")
            gasoil.add_LET_oil(**params_let_oil)
            logger.debug(
                "Added LET gas to GasOil object from parameters %s",
                str(params_let_oil.keys()),
            )
        else:
            logger.warning(
                "Missing or ambiguous parameters for oil curve in GasOil object"
            )
        if not gasoil.selfcheck():
            raise ValueError(
                ("Incomplete GasOil object, some parameters missing to factory")
            )

        return gasoil

    @staticmethod
    def create_water_oil_gas(
        params: Optional[Dict[str, float]] = None, fast: bool = False
    ) -> WaterOilGas:
        """Create a WaterOilGas object from a dictionary of parameters

        Parameterization (Corey/LET) is inferred from presence
        of certain parameters in the dictionary.

        Check create_water_oil() and create_gas_oil() for lists
        of supported parameters (case insensitive)

        Params:
            params: Dictionary with parameters describing the WaterOilGas object.
            fast: If fast-mode should be set for constructed object.
        """
        if not params:
            params = {}
        if not isinstance(params, dict):
            raise TypeError("Parameter to create_water_oil_gas must be a dictionary")

        check_deprecated(params)

        # For case insensitiveness, all keys are converted to lower case:
        params = {key.lower(): value for (key, value) in params.items()}

        wateroil: Optional[WaterOil]
        if sufficient_water_oil_params(params, failhard=False):
            wateroil = PyscalFactory.create_water_oil(params, fast=fast)
        else:
            logger.info("No wateroil parameters. Assuming only gas-oil in wateroilgas")
            wateroil = None

        # If the swl in WaterOil was initialized with swlheight,
        # ensure that result is passed on to the GasOil object:
        if "swl" not in params and "swlheight" in params and wateroil is not None:
            params["swl"] = wateroil.swl

        gasoil: Optional[GasOil]
        if sufficient_gas_oil_params(params, failhard=False):
            gasoil = PyscalFactory.create_gas_oil(params, fast=fast)
        else:
            logger.info("No gasoil parameters, assuming two-phase oilwatergas")
            gasoil = None

        wog_init_params = slicedict(params, WOG_INIT)
        wateroilgas = WaterOilGas(**wog_init_params, fast=fast)
        # The wateroilgas __init__ has already created WaterOil and GasOil objects
        # but we overwrite the references with newly created ones, this factory function
        # must then guarantee that they are compatible.
        wateroilgas.wateroil = wateroil  # This might be None
        wateroilgas.gasoil = gasoil  # This might be None
        if not wateroilgas.selfcheck():
            raise ValueError(
                f"Inconsistent WaterOilGas object. Bug? Input was {params}"
            )
        return wateroilgas

    @staticmethod
    def create_gas_water(
        params: Optional[Dict[str, float]] = None, fast: bool = False
    ) -> GasWater:
        """Create a GasWater object.

        Parameterization (Corey/LET) is inferred from presence
        of certain parameters in the dictionary.

        Args:
            params: Dictionary with parameters for GasWater.
            fast: If fast-mode should be set for constructed object.
        """
        if not params:
            params = {}
        if not isinstance(params, dict):
            raise TypeError("Parameter to create_gas_water must be a dictionary")

        # For case insensitiveness, all keys are converted to lower case:
        params = {key.lower(): value for (key, value) in params.items()}

        sufficient_gas_water_params(params, failhard=True)

        gw_init_params = slicedict(params, GW_INIT)
        gaswater = GasWater(**gw_init_params, fast=fast)

        # We are using the create_water_oil_gas factory function
        # to avoid replicating code. It works because GasWater and
        # WaterOilGas internally are very similar.
        wog_params = params.copy()
        if "sgrw" in params:
            wog_params["sorw"] = params["sgrw"]
        # Set some dummy parameters for oil:
        wog_params["nog"] = 1
        wog_params["now"] = 1
        wog = PyscalFactory.create_water_oil_gas(wog_params, fast=fast)
        assert wog.wateroil is not None
        assert wog.gasoil is not None
        gaswater.wateroil = wog.wateroil
        gaswater.gasoil = wog.gasoil
        return gaswater

    @staticmethod
    def create_scal_recommendation(
        params: Dict[str, Dict[str, float]],
        tag: str = "",
        h: Optional[float] = None,
        fast: bool = False,
    ) -> SCALrecommendation:
        """
        Set up a SCAL recommendation curve set from input as a
        dictionary of dictionary.

        The keys in in the dictionary *must* be "low", "base" and "high".

        The value for "low" must be a new dictionary with saturation
        endpoints and LET/Corey parameters, as you would feed it to
        the create_water_oil_gas() factory function, and then similarly
        for base and high.

        For oil-water only, you may omit the parameters for gas-oil.
        A WaterOilGas object for each case is created, but only the
        WaterOil part of it will be used.

        For gas-water, a GasWater object is created for each pess, base
        and high.

        Args:
            params: keys low, base and high.
                The value for "low" must be a new dictionary with saturation
                endpoints and LET/Corey parameters, as you would feed it to
                the create_water_oil_gas() factory function, and then similarly
                for base and high.
            tag: String to be used as the tag, will end up in comments.
            h: Saturation step length
            fast: If fast-mode should be set for constructed object.
        """
        if not isinstance(params, dict):
            raise ValueError("Input must be a dictionary (of dictionaries)")

        # For case insensitiveness, all keys are converted to lower case:
        params = {key.lower(): value for (key, value) in params.items()}

        if "low" not in params:
            raise ValueError('"low" case not supplied')
        if "base" not in params:
            raise ValueError('"base" case not supplied')
        if "high" not in params:
            raise ValueError('"high" case not supplied')
        if not (
            isinstance(params["low"], dict)
            and isinstance(params["base"], dict)
            and isinstance(params["high"], dict)
        ):
            raise ValueError("All values in parameter dict must be dictionaries")

        check_deprecated(params["low"])
        check_deprecated(params["base"])
        check_deprecated(params["high"])

        errored = False
        if h is not None:
            params["low"]["h"] = h
            params["base"]["h"] = h
            params["high"]["h"] = h

        # Check parameter availability, in order to determine phase configuration
        gaswater = all(
            sufficient_gas_water_params(params[case]) for case in params.keys()
        )
        gasoil = all(sufficient_gas_oil_params(params[case]) for case in params.keys())
        wateroil = all(
            sufficient_water_oil_params(params[case]) for case in params.keys()
        )

        wog_low: Union[WaterOilGas, GasWater]
        wog_base: Union[WaterOilGas, GasWater]
        wog_high: Union[WaterOilGas, GasWater]

        if wateroil or gasoil:
            try:
                wog_low = PyscalFactory.create_water_oil_gas(params["low"], fast=fast)
            except ValueError as err:
                raise ValueError(f"Problem with low/pess case: {err}") from err
            try:
                wog_base = PyscalFactory.create_water_oil_gas(params["base"], fast=fast)
            except ValueError as err:
                raise ValueError(f"Problem with base case: {err}") from err
            try:
                wog_high = PyscalFactory.create_water_oil_gas(params["high"], fast=fast)
            except ValueError as err:
                raise ValueError(f"Problem with high/opt case: {err}") from err
        elif gaswater:
            # Note that gaswater will be True in three-phase configs.
            try:
                wog_low = PyscalFactory.create_gas_water(params["low"], fast=fast)
            except ValueError as err:
                raise ValueError(f"Problem with low/pess case: {err}") from err
            try:
                wog_base = PyscalFactory.create_gas_water(params["base"], fast=fast)
            except ValueError as err:
                raise ValueError(f"Problem with base case: {err}") from err
            try:
                wog_high = PyscalFactory.create_gas_water(params["high"], fast=fast)
            except ValueError as err:
                raise ValueError(f"Problem with high/opt case: {err}") from err

        errored = all(not wog.selfcheck() for wog in [wog_low, wog_base, wog_high])

        if errored:
            raise ValueError("Incomplete SCAL recommendation")

        scal = SCALrecommendation(wog_low, wog_base, wog_high, tag)
        return scal

    @staticmethod
    def load_relperm_df(
        inputfile: Union[str, pd.DataFrame], sheet_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Read CSV or XLSX from file and return scal/relperm data
        a dataframe.

        Checks validity in SATNUM and CASE columns.
        Ensures case-insensitiveness SATNUM, CASE, TAG and COMMENT

        Merges COMMENT into TAG column, as only TAG is picked up downstream.
        Adds a prefix "SATNUM <number>" to all tags.

        All strings in CASE column are converted to lowercase. Applies
        aliasing in the CASE column so that "pessimistic" and "pess" map to
        "low", and "optimistic" and "opt" map to "high".

        Args:
            inputfile: Filename for XLSX or CSV file, or a
                pandas DataFrame.
            sheet_name: Sheet-name, only used when loading xlsx files.

        Returns:
            To be handed over to pyscal list factory methods.
            Empty dataframe in case of errors (messages will be logged).
        """
        if isinstance(inputfile, (str, Path)) and Path(inputfile).is_file():
            tabular_file_format = infer_tabular_file_format(inputfile)
            if not tabular_file_format:
                raise ValueError(
                    f"Impossible to infer file format for {inputfile}, not CSV/XLS/XLSX"
                )

            if tabular_file_format == "csv" and sheet_name is not None:
                logger.warning(
                    "Sheet name only relevant for XLSX files, ignoring %s", sheet_name
                )
            excel_engines = {"xls": "xlrd", "xlsx": "openpyxl"}
            if tabular_file_format != "csv" and sheet_name:
                try:
                    input_df = pd.read_excel(
                        inputfile,
                        sheet_name=sheet_name,
                        engine=excel_engines[tabular_file_format],
                    )
                    logger.info(
                        "Parsed %s file %s, sheet %s",
                        tabular_file_format.upper(),
                        inputfile,
                        sheet_name,
                    )
                except (KeyError, ValueError) as err:
                    raise ValueError(
                        f"Non-existing sheet-name {sheet_name} provided."
                    ) from err
            elif tabular_file_format.startswith("xls"):
                input_df = pd.read_excel(
                    inputfile, engine=excel_engines[tabular_file_format]
                )
                logger.info("Parsed %s file %s", tabular_file_format.upper(), inputfile)
            else:
                input_df = pd.read_csv(
                    inputfile, skipinitialspace=True, encoding="utf-8"
                )
                logger.info("Parsed CSV file %s", inputfile)

        elif isinstance(inputfile, pd.DataFrame):
            input_df = inputfile
        else:
            if isinstance(inputfile, str) and not Path(inputfile).is_file():
                raise IOError("File not found " + str(inputfile))
            raise ValueError("Unsupported argument " + str(inputfile))
        assert isinstance(input_df, pd.DataFrame)

        if input_df.empty:
            logger.error("Relperm input dataframe is empty!")

        # We will ignore case on the column names, solved by converting column
        # name to uppercase
        ignorecasecolumns = ["satnum", "case", "tag", "comment"]
        for colname in input_df.columns:
            if colname.lower() in ignorecasecolumns:
                input_df.rename(
                    {colname: colname.upper()}, axis="columns", inplace=True
                )

        # Support both TAG and COLUMN (as TAG)
        if "COMMENT" in input_df and "TAG" not in input_df:
            input_df.rename({"COMMENT": "TAG"}, axis="columns", inplace=True)
        if "COMMENT" in input_df and "TAG" in input_df:
            # It might never happen that a user puts both tag and comment in
            # the dataframe, but if so, merge them.
            input_df["TAG"] = (
                "tag: " + input_df["TAG"] + "; comment: " + input_df["COMMENT"]
            )

        # It is tempting to detect any NaN's at this point because that can
        # indicate merged cells, which is not supported, but that could
        # break optional comment columns which might only be defined on certain lines.
        if "SATNUM" not in input_df:
            raise ValueError("SATNUM must be present in CSV/XLSX file/dataframe")

        # Delete columns and rows that are all NaNs (this has been observed
        # to occur from seemingly empty cells in Excel/LibreOffice and
        # has been seen to vary with Pandas/Openpyxl versions in use)
        input_df.dropna(axis="columns", how="all", inplace=True)
        input_df.dropna(axis="index", how="all", inplace=True)

        if input_df["SATNUM"].isnull().sum() > 0:
            raise ValueError(
                "Found not-a-number in the SATNUM column. This could be due to "
                "merged cells in XLSX, which is not supported."
            )

        # Warn about any other columns with empty cells:
        nan_columns = set(input_df.columns[input_df.isnull().any()])
        allowed_nan_columns = set(["COMMENT"])
        if nan_columns - allowed_nan_columns:
            logger.warning(
                "Found empty cells in these columns, this might create trouble: %s",
                str(nan_columns - allowed_nan_columns),
            )

        # Check that SATNUM's are consecutive and integers:
        try:
            input_df["SATNUM"] = input_df["SATNUM"].astype(int)
        except ValueError as err:
            raise ValueError("SATNUM must contain only integers") from err

        if min(input_df["SATNUM"]) != 1:
            raise ValueError("SATNUM must start at 1")

        if max(input_df["SATNUM"]) != len(input_df["SATNUM"].unique()):
            raise ValueError(
                "Missing SATNUMs? Max SATNUM is not equal to number of unique SATNUMS"
            )
        if "CASE" not in input_df and len(input_df["SATNUM"].unique()) != len(input_df):
            raise ValueError("Non-unique SATNUMs?")
        # If we are in a SCAL recommendation setting
        if "CASE" in input_df:
            # Enforce lower case:
            if input_df["CASE"].isnull().sum() > 0:
                raise ValueError(
                    "Found not-a-number in the CASE column. This could be due "
                    "merged cells in XLSX, which is not supported."
                )
            input_df["CASE"] = PyscalFactory.remap_validate_cases(
                input_df["CASE"].values
            )

        # Add the SATNUM index to the TAG column
        if "TAG" not in input_df:
            input_df["TAG"] = ""
        input_df["TAG"].fillna(value="", inplace=True)  # Empty cells to empty string.
        input_df["TAG"] = (
            "SATNUM " + input_df["SATNUM"].astype(str) + " " + input_df["TAG"]
        )

        # Check if fast is defined as a column
        if "fast" in input_df:
            logger.warning("Fast mode is not an option for individual SATNUMs")
            logger.warning("it is implemented as a global option.")
            logger.warning("The fast column in the dataframe will be ignored.")
            logger.warning("Use fast=True in the function call instead.")
            logger.warning("Fast mode is only available through the Python API")

        # Check that we are able to make something out of the first row:
        firstrow = input_df.iloc[0, :]
        error: bool = False
        wo_ok = sufficient_water_oil_params(firstrow)
        go_ok = sufficient_gas_oil_params(firstrow)
        gw_ok = sufficient_gas_water_params(firstrow)
        if error or not wo_ok and not go_ok and not gw_ok:
            raise ValueError(
                "Can't make neither WaterOil, GasOil or GasWater from "
                "the given data. Check documentation for what you need to supply. "
                f"You provided the columns {input_df.columns.values}"
            )
        logger.info(
            "Loaded input data with %s SATNUMS, column %s",
            str(len(input_df["SATNUM"].unique())),
            str(input_df.columns.values),
        )
        return input_df.sort_values("SATNUM")

    @staticmethod
    def alias_sgrw(params: Dict[str, Any]) -> Dict[str, Any]:
        """Allow sgrw as an alias for sorw by remapping a
        sgrw value to a sorw value in an incoming dict.

        Will error hard of sorw already exists and is not nan.

        This aliasing is relevant when GasWater is modelled
        as three-phase to allow for condensate forming, and mirrors
        GasWater.__init__() which performs the same aliasing.

        Args:
            params: Keys must be lower case.
        """
        if "sgrw" in params:
            if "sorw" not in params or pd.isnull(params["sorw"]):
                params_copy = dict(params)
                params_copy["sorw"] = params["sgrw"]
                del params_copy["sgrw"]
                return params_copy
            if np.isclose(params["sgrw"], params["sorw"]):
                params_copy = dict(params)
                del params_copy["sgrw"]
                return params_copy
            raise ValueError(
                f"sgrw ({params['sgrw']}) must equal sorw ({params['sorw']}) "
                "when both are supplied to WaterOil."
            )
        return params

    @staticmethod
    def remap_validate_cases(casevalues: List[str]) -> List[str]:
        """Remap values in the CASE column so that we can use aliases.

        All values are first made lower case, then
        "pessimistic" and "pess" are mapped to "low" and
        "optimistic" and "opt" are mapped to "high".

        Will raise ValueError if some values are not understood, and
        if we don't have exactly three unique values.

        Args:
            casevalues: values to remap.
        """
        accepted = ["low", "base", "high"]
        aliases = {
            "pessimistic": "low",
            "pess": "low",
            "optimistic": "high",
            "opt": "high",
        }
        lowered = [value.lower() for value in casevalues]
        remapped = [aliases.get(value, value) for value in lowered]
        not_understood = set(remapped) - set(accepted)
        if not_understood:
            raise ValueError(f"Invalid case values: {not_understood}")
        if len(set(remapped)) != len(accepted):
            raise ValueError(
                f"You must supply low, base AND high, got only {set(remapped)}"
            )
        return remapped

    @staticmethod
    def create_scal_recommendation_list(
        input_df: pd.DataFrame, h: Optional[float] = None, fast: bool = False
    ) -> PyscalList:
        """Requires SATNUM and CASE to be defined in the input data

        Args:
            input_df: Input data, should have been processed
                through load_relperm_df().
            h: Saturation step-value
            fast: If fast-mode should be set for constructed object

        Returns:
            PyscalList, consisting of SCALrecommendation objects
        """
        scal_l = PyscalList()
        assert isinstance(input_df, pd.DataFrame)

        scalinput = input_df.set_index(["SATNUM", "CASE"])

        for satnum in scalinput.index.levels[0].values:
            # load_relperm_df only validates the CASE column for all SATNUMs at
            # once, errors for particular SATNUMs are caught here.
            if len(scalinput.loc[satnum, :]) > 3:
                raise ValueError(f"Too many cases supplied for SATNUM {satnum}")
            if len(scalinput.loc[satnum, :]) < 3:
                raise ValueError(f"Too few cases supplied for SATNUM {satnum}")
            try:
                scal_l.append(
                    PyscalFactory.create_scal_recommendation(
                        scalinput.loc[satnum, :].to_dict(orient="index"), h=h, fast=fast
                    )
                )
            except ValueError as err:
                raise ValueError(f"Error for SATNUM {satnum}: {str(err)}") from err

        return scal_l

    @staticmethod
    def create_pyscal_list(
        relperm_params_df: pd.DataFrame, h: Optional[float] = None, fast: bool = False
    ):
        """Create WaterOilGas, WaterOil, GasOil or GasWater list
        based on what is available

        Args:
            relperm_params_df: Input data, should have been processed
                through load_relperm_df().
            h: Saturation step-value
            fast: If fast-mode should be set for constructed object

        Returns:
            PyscalList, consisting of either WaterOil, GasOil or WaterOilGas objects
        """
        params = relperm_params_df.iloc[0, :]  # first row
        water_oil = sufficient_water_oil_params(params)
        gas_oil = sufficient_gas_oil_params(params)
        gas_water = sufficient_gas_water_params(params)

        if water_oil and gas_oil:
            return PyscalFactory.create_wateroilgas_list(relperm_params_df, h, fast)
        if water_oil:
            return PyscalFactory.create_wateroil_list(relperm_params_df, h, fast)
        if gas_oil:
            return PyscalFactory.create_gasoil_list(relperm_params_df, h, fast)
        if gas_water:
            return PyscalFactory.create_gaswater_list(relperm_params_df, h, fast)
        raise ValueError("Could not determine two or three phase from parameters")

    @staticmethod
    def create_wateroilgas_list(
        relperm_params_df: pd.DataFrame, h: Optional[float] = None, fast: bool = False
    ) -> PyscalList:
        """Create a PyscalList with WaterOilGas objects from
        a dataframe

        Args:
            relperm_params_df: Input data, should have been processed
                through load_relperm_df().
            h: Saturation step-value
            fast: If fast-mode should be set for constructed object

        Returns:
            PyscalList, consisting of WaterOilGas objects
        """
        wogl = PyscalList()
        for (row_idx, params) in relperm_params_df.sort_values("SATNUM").iterrows():
            if h is not None:
                params["h"] = h
            try:
                wogl.append(
                    PyscalFactory.create_water_oil_gas(params.to_dict(), fast=fast)
                )
            except (AssertionError, ValueError, TypeError) as err:
                raise ValueError(f"Error for SATNUM {row_idx+1}: {str(err)}") from err
        return wogl

    @staticmethod
    def create_wateroil_list(
        relperm_params_df: pd.DataFrame, h: Optional[float] = None, fast: bool = False
    ) -> PyscalList:
        """Create a PyscalList with WaterOil objects from
        a dataframe

        Args:
            relperm_params_df: A valid dataframe with
                WaterOil parameters, processed through load_relperm_df()
            h: Saturation steplength
            fast: If fast-mode should be set for constructed object

        Returns:
            PyscalList, consisting of WaterOil objects
        """
        wol = PyscalList()
        for (_, params) in relperm_params_df.iterrows():
            if h is not None:
                params["h"] = h
            try:
                wol.append(PyscalFactory.create_water_oil(params.to_dict(), fast=fast))
            except (AssertionError, ValueError, TypeError) as err:
                raise ValueError(
                    f"Error for SATNUM {params['SATNUM']}: {str(err)}"
                ) from err
        return wol

    @staticmethod
    def create_gasoil_list(
        relperm_params_df: pd.DataFrame, h: Optional[float] = None, fast: bool = False
    ) -> PyscalList:
        """Create a PyscalList with GasOil objects from
        a dataframe

        Args:
            relperm_params_df: A valid dataframe with GasOil parameters,
                processed through load_relperm_df()
            h: Saturation steplength
            fast: If fast-mode should be set for constructed object

        Returns:
            PyscalList, consisting of GasOil objects
        """
        gol = PyscalList()
        for (_, params) in relperm_params_df.iterrows():
            if h is not None:
                params["h"] = h
            try:
                gol.append(PyscalFactory.create_gas_oil(params.to_dict(), fast=fast))
            except (AssertionError, ValueError, TypeError) as err:
                raise ValueError(
                    f"Error for SATNUM {params['SATNUM']}: {str(err)}"
                ) from err
        return gol

    @staticmethod
    def create_gaswater_list(
        relperm_params_df: pd.DataFrame, h: Optional[float] = None, fast: bool = False
    ) -> PyscalList:
        """Create a PyscalList with WaterOilGas objects from
        a dataframe, to be used for GasWater

        Args:
            relperm_params_df: A valid dataframe with GasWater
                parameters, processed through load_relperm_df()
            h: Saturation steplength
            fast: If fast-mode should be set for constructed object

        Returns:
            PyscalList, consisting of GasWater objects
        """
        gwl = PyscalList()
        for (_, params) in relperm_params_df.iterrows():
            if h is not None:
                params["h"] = h
            try:
                gwl.append(PyscalFactory.create_gas_water(params.to_dict(), fast=fast))
            except (AssertionError, ValueError, TypeError) as err:
                raise ValueError(
                    f"Error for SATNUM {params['SATNUM']}: {str(err)}"
                ) from err
        return gwl


def sufficient_water_oil_params(params: dict, failhard: bool = False) -> bool:
    """Determine if the supplied parameters are sufficient for
    attempting creating a WaterOil object.

    In the factory context, relying on the defaults in the API
    is not allowed, as that would leave the tasks for the factory
    undefined (Corey or LET, and which pc?)

    Args:
        params: Dictionary of parameters to a WaterOil object.
        failhard: If True, will raise ValueError when
            parameters are insufficient. If defaulted, no
            exception is raised.

    Returns:
        True if a WaterOil object should be attempted constructed
        (but no guarantee for validity of numerical values)
    """
    # pylint: disable=C1801
    # For case insensitiveness, all keys are converted to lower case:
    params = {key.lower(): value for (key, value) in params.items()}

    water_ok = (
        len(slicedict(params, set(WO_COREY_WATER))) == 1
        or len(slicedict(params, set(WO_LET_WATER))) == 3
    )
    oil_ok = (
        len(slicedict(params, set(WO_COREY_OIL))) == 1
        or len(slicedict(params, set(WO_LET_OIL))) == 3
        or len(slicedict(params, set(WO_LET_OIL_ALT))) == 3
    )
    if failhard and not (water_ok and oil_ok):
        raise ValueError("Missing WaterOil parameters in " + str(params))
    return water_ok and oil_ok


def sufficient_gas_oil_params(params: dict, failhard: bool = False) -> bool:
    """Determine if the supplied parameters are sufficient for
    attempting at creating a GasOil object.

    In the factory context, relying on the defaults in the API
    is not allowed, as that would leave the tasks for the factory
    undefined (Corey or LET, and which pc?)

    Args:
        params: Dictionary of parameters to a GasOil object.
        failhard: If True, will raise ValueError when
            parameters are insufficient. If defaulted, no
            exception is raised.

    Returns:
        True if a GasOil object should be attempted constructed
        (but no guarantee for validity of numerical values)
    """
    # pylint: disable=C1801
    # For case insensitiveness, all keys are converted to lower case:
    params = {key.lower(): value for (key, value) in params.items()}

    oil_ok = (
        len(slicedict(params, set(GO_COREY_OIL))) == 1
        or len(slicedict(params, set(GO_LET_OIL))) == 3
    )
    gas_ok = (
        len(slicedict(params, set(GO_COREY_GAS))) == 1
        or len(slicedict(params, set(GO_LET_GAS))) == 3
    )
    if failhard and not (oil_ok and gas_ok):
        raise ValueError("Missing gas-oil parameters in " + str(params))
    return oil_ok and gas_ok


def sufficient_gas_water_params(params: dict, failhard: bool = False) -> bool:
    """Determine if the supplied parameters are sufficient for
    attempting creating a WaterOilGas object to be used for gas water.

    In the factory context, relying on the defaults in the API
    (wateroilgas.py) is not allowed, as that would leave the tasks
    for the factory undefined (Corey or LET, and which pc?)

    Args:
        params: Dictionary of parameters to a GasWater object.
        failhard: If True, will raise ValueError when
            parameters are insufficient. If defaulted, no
            exception is raised.

    Returns:
        True if a GasWater object should be attempted constructed
            (but no guarantee for validity of numerical values)
    """
    # pylint: disable=C1801
    # For case insensitiveness, all keys are converted to lower case:
    params = {key.lower(): value for (key, value) in params.items()}

    water_ok = (
        len(slicedict(params, set(GW_COREY_WATER))) == 1
        or len(slicedict(params, set(GW_LET_WATER))) == 3
    )
    gas_ok = (
        len(slicedict(params, set(GW_COREY_GAS))) == 1
        or len(slicedict(params, set(GW_LET_GAS))) == 3
    )
    if failhard and not (water_ok and gas_ok):
        raise ValueError("Missing gas-water parameters in " + str(params))
    return water_ok and gas_ok


def filter_nan_from_dict(params: dict) -> dict:
    """Clean out keys with NaN values in a dict.

    Key with string values are passed through (empty strings are allowed)

    Args:
        params (dict): Any dictionary

    Returns
        dict, with as many or fewer keys.
    """
    cleaned_params = {}
    for key, value in params.items():
        if isinstance(value, str):
            cleaned_params[key] = value
        else:
            if not np.isnan(value):
                cleaned_params[key] = value
    return cleaned_params


def infer_tabular_file_format(filename: Union[str, Path]) -> str:
    """Determine the file format of a file containing tabular data,
    distinguishes between csv, xls and xlsx

    Args:
        filename: Path to file

    Returns:
        One of "csv", "xlsx" or "xls". Empty string if nothing found out.
    """
    try:
        pd.read_excel(filename, engine="openpyxl")
        return "xlsx"
    except (
        ValueError,
        OSError,
        openpyxl.utils.exceptions.InvalidFileException,
        zipfile.BadZipFile,
    ):
        # < Pandas 1.2, we get InvalidFileException from openpyxl
        # Pandas 1.2.0 - 1.2.1: CSV gives ValueError,
        #                       XLS gives OSError
        # >= Pandas 1.2.2:      CSV gives zipfile.BadZipFile,
        #                       XLS gives OSError
        pass
    try:
        pd.read_excel(filename, engine="xlrd")
        return "xls"
    except (ValueError, TypeError, xlrd.biffh.XLRDError):
        # We get here for both CSV and XLSX files.
        pass
    try:
        dframe = pd.read_csv(filename, encoding="utf-8")
        # >= 1.2.1: Pandas: Bugfix makes read_csv() more encoding fault
        #                   tolerant which this code opts out from.
        if not dframe.empty:
            return "csv"
    except UnicodeDecodeError:
        # (xls and xlsx files)
        pass
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as csverror:
        logger.error("Message from CSV parser: %s", str(csverror))
        # (Some text file that is not CSV)

    return ""


def check_deprecated(params: Dict[str, Any]) -> None:
    """Check for deprecated parameter names

    Args:
        params: Dictionary of parameters for which only the keys are used here.
    """
    # Block long deprecated parameters with an exception.
    # Remove this block in pyscal 1.x
    if "krowend" in params and "kroend" not in params:
        raise ValueError("krowend is not supported by pyscal. Use kroend")
    if "krogend" in params and "kroend" not in params:
        raise ValueError("krogend is not supported by pyscal. Use kroend")
