"""Factory functions for creating the pyscal objects"""

import logging

import os
import six

import pandas as pd
import numpy as np
import xlrd

from pyscal import (
    WaterOil,
    GasOil,
    GasWater,
    WaterOilGas,
    SCALrecommendation,
    PyscalList,
)

logging.basicConfig()
logger = logging.getLogger(__name__)


def slicedict(dct, keys):
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
WO_INIT = ["swirr", "swl", "swcr", "sorw", "h", "tag"]
WO_COREY_WATER = ["nw"]
WO_WATER_ENDPOINTS = ["krwmax", "krwend"]
WO_COREY_OIL = ["now"]
WO_LET_WATER = ["lw", "ew", "tw"]  # Will translated to l, e and t in code below.
WO_LET_OIL = ["low", "eow", "tow"]
WO_LET_OIL_ALT = ["lo", "eo", "to"]  # Alternative parameter names.
WO_OIL_ENDPOINTS = ["kroend", "krowend"]  # krowend is deprecated in favour of kroend
WO_SIMPLE_J = ["a", "b", "poro_ref", "perm_ref", "drho"]  # "g" is optional
WO_NORM_J = ["a", "b", "poro", "perm", "sigma_costau"]
# 'a' in WO_NORM_J is the same as a_petro, but should possibly kept as is.
WO_SIMPLE_J_PETRO = [
    "a_petro",
    "b_petro",
    "poro_ref",
    "perm_ref",
    "drho",
]  # "g" is optional

GO_INIT = ["swirr", "sgcr", "sorg", "swl", "krgendanchor", "h", "tag"]
GO_COREY_GAS = ["ng"]
GO_GAS_ENDPOINTS = ["krgend", "krgmax"]
GO_COREY_OIL = ["nog"]
GO_OIL_ENDPOINTS = ["kroend", "krogend"]  # krogend is deprecated in favour of kroend
GO_LET_GAS = ["lg", "eg", "tg"]
GO_LET_OIL = ["log", "eog", "tog"]

GW_INIT = ["swirr", "swl", "swcr", "sgrw", "sgcr", "h", "tag"]
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

WOG_INIT = ["swirr", "swl", "swcr", "sorw", "sorg", "sgcr", "h", "tag"]

DEPRECATED = {
    "krogend": "and will be used as kroend for GasOil",
    "krowend": "and will be used as kroend for WaterOil",
    "kromax": "and will be ignored",
}


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
    def create_water_oil(params=None):
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
          swirr, swl, swcr, sorw, h, tag, nw, now, krwmax, krwend,
          lw, ew, tw, low, eow, tow, lo, eo, to, kroend,
          a, a_petro, b, b_petro, poro_ref, perm_ref, drho,
          a, b, poro, perm, sigma_costau

        Args:
            params (dict): Dictionary with parameters describing
                the WaterOil object.
        """
        if not params:
            params = dict()
        if not isinstance(params, dict):
            raise TypeError("Parameter to create_water_oil must be a dictionary")

        check_deprecated(params)

        sufficient_water_oil_params(params, failhard=True)

        # For case insensitiveness, all keys are converted to lower case:
        params = {key.lower(): value for (key, value) in params.items()}

        # Allowing sending in NaN values, delete those keys.
        params = filter_nan_from_dict(params)

        usedparams = set()
        # No requirements to the base objects, defaults are ok.
        wateroil = WaterOil(**slicedict(params, WO_INIT))
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
        else:
            logger.warning(
                "Missing or ambiguous parameters for water curve in WaterOil object"
            )

        # Oil curve:
        params_corey_oil = slicedict(params, WO_COREY_OIL + WO_OIL_ENDPOINTS)
        params_let_oil = slicedict(
            params, WO_LET_OIL + WO_LET_OIL_ALT + WO_OIL_ENDPOINTS
        )
        if set(WO_COREY_OIL).issubset(set(params_corey_oil)):
            if "krowend" in params_corey_oil:
                # (deprecation warning is triggered)
                params_corey_oil["kroend"] = params_corey_oil.pop("krowend")
            wateroil.add_corey_oil(**params_corey_oil)
            logger.debug(
                "Added Corey water to WaterOil object from parameters %s",
                str(params_corey_oil.keys()),
            )
        elif set(WO_LET_OIL).issubset(set(params_let_oil)):
            params_let_oil["l"] = params_let_oil.pop("low")
            params_let_oil["e"] = params_let_oil.pop("eow")
            params_let_oil["t"] = params_let_oil.pop("tow")
            if "krowend" in params_let_oil:
                # (deprecation warning is triggered)
                params_let_oil["kroend"] = params_let_oil.pop("krowend")
            wateroil.add_LET_oil(**params_let_oil)
            logger.debug(
                "Added LET water to WaterOil object from parameters %s",
                str(params_let_oil.keys()),
            )
        elif set(WO_LET_OIL_ALT).issubset(set(params_let_oil)):
            params_let_oil["l"] = params_let_oil.pop("lo")
            params_let_oil["e"] = params_let_oil.pop("eo")
            params_let_oil["t"] = params_let_oil.pop("to")
            if "krowend" in params_let_oil:
                # (deprecation warning is triggered)
                params_let_oil["kroend"] = params_let_oil.pop("krowend")
            wateroil.add_LET_oil(**params_let_oil)
            logger.debug(
                "Added LET water to WaterOil object from parameters %s",
                str(params_let_oil.keys()),
            )
        else:
            logger.warning(
                "Missing or ambiguous parameters for oil curve in WaterOil object"
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
    def create_gas_oil(params=None):
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
          ng, krgend, krgmax, nog, kroend,
          lg, eg, tg, log, eog, tog

        """
        if not params:
            params = dict()
        if not isinstance(params, dict):
            raise TypeError("Parameter to create_gas_oil must be a dictionary")

        check_deprecated(params)

        sufficient_gas_oil_params(params, failhard=True)

        # For case insensitiveness, all keys are converted to lower case:
        params = {key.lower(): value for (key, value) in params.items()}

        # Allowing sending in NaN values, delete those keys.
        params = filter_nan_from_dict(params)

        usedparams = set()
        # No requirements to the base objects, defaults are ok.
        gasoil = GasOil(**slicedict(params, GO_INIT))
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
            if "krogend" in params_corey_oil:
                # (deprecation warning will be triggered)
                params_corey_oil["kroend"] = params_corey_oil.pop("krogend")
            gasoil.add_corey_oil(**params_corey_oil)
            logger.debug(
                "Added Corey gas to GasOil object from parameters %s",
                str(params_corey_oil.keys()),
            )
        elif set(GO_LET_OIL).issubset(set(params_let_oil)):
            params_let_oil["l"] = params_let_oil.pop("log")
            params_let_oil["e"] = params_let_oil.pop("eog")
            params_let_oil["t"] = params_let_oil.pop("tog")
            if "krogend" in params_corey_oil:
                # (deprecation warning will be triggered)
                params_let_oil["kroend"] = params_let_oil.pop("krogend")
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
    def create_water_oil_gas(params=None):
        """Create a WaterOilGas object from a dictionary of parameters

        Parameterization (Corey/LET) is inferred from presence
        of certain parameters in the dictionary.

        Check create_water_oil() and create_gas_oil() for lists
        of supported parameters (case insensitive)

        Params:
            params (dict):  parameteres
            gaswater (bool): Flag to indicate if is to be used for GasWater
        """
        if not params:
            params = dict()
        if not isinstance(params, dict):
            raise TypeError("Parameter to create_water_oil_gas must be a dictionary")

        check_deprecated(params)

        # For case insensitiveness, all keys are converted to lower case:
        params = {key.lower(): value for (key, value) in params.items()}

        if sufficient_water_oil_params(params, failhard=False):
            wateroil = PyscalFactory.create_water_oil(params)
        else:
            logger.info("No wateroil parameters. Assuming only gas-oil in wateroilgas")
            wateroil = None

        if sufficient_gas_oil_params(params, failhard=False):
            gasoil = PyscalFactory.create_gas_oil(params)
        else:
            logger.info("No gasoil parameters, assuming two-phase oilwatergas")
            gasoil = None

        wog_init_params = slicedict(params, WOG_INIT)
        wateroilgas = WaterOilGas(**wog_init_params)
        # The wateroilgas __init__ has already created WaterOil and GasOil objects
        # but we overwrite the references with newly created ones, this factory function
        # must then guarantee that they are compatible.
        wateroilgas.wateroil = wateroil  # This might be None
        wateroilgas.gasoil = gasoil  # This might be None
        if not wateroilgas.selfcheck():
            raise ValueError(
                ("Incomplete WaterOilGas object, some parameters missing to factory")
            )
        return wateroilgas

    @staticmethod
    def create_gas_water(params=None):
        """Create a GasWater object.

        Parameterization (Corey/LET) is inferred from presence
        of certain parameters in the dictionary.

        Args:
            params (dict): Dictionary with parameters for GasWater.

        Returns:
            GasWater
        """
        if not params:
            params = dict()
        if not isinstance(params, dict):
            raise TypeError("Parameter to create_gas_water must be a dictionary")

        # For case insensitiveness, all keys are converted to lower case:
        params = {key.lower(): value for (key, value) in params.items()}

        sufficient_gas_water_params(params, failhard=True)

        gw_init_params = slicedict(params, GW_INIT)
        gaswater = GasWater(**gw_init_params)

        # We are using the create_water_oil_gas factory function
        # to avoid replicating code. It works because GasWater and
        # WaterOilGas internally are very similar.
        wog_params = params.copy()
        if "sgrw" in params:
            wog_params["sorw"] = params["sgrw"]
        # Set some dummy parameters for oil:
        wog_params["nog"] = 1
        wog_params["now"] = 1
        wog = PyscalFactory.create_water_oil_gas(wog_params)
        gaswater.wateroil = wog.wateroil
        gaswater.gasoil = wog.gasoil
        return gaswater

    @staticmethod
    def create_scal_recommendation(params, tag="", h=None):
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
            params (dict): keys low, base and high.
                The value for "low" must be a new dictionary with saturation
                endpoints and LET/Corey parameters, as you would feed it to
                the create_water_oil_gas() factory function, and then similarly
                for base and high.
            tag (string): String to be used as the tag, will end up in comments.
            h (float): Saturation step length

        Returns:
            SCALrecommendation
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
        if not all([isinstance(x, dict) for x in params.values()]):
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
            [sufficient_gas_water_params(params[case]) for case in params.keys()]
        )
        gasoil = all(
            [sufficient_gas_oil_params(params[case]) for case in params.keys()]
        )
        wateroil = all(
            [sufficient_water_oil_params(params[case]) for case in params.keys()]
        )

        if wateroil or gasoil:
            wog_low = PyscalFactory.create_water_oil_gas(params["low"])
            wog_base = PyscalFactory.create_water_oil_gas(params["base"])
            wog_high = PyscalFactory.create_water_oil_gas(params["high"])
        elif gaswater:
            # Note that gaswater will be True in three-phase configs.
            wog_low = PyscalFactory.create_gas_water(params["low"])
            wog_base = PyscalFactory.create_gas_water(params["base"])
            wog_high = PyscalFactory.create_gas_water(params["high"])

        errored = all([not wog.selfcheck() for wog in [wog_low, wog_base, wog_high]])

        if errored:
            raise ValueError("Incomplete SCAL recommendation")

        scal = SCALrecommendation(wog_low, wog_base, wog_high, tag)
        return scal

    @staticmethod
    def load_relperm_df(inputfile, sheet_name=None):
        """Read CSV or XLSX from file and return scal/relperm data
        a dataframe.

        Checks validity in SATNUM and CASE columns.
        Ensures case-insensitivenes SATNUM, CASE, TAG and COMMENT

        Merges COMMENT into TAG column, as only TAG is picked up downstream.
        Adds a prexix "SATNUM <number>" to all tags.

        All strings in CASE column are converted to lowercase. Applies
        aliasing in the CASE column so that "pessimistic" and "pess" map to
        "low", and "optimistic" and "opt" map to "high".

        Args:
            inputfile (str or pd.DataFrame): Filename for XLSX or CSV file, or a
                pandas DataFrame.
            sheet_name (str): Sheet-name, only used when loading xlsx files.

        Returns:
            pd.DataFrame. To be handed over to pyscal list factory methods.
        """
        if isinstance(inputfile, str) and os.path.exists(inputfile):
            if inputfile.lower().endswith("csv") and sheet_name is not None:
                logger.warning(
                    "Sheet name only relevant for XLSX files, ignoring %s", sheet_name
                )
            try:
                if sheet_name:
                    input_df = pd.read_excel(inputfile, sheet_name=sheet_name)
                    logger.info("Parsed XLSX file %s, sheet %s", inputfile, sheet_name)
                else:
                    input_df = pd.read_excel(inputfile)
                    logger.info("Parsed XLSX file %s", inputfile)
            except xlrd.XLRDError as xlserror:
                if inputfile.lower().endswith("xlsx") or inputfile.lower().endswith(
                    "xls"
                ):
                    logger.error(xlserror)
                try:
                    input_df = pd.read_csv(inputfile, skipinitialspace=True)
                    logger.info("Parsed CSV file %s", inputfile)
                except pd.errors.ParserError as csverror:
                    logger.error("Could not parse %s as XLSX or CSV", inputfile)
                    logger.error("Error message from csv-parser: %s", str(csverror))
                    input_df = pd.DataFrame()
                except ValueError:
                    # We end here when we use csv reader on xls files, that
                    # means that xls parsing failed in the first place. Other
                    # error messages have been, and will be printed.
                    input_df = pd.DataFrame()
        elif isinstance(inputfile, pd.DataFrame):
            input_df = inputfile
        else:
            if isinstance(inputfile, str) and not os.path.exists(inputfile):
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
            logger.error("SATNUM must be present in CSV/XLSX file/dataframe")
            raise ValueError("SATNUM missing")

        if input_df["SATNUM"].isnull().sum() > 0:
            logger.error(
                "Found not-a-number in the SATNUM column. This could be due to"
            )
            logger.error("merged cells in XLSX, which is not supported.")
            raise ValueError

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
        except ValueError:
            logger.error("SATNUM must contain only integers")
            raise ValueError
        if min(input_df["SATNUM"]) != 1:
            logger.error("SATNUM must start at 1")
            raise ValueError
        if max(input_df["SATNUM"]) != len(input_df["SATNUM"].unique()):
            logger.error(
                "Missing SATNUMs? Max SATNUM is not equal to number of unique SATNUMS"
            )
            raise ValueError
        if "CASE" not in input_df and len(input_df["SATNUM"].unique()) != len(input_df):
            logger.error("Non-unique SATNUMs?")
            raise ValueError
        # If we are in a SCAL recommendation setting
        if "CASE" in input_df:
            # Enforce lower case:
            if input_df["CASE"].isnull().sum() > 0:
                logger.error("Found not-a-number in the CASE column. This could be due")
                logger.error("merged cells in XLSX, which is not supported.")
                raise ValueError
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

        # Check that we are able to make something out of the first row:
        firstrow = input_df.loc[0, :]
        error = False
        try:
            wo_ok = sufficient_water_oil_params(firstrow)
            go_ok = sufficient_gas_oil_params(firstrow)
            gw_ok = sufficient_gas_water_params(firstrow)
        except ValueError:
            error = True
        if error or not wo_ok and not go_ok and not gw_ok:
            logger.error(
                "Can't make neither WaterOil, GasOil or GasWater from the given data."
            )
            logger.error("Check documentation for what you need to supply")
            logger.error("You provided the columns %s", str(input_df.columns.values))
            raise ValueError
        logger.info(
            "Loaded input data with %s SATNUMS, column %s",
            str(len(input_df["SATNUM"].unique())),
            str(input_df.columns.values),
        )
        return input_df.sort_values("SATNUM")

    @staticmethod
    def remap_validate_cases(casevalues):
        """Remap values in the CASE column so that we can use aliases.

        All values are first made lower case, then
        "pessimistic" and "pess" are mapped to "low" and
        "optimistic" and "opt" are mapped to "high".

        Will raise ValueError if some values are not understood, and
        if we don't have exactly three unique values.

        Args:
            casevalues (list of str): values to remap.
        """
        accepted = ["low", "base", "high"]
        aliases = {
            "pessimistic": "low",
            "pess": "low",
            "optimistic": "high",
            "opt": "high",
        }
        lowered = [value.lower() for value in casevalues]
        remapped = [
            aliases[value] if value in aliases.keys() else value for value in lowered
        ]
        not_understood = set(remapped) - set(accepted)
        if not_understood:
            logger.error("Invalid case values: %s", str(not_understood))
            raise ValueError
        if len(set(remapped)) != len(accepted):
            logger.error(
                "You must supply low, base AND high, got only %s", str(set(remapped))
            )
            raise ValueError
        return remapped

    @staticmethod
    def create_scal_recommendation_list(input_df, h=None):
        """Requires SATNUM and CASE to be defined in the input data

        Args:
            input_df (pd.DataFrame): Input data, should have been processed
                through load_relperm_df().
            h (float): Saturation step-value

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
                logger.error("Too many cases supplied for SATNUM %s", str(satnum))
                raise ValueError
            if len(scalinput.loc[satnum, :]) < 3:
                logger.error("Too few cases supplied for SATNUM %s", str(satnum))
                raise ValueError
            scal_l.append(
                PyscalFactory.create_scal_recommendation(
                    scalinput.loc[satnum, :].to_dict(orient="index"), h=h
                )
            )
        return scal_l

    @staticmethod
    def create_pyscal_list(relperm_params_df, h=None):
        """Create WaterOilGas, WaterOil, GasOil or GasWater list
        based on what is available

        Args:
            relperm_params_df (pd.DataFrame): Input data, should have been processed
                through load_relperm_df().
            h (float): Saturation step-value

        Returns:
            PyscalList, consisting of either WaterOil, GasOil or WaterOilGas objects
        """
        params = relperm_params_df.loc[0, :]  # first row
        water_oil = sufficient_water_oil_params(params)
        gas_oil = sufficient_gas_oil_params(params)
        gas_water = sufficient_gas_water_params(params)
        if water_oil and gas_oil:
            return PyscalFactory.create_wateroilgas_list(relperm_params_df, h)
        if water_oil:
            return PyscalFactory.create_wateroil_list(relperm_params_df, h)
        if gas_oil:
            return PyscalFactory.create_gasoil_list(relperm_params_df, h)
        if gas_water:
            return PyscalFactory.create_gaswater_list(relperm_params_df, h)
        logger.error("Could not determine two or three phase from parameters")
        return None

    @staticmethod
    def create_wateroilgas_list(relperm_params_df, h=None):
        """Create a PyscalList with WaterOilGas objects from
        a dataframe

        Args:
            relperm_params_df (pd.DataFrame): Input data, should have been processed
                through load_relperm_df().
            h (float): Saturation step-value

        Returns:
            PyscalList, consisting of WaterOilGas objects
        """
        wogl = PyscalList()
        for (_, params) in relperm_params_df.iterrows():
            if h is not None:
                params["h"] = h
            wogl.append(PyscalFactory.create_water_oil_gas(params.to_dict()))
        return wogl

    @staticmethod
    def create_wateroil_list(relperm_params_df, h=None):
        """Create a PyscalList with WaterOil objects from
        a dataframe

        Args:
            relperm_params_df (pd.DataFrame): A valid dataframe with
                WaterOil parameters, processed through load_relperm_df()
            h (float): Saturation steplength

        Returns:
            PyscalList, consisting of WaterOil objects
        """
        wol = PyscalList()
        for (_, params) in relperm_params_df.iterrows():
            if h is not None:
                params["h"] = h
            wol.append(PyscalFactory.create_water_oil(params.to_dict()))
        return wol

    @staticmethod
    def create_gasoil_list(relperm_params_df, h=None):
        """Create a PyscalList with GasOil objects from
        a dataframe

        Args:
            relperm_params_df (pd.DataFrame): A valid dataframe with GasOil parameters,
                processed through load_relperm_df()
            h (float): Saturation steplength

        Returns:
            PyscalList, consisting of GasOil objects
        """
        gol = PyscalList()
        for (_, params) in relperm_params_df.iterrows():
            if h is not None:
                params["h"] = h
            gol.append(PyscalFactory.create_gas_oil(params.to_dict()))
        return gol

    @staticmethod
    def create_gaswater_list(relperm_params_df, h=None):
        """Create a PyscalList with WaterOilGas objects from
        a dataframe, to be used for GasWater

        Args:
            relperm_params_df (pd.DataFrame): A valid dataframe with GasWater
                parameters, processed through load_relperm_df()
            h (float): Saturation steplength

        Returns:
            PyscalList, consisting of GasWater objects
        """
        gwl = PyscalList()
        for (_, params) in relperm_params_df.iterrows():
            if h is not None:
                params["h"] = h
            gwl.append(PyscalFactory.create_gas_water(params.to_dict()))
        return gwl


def sufficient_water_oil_params(params, failhard=False):
    """Determine if the supplied parameters are sufficient for
    attempting creating a WaterOil object.

    In the factory context, relying on the defaults in the API
    is not allowed, as that would leave the tasks for the factory
    undefined (Corey or LET, and which pc?)

    Args:
        params (dict): Dictionary of parameters to a WaterOil object.
        failhard (bool): If True, will raise ValueError when
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


def sufficient_gas_oil_params(params, failhard=False):
    """Determine if the supplied parameters are sufficient for
    attempting at creating a GasOil object.

    In the factory context, relying on the defaults in the API
    is not allowed, as that would leave the tasks for the factory
    undefined (Corey or LET, and which pc?)

    Args:
        params (dict): Dictionary of parameters to a GasOil object.
        failhard (bool): If True, will raise ValueError when
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


def sufficient_gas_water_params(params, failhard=False):
    """Determine if the supplied parameters are sufficient for
    attempting creating a WaterOilGas object to be used for gas water.

    In the factory context, relying on the defaults in the API
    (wateroilgas.py) is not allowed, as that would leave the tasks
    for the factory undefined (Corey or LET, and which pc?)

    Args:
        params (dict): Dictionary of parameters to a GasWater object.
        failhard (bool): If True, will raise ValueError when
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


def filter_nan_from_dict(params):
    """Clean out keys with NaN values in a dict.

    Key with string values are passed through (empty strings are allowed)

    Args:
        params (dict): Any dictionary

    Returns
        dict, with as many or fewer keys.
    """
    cleaned_params = {}
    for key, value in params.items():
        if isinstance(value, six.string_types):
            cleaned_params[key] = value
        else:
            if not np.isnan(value):
                cleaned_params[key] = value
    return cleaned_params


def check_deprecated(params):
    """Check for deprecated parameter names

    Args:
        params: Dictionary of parameters for which only the keys are used here.
    """
    for key in params:
        if key.lower() in DEPRECATED:
            logger.warning(
                "The key %s is deprecated %s", key, DEPRECATED[key.lower()],
            )
