"""Factory functions for creating the pyscal objects"""

import logging

import os
import six

import pandas as pd
import numpy as np
import xlrd

from pyscal import WaterOil, GasOil, WaterOilGas, SCALrecommendation, PyscalList

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
WO_OIL_ENDPOINTS = ["kromax", "krowend"]
WO_SIMPLE_J = ["a", "b", "poro_ref", "perm_ref", "drho"]  # "g" is optional
WO_NORM_J = ["a", "b", "poro", "perm", "sigma_costau"]

GO_INIT = ["swirr", "sgcr", "sorg", "swl", "krgendanchor", "h", "tag"]
GO_COREY_GAS = ["ng"]
GO_GAS_ENDPOINTS = ["krgend", "krgmax"]
GO_COREY_OIL = ["nog"]
GO_OIL_ENDPOINTS = ["krogend"]
GO_LET_GAS = ["lg", "eg", "tg"]
GO_LET_OIL = ["log", "eog", "tog"]

WOG_INIT = ["swirr", "swl", "swcr", "sorw", "sorg", "sgcr", "h", "tag"]

DEPRECATED = ["kroend"]  # This key will be ignored, as it it ambiguous.


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
          lw, ew, tw, low, eow, tow, lo, eo, to, kromax, krowend,
          a, b, poro_ref, perm_ref, drho,
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

        # For case insensitiveness, all keys are converted to lower case:
        params = {key.lower(): value for (key, value) in params.items()}

        # Allowing sending in NaN values, delete those keys.
        params = filter_nan_from_dict(params)

        usedparams = set()
        # No requirements to the base objects, defaults are ok.
        wateroil = WaterOil(**slicedict(params, WO_INIT))
        usedparams = usedparams.union(set(slicedict(params, WO_INIT).keys()))
        logger.info(
            "Initialized WaterOil object from parameters %s", str(list(usedparams))
        )

        # Water curve
        params_corey_water = slicedict(params, WO_COREY_WATER + WO_WATER_ENDPOINTS)
        params_let_water = slicedict(params, WO_LET_WATER + WO_WATER_ENDPOINTS)
        if set(WO_COREY_WATER).issubset(set(params_corey_water)):
            wateroil.add_corey_water(**params_corey_water)
            usedparams = usedparams.union(set(params_corey_water.keys()))
            logger.info(
                "Added Corey water to WaterOil object from parameters %s",
                str(params_corey_water.keys()),
            )
        elif set(WO_LET_WATER).issubset(set(params_let_water)):
            params_let_water["l"] = params_let_water.pop("lw")
            params_let_water["e"] = params_let_water.pop("ew")
            params_let_water["t"] = params_let_water.pop("tw")
            wateroil.add_LET_water(**params_let_water)
            usedparams = usedparams.union(set(params_let_water.keys()))
            logger.info(
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
                params_corey_oil["kroend"] = params_corey_oil.pop("krowend")
            wateroil.add_corey_oil(**params_corey_oil)
            logger.info(
                "Added Corey water to WaterOil object from parameters %s",
                str(params_corey_oil.keys()),
            )
        elif set(WO_LET_OIL).issubset(set(params_let_oil)):
            params_let_oil["l"] = params_let_oil.pop("low")
            params_let_oil["e"] = params_let_oil.pop("eow")
            params_let_oil["t"] = params_let_oil.pop("tow")
            if "krowend" in params_let_oil:
                params_let_oil["kroend"] = params_let_oil.pop("krowend")
            wateroil.add_LET_oil(**params_let_oil)
            logger.info(
                "Added LET water to WaterOil object from parameters %s",
                str(params_let_oil.keys()),
            )
        elif set(WO_LET_OIL_ALT).issubset(set(params_let_oil)):
            params_let_oil["l"] = params_let_oil.pop("lo")
            params_let_oil["e"] = params_let_oil.pop("eo")
            params_let_oil["t"] = params_let_oil.pop("to")
            if "krowend" in params_let_oil:
                params_let_oil["kroend"] = params_let_oil.pop("krowend")
            wateroil.add_LET_oil(**params_let_oil)
            logger.info(
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
        if set(WO_SIMPLE_J).issubset(set(params_simple_j)):
            wateroil.add_simple_J(**params_simple_j)
        elif set(WO_NORM_J).issubset(set(params_norm_j)):
            wateroil.add_normalized_J(**params_norm_j)
        else:
            logger.warning(
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
        translated to 'l'). Also note that in this factory context,
        kroend is an ambiguous parameter, krogend must be used.

        Recognized parameters:
          swirr, sgcr, sorg, swl, krgendanchor, h, tag,
          ng, krgend, krgmax, nog, krogend,
          lg, eg, tg, log, eog, tog

        """
        if not params:
            params = dict()
        if not isinstance(params, dict):
            raise TypeError("Parameter to create_gas_oil must be a dictionary")

        check_deprecated(params)

        # For case insensitiveness, all keys are converted to lower case:
        params = {key.lower(): value for (key, value) in params.items()}

        # Allowing sending in NaN values, delete those keys.
        params = filter_nan_from_dict(params)

        usedparams = set()
        # No requirements to the base objects, defaults are ok.
        gasoil = GasOil(**slicedict(params, GO_INIT))
        usedparams = usedparams.union(set(slicedict(params, GO_INIT).keys()))
        logger.info(
            "Initialized GasOil object from parameters %s", str(list(usedparams))
        )

        # Gas curve
        params_corey_gas = slicedict(params, GO_COREY_GAS + GO_GAS_ENDPOINTS)
        params_let_gas = slicedict(params, GO_LET_GAS + GO_GAS_ENDPOINTS)
        if set(GO_COREY_GAS).issubset(set(params_corey_gas)):
            gasoil.add_corey_gas(**params_corey_gas)
            usedparams = usedparams.union(set(params_corey_gas.keys()))
            logger.info(
                "Added Corey gas to GasOil object from parameters %s",
                str(params_corey_gas.keys()),
            )
        elif set(GO_LET_GAS).issubset(set(params_let_gas)):
            params_let_gas["l"] = params_let_gas.pop("lg")
            params_let_gas["e"] = params_let_gas.pop("eg")
            params_let_gas["t"] = params_let_gas.pop("tg")
            gasoil.add_LET_gas(**params_let_gas)
            usedparams = usedparams.union(set(params_let_gas.keys()))
            logger.info(
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
                params_corey_oil["kroend"] = params_corey_oil.pop("krogend")
            gasoil.add_corey_oil(**params_corey_oil)
            logger.info(
                "Added Corey gas to GasOil object from parameters %s",
                str(params_corey_oil.keys()),
            )
        elif set(GO_LET_OIL).issubset(set(params_let_oil)):
            params_let_oil["l"] = params_let_oil.pop("log")
            params_let_oil["e"] = params_let_oil.pop("eog")
            params_let_oil["t"] = params_let_oil.pop("tog")
            if "krogend" in params_corey_oil:
                params_let_oil["kroend"] = params_let_oil.pop("krogend")
            gasoil.add_LET_oil(**params_let_oil)
            logger.info(
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
        """
        if not params:
            params = dict()
        if not isinstance(params, dict):
            raise TypeError("Parameter to create_water_oil_gas must be a dictionary")

        check_deprecated(params)

        # For case insensitiveness, all keys are converted to lower case:
        params = {key.lower(): value for (key, value) in params.items()}

        if sufficient_water_oil_params(params):
            wateroil = PyscalFactory.create_water_oil(params)
        else:
            logger.info("No wateroil parameters. Assuming only gas-oil in wateroilgas")
            wateroil = None

        if sufficient_gas_oil_params(params):
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

        wog_low = PyscalFactory.create_water_oil_gas(params["low"])
        if not wog_low.selfcheck():
            logger.error("Incomplete parameter set for low case")
            errored = True
        wog_base = PyscalFactory.create_water_oil_gas(params["base"])
        if not wog_base.selfcheck():
            logger.error("Incomplete parameter set for base case")
            errored = True
        wog_high = PyscalFactory.create_water_oil_gas(params["high"])
        if not wog_high.selfcheck():
            logger.error("Incomplete parameter set for high case")
            errored = True
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

        All strings in CASE column are converted to lowercase.

        Args:
            inputfile (str or pd.DataFrame): Filename for XLSX or CSV file, or a
                pandas DataFrame.
            sheet_name (str): Sheet-name, only used when loading xlsx files.
        Returns:
            pd.DataFrame. To be handed over to pyscal list factory methods.
        """
        if isinstance(inputfile, str) and os.path.exists(inputfile):
            if inputfile.lower().endswith("csv"):
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
            input_df["CASE"] = [case_str.lower() for case_str in input_df["CASE"]]
            if input_df["CASE"].isnull().sum() > 0:
                logger.error("Found not-a-number in the CASE column. This could be due")
                logger.error("merged cells in XLSX, which is not supported.")
                raise ValueError
            if set(input_df["CASE"].unique()) != set(["low", "base", "high"]):
                logger.error(
                    "Contents of CASE-column must be exactly low, base and high"
                )
                logger.error("You provided %s", str(input_df["CASE"].unique()))
                raise ValueError

        # Add the SATNUM index to the TAG column
        if "TAG" not in input_df:
            input_df["TAG"] = ""
        input_df["TAG"].fillna(value="", inplace=True)  # Empty cells to empty string.
        input_df["TAG"] = (
            "SATNUM " + input_df["SATNUM"].astype(str) + " " + input_df["TAG"]
        )

        # Check that we are able to make something out of the first row:
        firstrow = input_df.loc[0, :]
        if not sufficient_water_oil_params(firstrow) and not sufficient_gas_oil_params(
            firstrow
        ):
            logger.error("Can't make neither WaterOil or GasOil from the given data.")
            logger.error("Check documentation for what you need to supply")
            logger.error("You provided the columns %s", str(input_df.columns))
            raise ValueError
        return input_df.sort_values("SATNUM")

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
            scal_l.append(
                PyscalFactory.create_scal_recommendation(
                    scalinput.loc[satnum, :].to_dict(orient="index"), h=h
                )
            )
        return scal_l

    @staticmethod
    def create_pyscal_list(relperm_params_df, h=None):
        """Create wateroilgas, wateroil or gasoil list
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
        if water_oil and gas_oil:
            return PyscalFactory.create_wateroilgas_list(relperm_params_df, h)
        elif water_oil:
            return PyscalFactory.create_wateroil_list(relperm_params_df, h)
        elif gas_oil:
            return PyscalFactory.create_gasoil_list(relperm_params_df, h)
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


def sufficient_water_oil_params(params):
    """Determine if the supplied parameters are sufficient for
    attempting creating a WaterOil object.

    In the factory context, relying on the defaults in the API
    is not allowed, as that would leave the tasks for the factory
    undefined (Corey or LET, and which pc?)

    Args:
        params: dict

    Returns:
        True if a WaterOil object should be attempted constructed
        (but no guarantee for validity of numerical values)
    """
    # pylint: disable=C1801
    # For case insensitiveness, all keys are converted to lower case:
    params = {key.lower(): value for (key, value) in params.items()}

    wo_corey_params = slicedict(params, set(WO_COREY_WATER + WO_COREY_OIL))
    if len(wo_corey_params) == 1:
        raise ValueError("Insufficient Corey parameters for WaterOil")
    if len(wo_corey_params) >= 2:
        return True
    wo_let_params = slicedict(params, set(WO_LET_WATER + WO_LET_OIL + WO_LET_OIL_ALT))
    if 0 < len(wo_let_params) < 6:
        raise ValueError("Insufficient LET parametres for WaterOil")
    if len(wo_let_params) >= 6:
        return True
    return False


def sufficient_gas_oil_params(params):
    """Determine if the supplied parameters are sufficient for
    attempting at creating a GasOil object.

    In the factory context, relying on the defaults in the API
    is not allowed, as that would leave the tasks for the factory
    undefined (Corey or LET, and which pc?)

    Args:
        params: dict

    Returns:
        True if a GasOil object should be attempted constructed
        (but no guarantee for validity of numerical values)
    """
    # pylint: disable=C1801
    # For case insensitiveness, all keys are converted to lower case:
    params = {key.lower(): value for (key, value) in params.items()}

    go_corey_params = slicedict(params, set(GO_COREY_GAS + GO_COREY_OIL))
    if len(go_corey_params) == 1:
        raise ValueError("Insufficient Corey parameters for GasOil")
    if len(go_corey_params) >= 2:
        return True
    go_let_params = slicedict(params, set(GO_LET_GAS + GO_LET_OIL))
    if 0 < len(go_let_params) < 6:  # noqa
        raise ValueError("Insufficient LET parameters for GasOil")
    if len(go_let_params) >= 6:
        return True
    return False


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
                "The key %s to PyscalFactory is deprecated and will be ignored", key
            )
