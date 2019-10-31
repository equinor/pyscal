"""Factory functions for creating various pyscal objects"""

import logging

from pyscal import WaterOil, GasOil, WaterOilGas, SCALrecommendation


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
WO_OIL_ENDPOINTS = ["kromax", "kroend"]
WO_SIMPLE_J = ["a", "b", "poro_ref", "perm_ref", "drho"]  # "g" is optional
WO_NORM_J = ["a", "b", "poro", "perm", "sigma_costau"]

GO_INIT = ["swirr", "sgcr", "sorg", "swl", "krgendanchor", "h", "tag"]
GO_COREY_GAS = ["ng"]
GO_GAS_ENDPOINTS = ["krgend", "krgmax"]
GO_COREY_OIL = ["nog"]
GO_OIL_ENDPOINTS = ["kroend"]
GO_LET_GAS = ["lg", "eg", "tg"]
GO_LET_OIL = ["log", "eog", "tog"]

WOG_INIT = ["swirr", "swl", "swcr", "sorw", "sorg", "sgcr", "h", "tag"]


class PyscalFactory(object):
    """Class for implementing the factory pattern for Pyscal objects

    The factory functions herein can take multiple parameter sets,
    determine what kind of parametrization to be used, and set up
    the full objects based on these parameters, instead of
    explicitly having to call the API for each task.

    Example:

        wo = WaterOil(sorw=0.05)
        wo.add_corey_water(nw=3)
        wo.add_corey_oil(now=2)
        # is equivalent to:
        wo = factory.create_water_oil(dict(sorw=0.05, nw=3, now=2))

    Parameter names to factory functions are case *insensitive*, while
    the add_*() parameters are not. This is becase the add_*() parameters
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
        """
        if not params:
            params = dict()
        if not isinstance(params, dict):
            raise TypeError("Parameter to create_water_oil must be a dictionary")

        # For case insensitiveness, all keys are converted to lower case:
        params = {key.lower(): value for (key, value) in params.items()}

        usedparams = set()
        # No requirements to the base objects, defaults are ok.
        wateroil = WaterOil(**slicedict(params, WO_INIT))
        usedparams = usedparams.union(set(slicedict(params, WO_INIT).keys()))
        logging.info(
            "Initialized WaterOil object from parameters %s", str(list(usedparams))
        )

        # Water curve
        params_corey_water = slicedict(params, WO_COREY_WATER + WO_WATER_ENDPOINTS)
        params_let_water = slicedict(params, WO_LET_WATER + WO_WATER_ENDPOINTS)
        if set(WO_COREY_WATER).issubset(set(params_corey_water)):
            wateroil.add_corey_water(**params_corey_water)
            usedparams = usedparams.union(set(params_corey_water.keys()))
            logging.info(
                "Added Corey water to WaterOil object from parameters %s",
                str(params_corey_water.keys()),
            )
        elif set(WO_LET_WATER).issubset(set(params_let_water)):
            params_let_water["l"] = params_let_water.pop("lw")
            params_let_water["e"] = params_let_water.pop("ew")
            params_let_water["t"] = params_let_water.pop("tw")
            wateroil.add_LET_water(**params_let_water)
            usedparams = usedparams.union(set(params_let_water.keys()))
            logging.info(
                "Added LET water to WaterOil object from parameters %s",
                str(params_let_water.keys()),
            )
        else:
            logging.warning(
                "Missing or ambiguous parameters for water curve in WaterOil object"
            )

        # Oil curve:
        params_corey_oil = slicedict(params, WO_COREY_OIL + WO_OIL_ENDPOINTS)
        params_let_oil = slicedict(
            params, WO_LET_OIL + WO_LET_OIL_ALT + WO_OIL_ENDPOINTS
        )
        if set(WO_COREY_OIL).issubset(set(params_corey_oil)):
            wateroil.add_corey_oil(**params_corey_oil)
            logging.info(
                "Added Corey water to WaterOil object from parameters %s",
                str(params_corey_oil.keys()),
            )
        elif set(WO_LET_OIL).issubset(set(params_let_oil)):
            params_let_oil["l"] = params_let_oil.pop("low")
            params_let_oil["e"] = params_let_oil.pop("eow")
            params_let_oil["t"] = params_let_oil.pop("tow")
            wateroil.add_LET_oil(**params_let_oil)
            logging.info(
                "Added LET water to WaterOil object from parameters %s",
                str(params_let_oil.keys()),
            )
        elif set(WO_LET_OIL_ALT).issubset(set(params_let_oil)):
            params_let_oil["l"] = params_let_oil.pop("lo")
            params_let_oil["e"] = params_let_oil.pop("eo")
            params_let_oil["t"] = params_let_oil.pop("to")
            wateroil.add_LET_oil(**params_let_oil)
            logging.info(
                "Added LET water to WaterOil object from parameters %s",
                str(params_let_oil.keys()),
            )
        else:
            logging.warning(
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
            logging.warning(
                "Missing or ambiguous parameters for capillary pressure in WaterOil object. Using zero."
            )
        if not wateroil.selfcheck():
            raise ValueError(
                ("Incomplete WaterOil object, some parameters " "missing to factory")
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
        translated to 'l')
        """
        if not params:
            params = dict()
        if not isinstance(params, dict):
            raise TypeError("Parameter to create_gas_oil must be a dictionary")

        # For case insensitiveness, all keys are converted to lower case:
        params = {key.lower(): value for (key, value) in params.items()}

        usedparams = set()
        # No requirements to the base objects, defaults are ok.
        gasoil = GasOil(**slicedict(params, GO_INIT))
        usedparams = usedparams.union(set(slicedict(params, GO_INIT).keys()))
        logging.info(
            "Initialized GasOil object from parameters %s", str(list(usedparams))
        )

        # Gas curve
        params_corey_gas = slicedict(params, GO_COREY_GAS + GO_GAS_ENDPOINTS)
        params_let_gas = slicedict(params, GO_LET_GAS + GO_GAS_ENDPOINTS)
        if set(GO_COREY_GAS).issubset(set(params_corey_gas)):
            gasoil.add_corey_gas(**params_corey_gas)
            usedparams = usedparams.union(set(params_corey_gas.keys()))
            logging.info(
                "Added Corey gas to GasOil object from parameters %s",
                str(params_corey_gas.keys()),
            )
        elif set(GO_LET_GAS).issubset(set(params_let_gas)):
            params_let_gas["l"] = params_let_gas.pop("lg")
            params_let_gas["e"] = params_let_gas.pop("eg")
            params_let_gas["t"] = params_let_gas.pop("tg")
            gasoil.add_LET_gas(**params_let_gas)
            usedparams = usedparams.union(set(params_let_gas.keys()))
            logging.info(
                "Added LET gas to GasOil object from parameters %s",
                str(params_let_gas.keys()),
            )
        else:
            logging.warning(
                "Missing or ambiguous parameters for gas curve in GasOil object"
            )

        # Oil curve:
        params_corey_oil = slicedict(params, GO_COREY_OIL + GO_OIL_ENDPOINTS)
        params_let_oil = slicedict(params, GO_LET_OIL + GO_OIL_ENDPOINTS)
        if set(GO_COREY_OIL).issubset(set(params_corey_oil)):
            gasoil.add_corey_oil(**params_corey_oil)
            logging.info(
                "Added Corey gas to GasOil object from parameters %s",
                str(params_corey_oil.keys()),
            )
        elif set(GO_LET_OIL).issubset(set(params_let_oil)):
            params_let_oil["l"] = params_let_oil.pop("log")
            params_let_oil["e"] = params_let_oil.pop("eog")
            params_let_oil["t"] = params_let_oil.pop("tog")
            gasoil.add_LET_oil(**params_let_oil)
            logging.info(
                "Added LET gas to GasOil object from parameters %s",
                str(params_let_oil.keys()),
            )
        else:
            logging.warning(
                "Missing or ambiguous parameters for oil curve in GasOil object"
            )
        if not gasoil.selfcheck():
            raise ValueError(
                ("Incomplete GasOil object, some parameters " "missing to factory")
            )

        return gasoil

    @staticmethod
    def create_water_oil_gas(params=None):
        """Create a WaterOilGas object from a dictionary of parameters

        Parameterization (Corey/LET) is inferred from presence
        of certain parameters in the dictionary.
        """
        if not params:
            params = dict()
        if not isinstance(params, dict):
            raise TypeError("Parameter to create_water_oil_gas must be a dictionary")

        # For case insensitiveness, all keys are converted to lower case:
        params = {key.lower(): value for (key, value) in params.items()}

        wateroil = PyscalFactory.create_water_oil(params)
        gasoil = PyscalFactory.create_gas_oil(params)
        wog_init_params = slicedict(params, WOG_INIT)
        wateroilgas = WaterOilGas(**wog_init_params)
        # The wateroilgas __init__ has already created WaterOil and GasOil objects
        # but we overwrite the references with newly created ones, this factory function
        # must then guarantee that they are compatible.
        wateroilgas.wateroil = wateroil
        wateroilgas.gasoil = gasoil
        if not wateroilgas.selfcheck():
            raise ValueError(
                ("Incomplete WaterOilGas object, " "some parameters missing to factory")
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
        the create_water_oil_gas() factory function;

           Lw, Ew, Tw, Low, Eow, Tow,
           Lg, Eg, Tg, Log, Eog, Tog,
           nw, now, ng, nog
           swirr, swl, sorw, sorg, sgcr

        For oil-water only, you may omit the LET parameters for gas and oil-gas
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

        errored = False
        wog_low = PyscalFactory.create_water_oil_gas(params["low"])
        if not wog_low.selfcheck():
            logging.error("Incomplete parameter set for low case")
            errored = True
        wog_base = PyscalFactory.create_water_oil_gas(params["base"])
        if not wog_base.selfcheck():
            logging.error("Incomplete parameter set for base case")
            errored = True
        wog_high = PyscalFactory.create_water_oil_gas(params["high"])
        if not wog_high.selfcheck():
            logging.error("Incomplete parameter set for high case")
            errored = True
        if errored:
            raise ValueError("Incomplete SCAL recommendation")

        scal = SCALrecommendation(wog_low, wog_base, wog_high, tag, h)
        return scal

    @staticmethod
    def create_scal_recommendation_list():
        """Reserved"""
        raise NotImplementedError

    @staticmethod
    def create_wog_list():
        """Reserved for future implementation"""
        raise NotImplementedError
