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
WO_ENDPOINTS = ["swirr", "swl", "swcr", "sorw"]
WO_COREY_WATER = ["nw"]
WO_WATER_ENDPOINTS = ["krwmax", "krwend"]
WO_COREY_OIL = ["now"]
WO_LET_WATER = ["Lw", "Ew", "Tw"]  # Will translated to l, e and t in code below.
WO_LET_OIL = ["Low", "Eow", "Tow"]
WO_OIL_ENDPOINTS = ["kromax", "kroend"]


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
    """

    @staticmethod
    def create_water_oil(params):
        """Create a WaterOil object from a dictionary of parameters.

        Parametrization (Corey/LET) is inferred from presence
        of certain parameters in the dictionary.

        Don't rely on behaviour of you supply both Corey and LET at
        the same time.

        NB: the add_LET_* methods have the names 'l', 'e' and 't'
        in their signatures, which is not precise enough in this
        situations, so we require e.g. 'Lw' (which will be
        translated to 'l')
        """
        usedparams = set()
        # No requirements to the base objects, defaults are ok.
        wateroil = WaterOil(**slicedict(params, WO_ENDPOINTS + ["h"]))
        usedparams = usedparams.union(
            set(slicedict(params, WO_ENDPOINTS + ["h"]).keys())
        )
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
            params_let_water["l"] = params_let_water.pop("Lw")
            params_let_water["e"] = params_let_water.pop("Ew")
            params_let_water["t"] = params_let_water.pop("Tw")
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
        params_let_oil = slicedict(params, WO_LET_OIL + WO_OIL_ENDPOINTS)
        if set(WO_COREY_OIL).issubset(set(params_corey_oil)):
            wateroil.add_corey_water(**params_corey_oil)
            logging.info(
                "Added Corey water to WaterOil object from parameters %s",
                str(params_corey_oil.keys()),
            )
        elif set(WO_LET_OIL).issubset(set(params_let_oil)):
            params_let_oil["l"] = params_let_oil.pop("Low")
            params_let_oil["e"] = params_let_oil.pop("Eow")
            params_let_oil["t"] = params_let_oil.pop("Tow")
            wateroil.add_LET_oil(**params_let_oil)
            logging.info(
                "Added LET water to WaterOil object from parameters %s",
                str(params_let_oil.keys()),
            )
        else:
            logging.warning(
                "Missing or ambiguous parameters for oil curve in WaterOil object"
            )

        return wateroil

    def create_gas_oil(self, params):
        """foo"""
        gasoil = GasOil()
        return gasoil

    def create_water_oil_gas(self, params):
        """foo"""
        wateroilgas = WaterOilGas()
        return wateroilgas

    def create_scal_recommendation(self, params):
        """foo"""
        # Allow YAML-object?
        wateroil_low = WaterOilGas(params["low"])
        wateroil_base = WaterOilGas(params["base"])
        wateroil_high = WaterOilGas(params["high"])
        # Should we search for 'h' in the incoming dict?
        h_dict = dict(h=0.02)
        scal = SCALrecommendation(wateroil_low, wateroil_base, wateroil_high, **h_dict)
        return scal

    def create_scal_recommendation_list(self, yamlobject):
        """Reserved"""

    def create_wog_list(self):
        """Reserved for future implementation"""
        pass
