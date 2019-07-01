"""Factory functions for creating various pyscal objects"""

import logging

from pyscal import WaterOil, GasOil, WaterOilGas


def slicedict(dct, keys):
    """Slice a dictionary for a set of keys.
    Keys not existing will be ignored.
    """
    return dict((k, dct[k]) for k in keys if k in dct)


# Sets of dictionary keys, presence of these
# in incoming dictionary determines the codepaths. These must
# again match the API of the functions downstream in e.g. WaterOil
wo_endpoints = ["swirr", "swl", "swcr", "sorw"]
wo_corey_water = ["nw"]
wo_water_endpoints = ["krwmax", "krwend"]
wo_corey_oil = ["now"]
wo_let_water = ["Lw", "Ew", "Tw"]
wo_let_oil = ["Low", "Eow", "Tow"]
wo_oil_endpoints = ["kromax", "kroend"]


class PyscalFactory(object):
    def createWaterOil(self, params):

        usedparams = set()
        # No requirements to the base objects, defaults are ok.
        wo = WaterOil(**slicedict(params, wo_endpoints + ["h"]))
        usedparams = usedparams.union(
            set(slicedict(params, wo_endpoints + ["h"]).keys())
        )

        # Water curve
        params_corey_water = slicedict(params, wo_corey_water + wo_water_endpoints)
        if wo_corey_water[0] in params_corey_water:
            wo.add_corey_water(**params_corey_water)
            logging.info("Added Corey water to WaterOil object")
            usedparams = usedparams.union(set(params_corey_water.keys()))
        elif len(slicedict(params, wo_let_water)) == 3:
            wo.add_let_water(**slicedict(params, wo_let_water))
            logging.info("Added LET water to WaterOil object")
        else:
            logging.warning(
                "Missing or ambiguous parameters for water curve in WaterOil object"
            )

        # Oil curve:
        if len(slicedict(params, wo_corey_oil)) == 1:
            wo.add_corey_water(**slicedict(params, wo_corey_oil))
            logging.info("Added Corey water to WaterOil object")
        elif len(slicedict(params, wo_let_oil)) == 3:
            wo.add_let_water(**slicedict(params, wo_let_oil))
            logging.info("Added LET water to WaterOil object")
        else:
            logging.warning(
                "Missing or ambiguous parameters for oil curve in WaterOil object"
            )

        return wo

    def createGasOil(params):
        go = GasOil()
        return go

    def createWaterOilGas(params):
        wog = WaterOilGas()
        return wog

    def createSCALrecommendation(params):
        # Allow YAML-object?
        wog_low = WaterOilGas(params["low"])
        wog_base = WaterOilGas(params["base"])
        wog_high = WaterOilGas(params["high"])
        # Should we search for 'h' in the incoming dict?
        h_dict = dict(h=0.02)
        scal = SCALrecommendation(wog_low, wog_base, wog_high, **h_dict)
        return scal

    def createSCALrecommendationList(yamlobject):
        """Reserved"""

    def createWOGList():
        """Reserved for future implementation"""
        pass
