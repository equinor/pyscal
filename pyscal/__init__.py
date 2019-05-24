# -*- coding: utf-8 -*-
"""A module for creating relative permeability input curves for
Eclipse and Nexus, either directly from parametrizations or from a
SCAL recommendation parameter set. Curves can be interpolated within a
SCAL recommendation. History match can be done directly on curve
parameters, or on interpolation parameters.

Both LET and Corey are supported. One table can have LET for krw and
Corey for kro if so is wished.

Output is in SWOF/SGOF for Eclipse, and WOTABLE/GOTABLE for Nexus

Errors in the tables are catched and reported, similiar to Eclipse
error checking.

Missing:
 * Hysteresis
 * Alternative capillary pressure functions
 * Exception handling

Classes:
 * SCALrecommendation (contains three OilWaterGas objects)
 * OilWaterGas (holds one WaterOil and one GasOil object)
 * WaterOil (representing krw and krow for one facies/satnum)
 * GasOil (representing krg and krog for one facies/satnum)

History matching relative permeability can be done in two ways:
 1: Interpolation between low/base/high in a SCALrecommendation object,
    you may choose correlated parameter for wateroil and oilgas
    if you prefer.
 2: Direct matching on parameters that goes into WaterOil/GasOil,
    supposedly difficult if you use LET.

For multiple SATNUMs, use one SCALrecommendation object for each and
loop over your SATNUMs. See example in the test functions at the
bottom

Author: Håvard Berland, havb@statoil.com, September 2017

"""

from .wateroil import WaterOil
from .wateroilgas import WaterOilGas
from .gasoil import GasOil
from .scalrecommendation import SCALrecommendation


# Number of different Sw values within [0,1] we allow
# This is used to create integer indices of Sw, since Floating Point
# indices are flaky in Pandas (and in general on computers)
SWINTEGERS = 10000

# Used as "a small number" for ensuring no floating point
# comparisons/errors pop up.  You cannot have the h parameter less
# than this when generating relperm tables
epsilon = 1e-08
