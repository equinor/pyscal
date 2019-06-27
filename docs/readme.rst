Introduction to pyscal
======================

*pyscal* is a Python module for creating `relative permeability`_ input
curves for Eclipse and Nexus, either directly from parametrizations or
from a SCAL recommendation parameter set. Curves can be interpolated
within a SCAL recommendation. History match can be done directly on
curve parameters, or on interpolation parameters.

Both LET and Corey are supported. One table can have LET for krw and
Corey for kro if so is wished. Relative permeability data can also be
parsed from tabulated data.

Output is in SWOF/SGOF for Eclipse, and WOTABLE/GOTABLE for Nexus

Errors in the tables are catched and reported, similiar to Eclipse
error checking.

Classes:
 * SCALrecommendation (container for three OilWaterGas objects)
 * OilWaterGas (holds one WaterOil and one GasOil object)
 * WaterOil (representing krw and krow for one facies/satnum)
 * GasOil (representing krg and krog for one facies/satnum)

History matching relative permeability can be done in two ways:

1. Interpolation between low/base/high in a SCALrecommendation object,
   you may choose correlated parameter for wateroil and oilgas
   if you prefer.
2. Direct matching on parameters that goes into WaterOil/GasOil,
   supposedly difficult if you use LET.

For multiple SATNUMs, use one SCALrecommendation object for each and
loop over your SATNUMs.

.. _relative permeability: http://en.wikipedia.org/wiki/Relative_permeability
