Introduction to pyscal
======================

.. image:: images/pyscal-logo.png
   :width: 200

*pyscal* is a Python module for creating `relative permeability`_ input
curves for Eclipse, Flow (OPM) and Nexus, either directly from parametrizations (Corey
or LET) or from a SCAL recommendation parameter set.  

Alternatively, interpolated curves can be generated, that are within
a set of other curves, for example from a SCAL recommendation from low
to high. This enables history match both directly on
curve parameters (Corey/LET), or on interpolation parameters.

Relative permeability data can also be parsed from tabulated data and
then used for interpolation.

Capillary pressure is supported through a selected number of parametrizations.

Objects:
~~~~~~~~

WaterOil
  represents the data for water-oil relative permeability and
  capillary pressure. Essentially the data for SWOF plus metadata.

GasOil
  ditto for gas-oil relative permeability

WaterOilGas
  container object for one ``WaterOil`` and one ``GasOil``. Useful
  for making SOF3 output, and for ensuring endpoint consistency
  in three-phase simulations.

SCALrecommendation
  container object for three ``WaterOilGas`` objects which are tagged
  as low, base and high. Useful for interpolating between low and high, 
  going from -1 (low) through 0 (base) to 1 (high).

PyscalFactory
  Contains convenience functions for initializing the above objects from
  Python dictionaries. 


.. _relative permeability: http://en.wikipedia.org/wiki/Relative_permeability
