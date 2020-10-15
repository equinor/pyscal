Introduction to pyscal
======================

.. image:: images/pyscal-logo.png
   :width: 200

*pyscal* is a Python tool and module for creating `relative permeability`_ input
curves for Eclipse, Flow (OPM) and Nexus. Curves are parametrized
using Corey or LET.

SCAL recommendations are curve sets consisting of "low", "base" and "high" curves,
each parametrized individually. Pyscal has support for interpolating between
such enveloping curves.

Pyscal can be used both for sensitivity tests on relative permeability parameters,
and for sensitivity tests of a SCAL recommendation. For history matching, matching
interpolation parameters in a SCAL recommendation is the recommended practice.

Relative permeability data can also be parsed from tabulated data and
then used for interpolation.

Capillary pressure is supported through a selected number of parametrizations.

Command line tool
-----------------

The command line tool is installed as ``pyscal`` and takes a table in a CSV file
or in an XLSX file as the main input. Each SATNUM is represented by one row of
data in the table.

Example use with CSV input for one SATNUM:

.. code-block:: console

    $ cat relperminput.csv  # Show example input file
    SATNUM, swl, sorw, Nw, Now
    1,      0.1, 0.05, 2, 3
    $ # Run pyscal on example input:
    $ pyscal relperminput.csv --delta_s 0.2 -o relperm.inc
    Written to relperm.inc

Exactly the same table in an XLSX file would give identical results. The output
from the pyscal command above will look like:

.. code-block:: console

    $ cat relperm.inc
    SWOF
    -- SATNUM 1
    -- pyscal: v0.7.X
    -- swirr=0 swl=0.1 swcr=0.1 sorw=0.05
    -- Corey krw, nw=2, krwend=1, krwmax=1
    -- Corey krow, now=3, kroend=1
    -- krw = krow @ sw=0.46856
    -- Zero capillary pressure
    -- SW     KRW       KROW      PC
    0.1000000 0.0000000 1.0000000 0
    0.3000000 0.0553633 0.4471809 0
    0.5000000 0.2214533 0.1483818 0
    0.7000000 0.4982699 0.0254427 0
    0.9000000 0.8858131 0.0002035 0
    0.9500000 1.0000000 0.0000000 0
    1.0000000 1.0000000 0.0000000 0
    /


The saturation step-length ``--delta_s`` was set artificially high for the sake
of the example. Leave it defaulted (0.01) for practical use.

Python API
----------

The corresponding API can be used directly for more control and for custom-made
solutions. An example recreating the the same table as above is given by:

.. code-block:: python

    from pyscal import WaterOil

    wo = WaterOil(h=0.2, sorw=0.05, swl=0.1)
    wo.add_corey_water(nw=2)
    wo.add_corey_oil(now=3)
    print(wo.SWOF())

which will give the same output as the example above.

See the full API for in-depth information.

Classes
^^^^^^^

WaterOil
  Represents the data for water-oil relative permeability and
  capillary pressure. Essentially the data for SWOF plus metadata.
  All tabular data is built up in the interal object member ``table``
  (a pandas DataFrame) which can be viewed for debugging. The
  object contains export functions for Eclipse keywords, SWOF, SWFN
  etc.

GasOil
  Ditto for gas-oil relative permeability

GasWater
  Ditto for gas-water relative permeability

WaterOilGas
  Container object for one ``WaterOil`` and one ``GasOil``. Useful
  for making SOF3 output, and for ensuring endpoint consistency
  in three-phase simulations. The objects members ``wateroil`` and
  ``gasoil`` refer to the contained objects. It is allowed to use
  this container for two-phase only, if used for oil-water, the
  gasoil reference would be ``None``.

SCALrecommendation
  Container object for three ``WaterOilGas`` objects which are tagged
  as low, base and high. Useful for interpolating between low and high,
  going from -1 (low) through 0 (base) to 1 (high).

PyscalFactory
  Contains convenience functions for initializing the above objects from
  Python dictionaries. If provided a table input (Pandas DataFrame, CSV-
  or XLSX-file), PyscalList objects are constructed.

PyscalList
  Container for a sequence of WaterOil, GasOil, GasWater, WaterOilGas or
  SCALrecommendation objects. Objects of this class can make up the entire
  relative permeability input to Eclipse through f.ex.  the function
  ``dump_family_1()``.

.. _relative permeability: http://en.wikipedia.org/wiki/Relative_permeability
