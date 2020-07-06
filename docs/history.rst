History
=======

This package was developed internally at Equinor starting in 2017
based on several internal codes in Python and other
languages. Open sourced (LGPLv3) in 2019.

Release notes
=============

.. Release note sections:
   New features
   Improvements
   Bugfixes
   Deprecations
   Dependencies
   Miscellaneous

Detailed release history (with minor releases) at

  * https://github.com/equinor/pyscal/releases

v0.6.0
------
**Deprecations**
  - kromax is no longer in use by pyscal. The oil relperm curve is anchored
    at kroend (and krowend and krogend are renamed to kroend accordingly, as
    they cannot be different any longer)
**New features**
  - Support for GasWater.

v0.5.0
------
**New features**
  - Adds saturation points between sorw and 1 in WaterOil saturation tables.

v0.4.0
------

**New features**
  - Added command line tool

v0.3.0
------

**Improvements**
  - Rewritten interpolation code

v0.2.0
------
**New features**
  - Includes PyscalFactory for creating objects.

v0.1.0
------

**Miscellaneous**
  - first open source version
