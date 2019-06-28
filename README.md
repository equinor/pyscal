# pyscal [![Build Status](https://travis-ci.com/equinor/pyscal.svg?branch=master)](https://travis-ci.com/equinor/pyscal) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/11d947d518bd41729dc104d24fce33cd)](https://www.codacy.com/app/berland/pyscal?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=equinor/pyscal&amp;utm_campaign=Badge_Grade)

Python module for relative permeability/SCAL support in reservoir simulation

## Documentation

*   <http://equinor.github.io/pyscal>

## Feature overview

*   API to create relative permeability curves through correlations or
    tables

*   Similar for capillary pressure.

*   Consistency checks for three-phase setups, makes your oil-water
    tables and gas-oil tables compatible

*   Support for handling uncertainty, doing book-keeping for low, base
    and high cases, and the possiblity to interpolate between these
    cases using a number from -1 to +1.

## Scripts

There will eventually be some end-user scripts for this module

*   create_relperm.py - will read configation from Excel worksheets with
    parameters, and produce Eclipse include files

*   interpolate_relperm.py - reads low-base-high Eclipse include files,
    and interpolates between them

## Library usage

Illustrative example of how to produce a SWOF include file for Eclipse 
with a Corey relative permeability curve

```python
from pyscal import WaterOil

wo = WaterOil(h=0.1, sorw=0.05, swl=0.1)
wo.add_corey_water(nw=2)
wo.add_corey_oil(now=3)
print(wo.SWOF())
```
which will produce the string
```
SWOF
--
-- Sw Krw Krow Pc
-- swirr=0 swl=0.1 swcr=0.1 sorw=0.05
-- Corey krw, nw=2, krwend=1, krwmax=1
-- Corey krow, now=3, kroend=1, kromax=1
-- krw = krow @ sw=0.46670
-- Zero capillary pressure
0.1000000 0.0000000 1.0000000 0
0.2000000 0.0138408 0.6869530 0
0.3000000 0.0553633 0.4471809 0
0.4000000 0.1245675 0.2709139 0
0.5000000 0.2214533 0.1483818 0
0.6000000 0.3460208 0.0698148 0
0.7000000 0.4982699 0.0254427 0
0.8000000 0.6782007 0.0054956 0
0.9000000 0.8858131 0.0002035 0
0.9500000 1.0000000 0.0000000 0
1.0000000 1.0000000 0.0000000 0
/
```
