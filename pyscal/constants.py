"""Constants used for pyscal modules:

 * ``SWINTEGERS``: Number of different Sw values within [0,1] we allow
   This is used to create integer indices of Sw, since Floating Point
   indices are flaky in Pandas (and in general on computers)

 * ``EPSILON``: Used as "a small number" for ensuring no floating point
   comparisons/errors pop up.  You cannot have the h parameter less
   than this when generating relperm tables

 * ``MAX_EXPONENT``: Maximal number for exponents in relperm parametrizations.
   Used to avoid numerical instabilities. It could probably be much
   higher than the chosen number in most circumstances, but such high
   numbers should not be relevant for relative permeability
"""
SWINTEGERS: int = 10000

EPSILON: float = 1e-08

MAX_EXPONENT: float = 100
