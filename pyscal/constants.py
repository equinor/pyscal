# Number of different Sw values within [0,1] we allow
# This is used to create integer indices of Sw, since Floating Point
# indices are flaky in Pandas (and in general on computers)
SWINTEGERS = 10000

# Used as "a small number" for ensuring no floating point
# comparisons/errors pop up.  You cannot have the h parameter less
# than this when generating relperm tables
EPSILON = 1e-08

