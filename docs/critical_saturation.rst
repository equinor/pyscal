Auditing critical saturation in exported tables
===============================================

Three different quantities can matter when reviewing a water relative
permeability table:

* The ``swcr`` supplied to pyscal defines the critical saturation used to
  construct the curve. It is reported in the table comments.
* The last tabulated saturation with a printed ``KRW`` of exactly zero can
  be higher: a small positive value can disappear at the output precision.
* Applying a positive numerical cutoff to the printed values can move the
  apparent endpoint further. This is a separate effect from rounding.

The following synthetic Corey example illustrates the distinction using the
actual ``SWOF()`` output. It is a diagnostic example, not a simulator or an
endpoint-scaling implementation.

.. literalinclude:: examples/critical_saturation.py
   :language: python

Run the :download:`script <examples/critical_saturation.py>` with Python. The
specified ``swcr`` and last in-memory zero are both 0.057, the last printed
zero is 0.064, and the last printed value at or below the chosen cutoff of
``1e-6`` is at 0.074. These numbers describe this example, not universal
endpoints or simulator defaults. The script uses an inclusive ``<=`` rule;
the exact rule and cutoff of a particular simulator must be checked separately.

How to use the comparison
-------------------------

Inspect exported tables when checking endpoint consistency, particularly
before endpoint scaling. Preserve the specified curve parameters and the
exported values as separate records. A difference between them is not by
itself evidence that the original curve parameters are wrong.

Increasing output precision can reduce the loss of small positive values,
but does not remove a consumer's positive cutoff. Changing precision or
saturation spacing also interacts with monotonicity handling, so neither
should be changed uncritically. This example leaves all production defaults
unchanged.

It does not establish the effect on a scaled curve or on simulated flow.
Those require the actual consumer settings and separate scaled/unscaled
comparisons. See the discussion in
`issue 469 <https://github.com/equinor/pyscal/issues/469>`_.
