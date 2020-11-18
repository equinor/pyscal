
Contributing to pyscal
=========================

* The code is hosted on https://github.com/equinor/pyscal
* Submit bugs on github
* Pull requests are welcome!

Code style
----------

* Use the black formatter to format your code

  * ``pip install black``
  * ``black <modulename.py>`` -- must be done prior to any pull request.

* Use flake8 code checker

  * ``pip install flake8``
  * ``flake8 src tests`` must pass before any pull request is accepted
  * Exceptions are listed in ``setup.cfg``

* Use pylint to improve coding

  * ``pip install pylint``
  * Then run ``pylint pyscal``
  * Deviations from default (strict) pylint are stored in ``.pylintrc`` at root level,
    or as comments in the file e.g. ``# pylint: disable=broad-except``.
  * Only use deviations when e.g. black and pylint are in conflict, or if conformity with
    pylint would clearly make the code worse or not work at all. Do not use it to
    increase pylint score.

* All code must be throroughly tested with ``pytest``.

Building documentation
----------------------

Install the development requirements::

  pip install .[tests]

Then, to build the documentation for pyscal run the following command::

  python setup.py build_sphinx

And now you can find the start page of the documentation in the
build folder: ``build/sphinx/html/index.html``
