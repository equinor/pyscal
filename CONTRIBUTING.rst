Contributing to pyscal
=========================

* The code is hosted on https://github.com/equinor/pyscal
* Submit bugs on github
* Pull requests are welcome!


Create pull request
-------------------

1. Fork the pyscal repository from the Equinor repository to your GitHub
   user account.

2. Clone your fork locally:
  
  .. code-block:: bash
  
        git clone git@github.com:your_name_here/pyscal
        cd pyscal
        git remote add upstream git@github.com:equinor/pyscal
        git remote -v
        # origin	git@github.com:your_name_here/pyscal (fetch)
        # origin	git@github.com:your_name_here/pyscal (push)
        # upstream	git@github.com:equinor/pyscal (fetch)
        # upstream	git@github.com:equinor/pyscal (push)

4. Install your forked copy into a local venv:

  .. code-block:: bash
  
        python -m venv ~/venv/pyscal
        source ~/venv/pyscal/bin/activate
        pip install -U pip
        pip install -e ".[tests,docs]"
  
5. Run the tests to ensure everything works:

  .. code-block:: bash
  
        pytest -n auto

6. Create a branch for local development:

  .. code-block:: bash
  
        git checkout -b name-of-your-bugfix-or-feature
  
Now you can make your changes locally.

7. When you're done making changes, check that your changes pass ruff, mypy and the
   tests:

  .. code-block:: bash
  
        ruff check .
        ruff format .
        mypy src/pyscal
        pytest -n auto

In addition, it is recommended to use pylint to improve code quality

   .. code-block:: bash

        pylint src

Deviations from default (strict) pylint are stored in ``.pylintrc`` at root level,
or as comments in the file e.g. ``# pylint: disable=broad-except``.
  
Only use deviations when e.g. ruff and pylint are in conflict, or if conformity with
pylint would clearly make the code worse or not work at all. Do not use it to
increase pylint score.

8. Commit your changes and push your branch to GitHub:

  .. code-block:: bash
  
        git add file1.py file2.py
        git commit -m "Add some feature"
        git push origin name-of-your-bugfix-or-feature

9. Submit a pull request through GitHub.


Building documentation
----------------------

To build the documentation for pyscal run the following command:

  .. code-block:: bash

        python docs/make_plots.py
        sphinx-build -b html docs ./build/sphinx/html

And now you can find the start page of the documentation in the
build folder: ``build/sphinx/html/index.html``.
