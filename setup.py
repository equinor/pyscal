# coding: utf-8

from os import path

from setuptools import setup
import versioneer

# Read the contents of README.md, for PyPI
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md")) as f_handle:
    long_description = f_handle.read()

setup(
    name="pyscal",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Generate relative permeability include files for Eclipse reservoir simulator",
    entry_points={"console_scripts": ["pyscal = pyscal.pyscalcli:main"]},
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/equinor/pyscal",
    author="Håvard Berland",
    author_email="havb@equinor.com",
    license="LGPLv3",
    packages=["pyscal"],
    zip_safe=False,
)
