# coding: utf-8

from os import path

from setuptools import setup
from setuptools_scm import get_version

# Read the contents of README.md, for PyPI
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md")) as f_handle:
    LONG_DESCRIPTION = f_handle.read()


def parse_requirements(filename):
    """Load requirements from a pip requirements file"""
    try:
        lineiter = (line.strip() for line in open(filename))
        return [line for line in lineiter if line and not line.startswith("#")]
    except IOError:
        return []


REQUIREMENTS = parse_requirements("requirements.txt")
TEST_REQUIREMENTS = parse_requirements("requirements_dev.txt")
SETUP_REQUIREMENTS = ["pytest-runner", "setuptools >=28", "setuptools_scm"]

setup(
    name="pyscal",
    use_scm_version={"write_to": "pyscal/version.py",},
    cmdclass={},
    description=(
        "Generate relative permeability include files for "
        "Eclipse reservoir simulator"
    ),
    entry_points={"console_scripts": ["pyscal = pyscal.pyscalcli:main"]},
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="http://github.com/equinor/pyscal",
    author="Håvard Berland",
    author_email="havb@equinor.com",
    license="LGPLv3",
    packages=["pyscal"],
    zip_safe=False,
    tests_suite="tests",
    install_requires=REQUIREMENTS,
    tests_require=TEST_REQUIREMENTS,
    setup_requires=SETUP_REQUIREMENTS,
)
