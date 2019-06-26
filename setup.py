# coding: utf-8

from setuptools import setup

test_requirements = [
    'hypothesis',
    'pytest',
]


setup(
    name="pyscal",
    version="0.0.1",
    description="Generate relative permeability include files for Eclipse reservoir simulator",
    url="http://github.com/equinor/pyscal",
    author="Håvard Berland",
    author_email="havb@equinor.com",
    license="LGPLv3",
    packages=["pyscal"],
    zip_safe=False,
    entry_points={"console_scripts": []},
)
