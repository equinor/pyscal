from setuptools import setup

setup(
    name="pyscal",
    version="0.0.1",
    description="Generate relperm include files for Eclipse 100",
    url="http://github.com/equinor/pyscal",
    author="Håvard Berland",
    author_email="havb@equinor.com",
    license="LGPLv3",
    packages=["pyscal"],
    zip_safe=False,
    entry_points={"console_scripts": []},
)
