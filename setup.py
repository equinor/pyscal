from pathlib import Path

from setuptools import find_packages, setup

try:
    from sphinx.setup_command import BuildDoc

    cmdclass = {"build_sphinx": BuildDoc}
except ImportError:
    # sphinx not installed - do not provide build_sphinx cmd
    cmdclass = {}

# Read the contents of README.md, for PyPI
LONG_DESCRIPTION = (Path(__file__).parent / "README.md").read_text()

REQUIREMENTS = [
    "matplotlib",
    "numpy",
    "openpyxl",
    "pandas",
    "packaging",
    "scipy",
    "typing-extensions>=3.7; python_version<'3.8'",
    "xlrd",
]

TEST_REQUIREMENTS = Path("test_requirements.txt").read_text().splitlines()

SETUP_REQUIREMENTS = ["pytest-runner", "setuptools >= 28", "setuptools_scm"]
EXTRAS_REQUIRE = {"tests": TEST_REQUIREMENTS}

setup(
    name="pyscal",
    description=(
        "Generate relative permeability include files for "
        "Eclipse reservoir simulator"
    ),
    author="Håvard Berland",
    author_email="havb@equinor.com",
    url="http://github.com/equinor/pyscal",
    use_scm_version={"write_to": "pyscal/version.py"},
    project_urls=None,
    keywords="relative permeability, capillary pressure, reservoir simulation",
    license="LGPLv3",
    platforms="any",
    cmdclass=cmdclass,
    include_package_data=True,
    packages=find_packages(),
    package_dir=None,
    package_data={"pyscal": ["py.typed"]},
    py_modules=None,
    install_requires=REQUIREMENTS,
    entry_points={"console_scripts": ["pyscal = pyscal.pyscalcli:main"]},
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        (
            "License :: OSI Approved :: "
            "GNU Lesser General Public License v3 or later (LGPLv3+)"
        ),
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    zip_safe=False,
    test_suite="tests",
    setup_requires=SETUP_REQUIREMENTS,
    extras_require=EXTRAS_REQUIRE,
)
