# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Base module variables
"""
from pathlib import Path

readme_path = Path(__file__).parent.parent.joinpath("README.md")

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

__author__ = "mapca developers"
__copyright__ = "Copyright 2020, mapca developers"
__credits__ = ["Eneko Urunuela"]
__license__ = ""
__maintainer__ = "Eneko Urunuela"
__email__ = "e.urunuela@bcbl.eu"
__status__ = "Prototype"
__url__ = "https://github.com/me-ica/mapca"
__packagename__ = "mapca"
__description__ = (
    "A Python implementation of the moving average principal components analysis "
    "methods from GIFT."
)
__longdesc__ = readme_path.open().read()

DOWNLOAD_URL = "https://github.com/ME-ICA/{name}/archive/{ver}.tar.gz".format(
    name=__packagename__, ver=__version__
)

REQUIRES = [
    "nibabel>=2.5.1",
    "numpy>=1.15",
    "scikit-learn>=0.22",
    "scipy>=1.3.3",
]

TESTS_REQUIRES = [
    "codecov",
    "coverage<5.0",
    "flake8>=3.7",
    "pytest",
    "pytest-cov",
    "requests",
]

EXTRA_REQUIRES = {
    "dev": ["versioneer"],
    "doc": [
        "sphinx>=1.5.3",
        "sphinx_rtd_theme",
        "sphinx-argparse",
    ],
    "tests": TESTS_REQUIRES,
    "duecredit": ["duecredit"],
}

# Enable a handle to install all extra dependencies at once
EXTRA_REQUIRES["all"] = list(set([v for deps in EXTRA_REQUIRES.values() for v in deps]))

# Supported Python versions using PEP 440 version specifiers
# Should match the same set of Python versions as classifiers
PYTHON_REQUIRES = ">=3.5"

# Package classifiers
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
]
