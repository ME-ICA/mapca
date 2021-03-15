# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Base module variables
"""
import importlib.util
import json
import os.path as op

from pathlib import Path

# Get version
spec = importlib.util.spec_from_file_location(
    "_version", op.join(op.dirname(__file__), "mapca/_version.py")
)
_version = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_version)

VERSION = _version.get_versions()["version"]
del _version

# Get list of authors from Zenodo file
with open(op.join(op.dirname(__file__), ".zenodo.json"), "r") as fo:
    zenodo_info = json.load(fo)
authors = [author["name"] for author in zenodo_info["creators"]]
author_names = []
for author in authors:
    if ", " in author:
        author_names.append(author.split(", ")[1] + " " + author.split(", ")[0])
    else:
        author_names.append(author)

# Get package description from README
# Since this file is executed from ../setup.py, the path to the README is determined by the
# location of setup.py.
readme_path = Path(__file__).parent.joinpath("README.md")
longdesc = readme_path.open().read()

# Fields
AUTHOR = "mapca developers"
COPYRIGHT = "Copyright 2020, mapca developers"
CREDITS = author_names
LICENSE = "GPL-2.0"
MAINTAINER = "Eneko Urunuela"
EMAIL = "e.urunuela@bcbl.eu"
STATUS = "Prototype"
URL = "https://github.com/me-ica/mapca"
PACKAGENAME = "mapca"
DESCRIPTION = (
    "A Python implementation of the moving average principal components analysis "
    "methods from GIFT."
)
LONGDESC = longdesc

DOWNLOAD_URL = "https://github.com/ME-ICA/{name}/archive/{ver}.tar.gz".format(
    name=PACKAGENAME, ver=VERSION
)

REQUIRES = [
    "nibabel>=2.5.1",
    "nilearn",
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

ENTRY_POINTS = {}

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
