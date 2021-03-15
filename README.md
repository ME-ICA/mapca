# mapca
A Python implementation of the moving average principal components analysis methods from GIFT

[![Latest Version](https://img.shields.io/pypi/v/mapca.svg)](https://pypi.python.org/pypi/mapca/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mapca.svg)](https://pypi.python.org/pypi/mapca/)
[![License](https://img.shields.io/badge/license-GPL--2.0-blue.svg)](https://opensource.org/licenses/GPL-2.0)
[![CircleCI](https://circleci.com/gh/ME-ICA/mapca.svg?style=shield)](https://circleci.com/gh/ME-ICA/mapca)
[![Codecov](https://codecov.io/gh/ME-ICA/mapca/branch/main/graph/badge.svg?token=GEKDT6R0B7)](https://codecov.io/gh/ME-ICA/mapca)
[![Average time to resolve an issue](http://isitmaintained.com/badge/resolution/ME-ICA/mapca.svg)](http://isitmaintained.com/project/ME-ICA/mapca "Average time to resolve an issue")
[![Percentage of issues still open](http://isitmaintained.com/badge/open/ME-ICA/mapca.svg)](http://isitmaintained.com/project/ME-ICA/mapca "Percentage of issues still open")
[![Join the chat at https://gitter.im/ME-ICA/mapca](https://badges.gitter.im/ME-ICA/mapca.svg)](https://gitter.im/ME-ICA/mapca?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

## About

`mapca` is a Python package that performs dimensionality reduction with principal component analysis (PCA) on functional magnetic resonance imaging (fMRI) data. It is a translation to Python of the dimensionality reduction technique used in the MATLAB-based [GIFT package](https://trendscenter.org/software/gift/) and introduced by Li et al. 2007[^1].

[^1]: Li, Y. O., Adali, T., & Calhoun, V. D. (2007). Estimating the number of independent components for functional magnetic resonance imaging data. Human Brain Mapping, 28(11), 1251–1266. https://doi.org/10.1002/hbm.20359
