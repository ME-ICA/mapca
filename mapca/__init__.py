"""
mapca: A Python implementation of the moving average principal components analysis methods from
GIFT.
"""

from .due import due, Doi

from .info import (
    __version__,
    __author__,
    __copyright__,
    __credits__,
    __license__,
    __maintainer__,
    __email__,
    __status__,
    __url__,
    __packagename__,
    __description__,
    __longdesc__,
)

from .mapca import ma_pca

import warnings

# cmp is not used, so ignore nipype-generated warnings
warnings.filterwarnings("ignore", r"cmp not installed")

__all__ = [
    "ma_pca",
]
