"""
mapca: A Python implementation of the moving average principal components analysis methods from
GIFT.
"""

import warnings

from .due import Doi, due
from .info import (__author__, __copyright__, __credits__, __description__,
                   __email__, __license__, __longdesc__, __maintainer__,
                   __packagename__, __status__, __url__, __version__)
from .mapca import ma_pca

# cmp is not used, so ignore nipype-generated warnings
warnings.filterwarnings("ignore", r"cmp not installed")

__all__ = [
    "ma_pca",
]
