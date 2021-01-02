"""
mapca: A Python implementation of the moving average principal components analysis methods from
GIFT.
"""

from .due import due, Doi
from ._version import get_versions
from .mapca import ma_pca

__version__ = get_versions()["version"]

import warnings

# cmp is not used, so ignore nipype-generated warnings
warnings.filterwarnings("ignore", r"cmp not installed")

__all__ = [
    "ma_pca",
    "__version__",
]

del get_versions
