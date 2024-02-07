"""mapca: Python implementation of the moving average principal components analysis from GIFT."""

import warnings

from mapca.__about__ import __version__

from .mapca import MovingAveragePCA, ma_pca

# cmp is not used, so ignore nipype-generated warnings
warnings.filterwarnings("ignore", r"cmp not installed")

__all__ = [
    "ma_pca",
    "MovingAveragePCA",
    "__version__",
]
