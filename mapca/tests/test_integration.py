"""
Integration test for mapca.
"""

import nibabel as nib
import os.path as op

from mapca.mapca import ma_pca


def get_resources_path():
    """Return the path to test resources.

    Returns the path to test resources, terminated with separator.
    Resources are kept outside package folder in "resources".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.

    Returns
    -------
    resource_path : str
        Absolute path to resources folder.
    """
    return op.abspath(op.join(op.dirname(__file__), "resources") + op.sep)


def test_integration():

    # Import data
    data_img = nib.load(op.join(get_resources_path(), "data.nii.gz"))
    mask_img = nib.load(op.join(get_resources_path(), "mask.nii.gz"))

    u, s, varex_norm, v = ma_pca(data_img, mask_img)
