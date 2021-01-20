"""
Integration test for mapca.
"""

import os.path as op

import nibabel as nib
import numpy as np
import pytest

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


def test_integration(skip_integration):

    if skip_integration:
        pytest.skip('Skipping integration test')

    # Import data
    data_img = nib.load(op.join(get_resources_path(), "data.nii.gz"))
    mask_img = nib.load(op.join(get_resources_path(), "mask.nii.gz"))
    u, s, varex_norm, v = ma_pca(data_img, mask_img, normalize=True)

    voxel_comp_weights = np.load(op.join(get_resources_path(), "voxel_comp_weights.npy"))
    varex = np.load(op.join(get_resources_path(), "varex.npy"))
    varex_norm_test = np.load(op.join(get_resources_path(), "varex_norm.npy"))
    comp_ts = np.load(op.join(get_resources_path(), "comp_ts.npy"))

    assert np.allclose(voxel_comp_weights, u)
    assert np.allclose(varex, s)
    assert np.allclose(varex_norm_test, varex_norm)
    assert np.allclose(comp_ts, v)
