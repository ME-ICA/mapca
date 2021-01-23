"""
Integration test for mapca.
"""
import nibabel as nib
import numpy as np
import pytest

from mapca.mapca import ma_pca


def test_integration(skip_integration, test_img, test_mask, test_ts, test_varex,
                     test_varex_norm, test_weights):

    if skip_integration:
        pytest.skip('Skipping integration test')

    # Import data
    data_img = nib.load(test_img)
    mask_img = nib.load(test_mask)
    # data_img = nib.load(op.join(get_resources_path(), "data.nii.gz"))
    # mask_img = nib.load(op.join(get_resources_path(), "mask.nii.gz"))
    u, s, varex_norm, v = ma_pca(data_img, mask_img, normalize=True)

    voxel_comp_weights = np.load(test_weights)
    varex = np.load(test_varex)
    v_norm = np.load(test_varex_norm)
    comp_ts = np.load(test_ts)

    assert np.allclose(voxel_comp_weights, u)
    assert np.allclose(varex, s)
    assert np.allclose(v_norm, varex_norm)
    assert np.allclose(comp_ts, v)
