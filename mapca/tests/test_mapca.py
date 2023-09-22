"""
Tests for mapca
"""

import nibabel as nib
import numpy as np
from nilearn import masking

from mapca.mapca import MovingAveragePCA, ma_pca


def test_ma_pca():
    """Check that ma_pca runs correctly with all three options"""

    n_timepoints = 200
    n_voxels = 20
    n_vox_total = n_voxels**3

    # Creates fake data to test with
    test_data = np.random.random((n_voxels, n_voxels, n_voxels, n_timepoints))
    time = np.linspace(0, 400, n_timepoints)
    freq = 1
    test_data = test_data + np.sin(2 * np.pi * freq * time)
    xform = np.eye(4) * 2
    test_img = nib.nifti1.Nifti1Image(test_data, xform)

    # Creates mask
    test_mask = np.ones((n_voxels, n_voxels, n_voxels))
    test_mask_img = nib.nifti1.Nifti1Image(test_mask, xform)

    # Testing AIC option
    u, s, varex_norm, v = ma_pca(test_img, test_mask_img, "aic")

    assert u.shape[0] == n_vox_total
    assert s.shape[0] == 1
    assert varex_norm.shape[0] == 1
    assert v.shape[0] == n_timepoints

    del u, s, varex_norm, v

    # Testing KIC option
    u, s, varex_norm, v = ma_pca(test_img, test_mask_img, "kic")

    assert u.shape[0] == n_vox_total
    assert s.shape[0] == 1
    assert varex_norm.shape[0] == 1
    assert v.shape[0] == n_timepoints

    del u, s, varex_norm, v

    # Testing MDL option
    u, s, varex_norm, v = ma_pca(test_img, test_mask_img, "mdl")

    assert u.shape[0] == n_vox_total
    assert s.shape[0] == 1
    assert varex_norm.shape[0] == 1
    assert v.shape[0] == n_timepoints


def test_MovingAveragePCA():
    """Check that MovingAveragePCA runs correctly with "aic" option."""

    N_TIMEPOINTS = 200
    N_VOXELS = 20  # number of voxels in each dimension

    # Create fake data to test with
    test_data = np.random.random((N_VOXELS, N_VOXELS, N_VOXELS, N_TIMEPOINTS))
    time = np.linspace(0, 400, N_TIMEPOINTS)
    freq = 1
    test_data = test_data + np.sin(2 * np.pi * freq * time)
    xform = np.eye(4) * 2
    test_img = nib.nifti1.Nifti1Image(test_data, xform)

    # Create mask
    test_mask = np.zeros((N_VOXELS, N_VOXELS, N_VOXELS), dtype=int)
    test_mask[5:-5, 5:-5, 5:-5] = 1
    test_mask_img = nib.nifti1.Nifti1Image(test_mask, xform)
    n_voxels_in_mask = np.sum(test_mask)

    test_data = masking.apply_mask(test_img, test_mask_img).T

    # Testing AIC option
    pca = MovingAveragePCA(criterion="mdl", normalize=True)
    u = pca.fit_transform(test_img, test_mask_img)
    assert pca.u_.shape[0] == n_voxels_in_mask
    assert pca.explained_variance_.shape[0] == 1
    assert pca.explained_variance_ratio_.shape[0] == 1
    assert pca.components_.T.shape[0] == N_TIMEPOINTS

    # Test other stuff
    pca2 = MovingAveragePCA(criterion="mdl", normalize=True)
    pca2.fit(test_img, test_mask_img)
    u2 = pca2.transform(test_img)
    assert np.array_equal(u2.get_fdata(), u.get_fdata())

    test_data_est = pca2.inverse_transform(u2, test_mask_img)
    assert test_data_est.shape == test_img.shape

    # Testing setting inputting a pre-defined subsampling depth
    pca3 = MovingAveragePCA(criterion="mdl", normalize=False)
    pca3.fit(test_img, test_mask_img, subsample_depth=2)
    assert pca3.subsampling_["calculated_IID_subsample_depth"] == 1
    assert pca3.subsampling_["used_IID_subsample_depth"] == 2
