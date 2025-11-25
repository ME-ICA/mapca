"""Unit tests for utils."""

import numpy as np
from pytest import raises
from scipy.signal import detrend
from scipy.signal.windows import parzen
from scipy.stats import kurtosis

from mapca.utils import _autocorr, _eigensp_adj, _icatb_svd, _subsampling, ent_rate_sp


def test_autocorr():
    """Unit test on _autocorr function."""
    test_data = np.array([1, 2, 3, 4])
    test_result = np.array([30, 20, 11, 4])
    autocorr = _autocorr(test_data)
    assert np.array_equal(autocorr, test_result)


def test_parzen_win():
    """Test parzen gives expected output."""
    test_npoints = 3
    test_result = np.array([0.07407407, 1, 0.07407407])
    win = parzen(test_npoints)
    assert np.allclose(win, test_result)

    test_npoints = 1
    win = parzen(test_npoints)
    assert win == 1


def test_ent_rate_sp():
    """Check that ent_rate_sp runs correctly, i.e. returns a float."""
    test_data = np.random.rand(200, 10, 10)
    ent_rate = ent_rate_sp(test_data, 0)
    ent_rate = ent_rate_sp(test_data, 1)
    assert isinstance(ent_rate, float)
    assert ent_rate.ndim == 0
    assert ent_rate.size == 1

    # Checks ValueError with std = 0
    test_data = np.ones((200, 10, 10))
    with raises(ValueError) as errorinfo:
        ent_rate = ent_rate_sp(test_data, 1)
    assert "Divide by zero encountered" in str(errorinfo.value)

    # Checks ValueError with incorrect matrix dimensions
    test_data = np.ones((200, 10, 10, 200))
    with raises(ValueError) as errorinfo:
        ent_rate = ent_rate_sp(test_data, 1)
    assert "Incorrect matrix dimensions" in str(errorinfo.value)


def test_subsampling():
    """Unit test for subsampling function."""
    # 1D input
    test_data = np.array([1])
    sub_data = _subsampling(test_data, sub_depth=2)
    assert sub_data.shape == (1,)
    assert np.all(sub_data == test_data[::2])
    # 2D input
    test_data = np.random.rand(6, 9)
    sub_data = _subsampling(test_data, sub_depth=3)
    assert sub_data.shape == (2, 3)
    assert np.all(sub_data == test_data[::3, ::3])
    # 3D input
    test_data = np.random.rand(2, 3, 4)
    sub_data = _subsampling(test_data, sub_depth=2)
    assert sub_data.shape == (1, 2, 2)
    assert np.all(sub_data == test_data[::2, ::2, ::2])


def test_icatb_svd():
    """Unit test for icatb_svd function."""
    test_data = np.diag(np.random.rand(5))
    v, lambda_var = _icatb_svd(test_data)
    assert np.allclose(np.sum(v, axis=0), np.ones((5,)))


def test_eigensp_adj():
    """Unit test for eigensp_adj function."""
    test_eigen = np.array([0.9, 0.5, 0.2, 0.1, 0])
    n_effective = 2
    test_result = np.array([0.13508894, 0.11653465, 0.06727316, 0.05211424, 0.0])
    lambd_adj = _eigensp_adj(test_eigen, n_effective, p=test_eigen.shape[0])
    assert np.allclose(lambd_adj, test_result)


def test_kurtosis():
    """Generate data."""
    test_data = np.array([[-10, 2, 500, 0, -0.4], [-4, -200, -40, 0.1, 90]]).T

    # Run scipy function
    kurt_scipy = kurtosis(test_data, axis=0, fisher=True)
    kurt_scipy[kurt_scipy < 0] = 0
    kurt_scipy = np.expand_dims(kurt_scipy, 1)

    # Calculate kurtosis like GIFT
    kurt_gift = np.zeros((test_data.shape[1], 1))
    for i in range(test_data.shape[1]):
        data_norm = detrend(test_data[:, i], type="constant")
        data_norm /= np.std(data_norm)
        kurt_gift[i] = np.mean(data_norm**4) - 3
    kurt_gift[kurt_gift < 0] = 0

    assert np.allclose(kurt_gift, kurt_scipy)
