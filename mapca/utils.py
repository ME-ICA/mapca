"""PCA based on Moving Average (stationary Gaussian) process."""

import logging

import numpy as np
from scipy.fftpack import fftn, fftshift
from scipy.linalg import svd
from scipy.signal import fftconvolve
from scipy.signal.windows import parzen
from scipy.stats import kurtosis

LGR = logging.getLogger(__name__)


def _autocorr(data):
    """
    Calculate the auto correlation of a given array.

    Parameters
    ----------
    data : array-like
        The array to calculate the autocorrelation of

    Returns
    -------
    u : ndarray
        The array of autocorrelations
    """
    u = np.correlate(data, data, mode="full")
    # Take upper half of correlation matrix
    return u[u.size // 2 :]


def ent_rate_sp(data, sm_window):
    """
    Calculate the entropy rate of a stationary Gaussian random process.

    Uses spectrum estimation with smoothing window.

    Parameters
    ----------
    data : ndarray
        Data to calculate the entropy rate of and smooth
    sm_window : boolean
        Whether there is a Parzen window to use

    Returns
    -------
    ent_rate : float
        The entropy rate

    Notes
    -----
    This function attempts to calculate the entropy rate following

    References
    ----------
    * Li, Y.O., Adalı, T. and Calhoun, V.D., (2007).
      Estimating the number of independent components for
      functional magnetic resonance imaging data.
      Human brain mapping, 28(11), pp.1251-1266.
    """
    dims = data.shape

    if data.ndim == 3 and min(dims) != 1:
        pass
    else:
        raise ValueError("Incorrect matrix dimensions.")

    # Normalize x_sb to be unit variance
    data_std = np.std(np.reshape(data, (-1, 1)))

    # Make sure we do not divide by zero
    if data_std == 0:
        raise ValueError("Divide by zero encountered.")
    data = data / data_std

    # Apply windows to 3D
    data_corr = fftconvolve(data, np.flip(data))

    # Create bias-correcting vectors
    v1 = np.hstack((np.arange(1, dims[0] + 1), np.arange(dims[0] - 1, 0, -1)))[np.newaxis, :]
    v2 = np.hstack((np.arange(1, dims[1] + 1), np.arange(dims[1] - 1, 0, -1)))[np.newaxis, :]
    v3 = np.arange(dims[2], 0, -1)

    vd = np.dot(v1.T, v2)
    vcu = np.zeros((2 * dims[0] - 1, 2 * dims[1] - 1, 2 * dims[2] - 1))
    for m3 in range(dims[2]):
        vcu[:, :, (dims[2] - 1) - m3] = vd * v3[m3]
        vcu[:, :, (dims[2] - 1) + m3] = vd * v3[m3]

    data_corr /= vcu

    if sm_window:
        m = [int(i) for i in np.ceil(np.array(dims) / 10)]

        # Get Parzen window for each spatial direction
        parzen_w_3 = np.zeros((2 * dims[2] - 1,))
        parzen_w_3[(dims[2] - m[2] - 1) : (dims[2] + m[2])] = parzen(2 * m[2] + 1)

        parzen_w_2 = np.zeros((2 * dims[1] - 1,))
        parzen_w_2[(dims[1] - m[1] - 1) : (dims[1] + m[1])] = parzen(2 * m[1] + 1)

        parzen_w_1 = np.zeros((2 * dims[0] - 1,))
        parzen_w_1[(dims[0] - m[0] - 1) : (dims[0] + m[0])] = parzen(2 * m[0] + 1)

        # Scale Parzen windows
        parzen_window_2d = np.dot(parzen_w_1[np.newaxis, :].T, parzen_w_2[np.newaxis, :])
        parzen_window_3d = np.zeros((2 * dims[0] - 1, 2 * dims[1] - 1, 2 * dims[2] - 1))
        for m3 in range(dims[2] - 1):
            parzen_window_3d[:, :, (dims[2] - 1) - m3] = np.dot(
                parzen_window_2d, parzen_w_3[dims[2] - 1 - m3]
            )
            parzen_window_3d[:, :, (dims[2] - 1) + m3] = np.dot(
                parzen_window_2d, parzen_w_3[dims[2] - 1 + m3]
            )
        # Apply 3D Parzen Window
        data_corr *= parzen_window_3d

    data_fft = abs(fftshift(fftn(data_corr)))
    data_fft[data_fft < 1e-4] = 1e-4

    # Estimation of the entropy rate
    ent_rate = 0.5 * np.log(2 * np.pi * np.exp(1)) + np.sum(
        np.log(abs((data_fft)))[:]
    ) / 2 / np.sum(abs(data_fft)[:])

    return ent_rate


def _est_indp_sp(data):
    """
    Estimate the effective number of independent samples.

    Bbased on the maximum entropy rate principle of stationary random process.

    Parameters
    ----------
    data : ndarray
        The data to have the number of samples estimated

    Returns
    -------
    n_iters : int
        Number of iterations required to estimate entropy rate
    ent_rate : float
        The entropy rate of the data

    Notes
    -----
    This function estimates the effective number of independent samples by omitting
    the least significant components with the subsampling scheme (Li et al., 2007)
    """
    dims = data.shape
    n_iters_0 = None

    for j in range(np.min(dims) - 1):
        data_sb = _subsampling(data, j + 1)
        ent_rate = ent_rate_sp(data_sb, 1)

        # Upper-bound.
        ent_ref = 1.41

        # If entropy rate of a subsampled Gaussian sequence reaches the upper bound
        # of the entropy rate, the subsampled sequence is an i.i.d. sequence.
        if ent_rate > ent_ref:
            n_iters_0 = j
            break

    if n_iters_0 is None:
        raise ValueError("Ill conditioned data, can not estimate " "independent samples.")
    n_iters = n_iters_0
    LGR.debug(
        f"Estimated the entropy rate of the Gaussian component with subsampling depth {j + 1}"
    )

    return n_iters, ent_rate


def _subsampling(data, sub_depth):
    """
    Subsampling the data evenly with space 'sub_depth'.

    Parameters
    ----------
    data : ndarray
        The data to be subsampled
    sub_depth : int
        The subsampling depth

    Returns
    -------
    out : ndarray
        Subsampled data
    """
    slices = [slice(None, None, sub_depth)] * data.ndim
    out = data[tuple(slices)]
    return out


def _kurtn(data):
    """
    Normalize kurtosis funtion so that for a Gaussian r.v. the kurtn(g) = 0.

    Parameters
    ----------
    data : ndarray
        The data to calculate the kurtosis of

    Returns
    -------
    kurt : (1:N) array-like
        The kurtosis of each vector in x along the second dimension. For
        tedana, this will be the kurtosis of each PCA component.
    """
    kurt = kurtosis(data, axis=0, fisher=True)
    kurt[kurt < 0] = 0
    kurt = np.expand_dims(kurt, 1)

    return kurt


def _icatb_svd(data, n_comps=None):
    """
    Run Singular Value Decomposition (SVD) on input data.

    Runs SVD and extracts the given number of components (n_comps).

    Parameters
    ----------
    data : array
        The data to compute SVD for
    n_comps : int
        Number of PCA components to be kept

    Returns
    -------
    V : 2D array
        Eigenvectors from SVD
    Lambda : float
        Eigenvalues
    """
    if not n_comps:
        n_comps = np.min((data.shape[0], data.shape[1]))

    _, lambda_var, vh = svd(data, full_matrices=False)

    # Sort eigen vectors in Ascending order
    v = vh.T
    lambda_var = lambda_var / np.sqrt(data.shape[0] - 1)  # Whitening (sklearn)
    inds = np.argsort(np.power(lambda_var, 2))
    lambda_var = np.power(lambda_var, 2)[inds]
    v = v[:, inds]
    sum_all = np.sum(lambda_var)

    # Return only the extracted components
    v = v[:, (v.shape[1] - n_comps) :]
    lambda_var = lambda_var[lambda_var.shape[0] - n_comps :]
    sum_used = np.sum(lambda_var)
    retained = (sum_used / sum_all) * 100
    LGR.debug(f"{retained}% of non-zero components retained")

    return v, lambda_var


def _eigensp_adj(lam, n, p):
    """
    Eigen spectrum adjustment for EVD on finite samples.

    Parameters
    ----------
    lam : [Px1] array-like
        Component eigenvalues
    n : int
        Effective number of i.i.d. samples.
    p : int
        Number of eigen values.

    Returns
    -------
    lam_adj : (p,) array-like
              adjusted eigen values.

    Notes
    -----
    Adjusts the eigen spectrum to account for the finite samples
    after subsampling (Li et al., 2007)

    References
    ----------
    * Li, Y.O., Adalı, T. and Calhoun, V.D., (2007).
      Estimating the number of independent components for
      functional magnetic resonance imaging data.
      Human brain mapping, 28(11), pp.1251-1266.
    """
    r = p / n
    bp = np.power((1 + np.sqrt(r)), 2)
    bm = np.power((1 - np.sqrt(r)), 2)
    vv_step = (bp - bm) / (5 * p - 1)
    vv = np.arange(bm, bp + vv_step, vv_step)
    gv = (1 / (2 * np.pi * r * vv)) * np.sqrt(abs((vv - bm) * (bp - vv)))
    gvd = np.zeros(gv.shape)
    for i in range(gv.shape[0]):
        gvd[i] = sum(gv[0:i])

    gvd /= np.max(gvd)

    lam_emp = np.zeros(lam.shape)
    for idx, i in enumerate(np.arange(1, p + 1)):
        i_norm = (i) / p
        minx = np.argmin(abs(i_norm - gvd))
        lam_emp[idx] = vv[minx]

    lam_emp = np.flip(lam_emp)

    lam_adj = lam / lam_emp

    return lam_adj
