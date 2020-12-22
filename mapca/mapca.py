"""PCA based on Moving Average (stationary Gaussian) process
"""
import logging

import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import utils

LGR = logging.getLogger(__name__)


class MovingAveragePCA:
    """Scikit-learn-style moving-average PCA estimator."""

    def __init__(self, criteria="mdl", normalize=True):
        self.criteria = criteria
        self.normalize = normalize

    def _fit(self, X, shape_3d, mask_vec):
        n_x, n_y, n_z = shape_3d
        n_samples, n_timepoints = X.shape
        if self.normalize:
            scaler = StandardScaler(with_mean=True, with_std=True)
            # TODO: determine if tedana is already normalizing before this
            X = scaler.fit_transform(X)  # This was X_sc

        LGR.info("Performing SVD on original data...")
        V, eigenvalues = utils._icatb_svd(X, n_timepoints)
        LGR.info("SVD done on original data")

        # Reordering of values
        eigenvalues = eigenvalues[::-1]
        dataN = np.dot(X, V[:, ::-1])
        # Potentially the small differences come from the different signs on V

        # Using 12 gaussian components from middle, top and bottom gaussian
        # components to determine the subsampling depth.
        # Final subsampling depth is determined using median
        kurtv1 = utils._kurtn(dataN)
        kurtv1[eigenvalues > np.mean(eigenvalues)] = 1000
        idx_gauss = np.where(
            ((kurtv1[:, 0] < 0.3) & (kurtv1[:, 0] > 0) & (eigenvalues > np.finfo(float).eps)) == 1
        )[
            0
        ]  # NOTE: make sure np.where is giving us just one tuple
        idx = np.array(idx_gauss[:]).T
        dfs = np.sum(eigenvalues > np.finfo(float).eps)  # degrees of freedom
        minTp = 12

        if len(idx) >= minTp:
            middle = int(np.round(len(idx) / 2))
            idx = np.hstack([idx[0:4], idx[middle - 1 : middle + 3], idx[-4:]])
        else:
            minTp = np.min([minTp, dfs])
            idx = np.arange(dfs - minTp, dfs)

        idx = np.unique(idx)

        # Estimate the subsampling depth for effectively i.i.d. samples
        LGR.info("Estimating the subsampling depth for effective i.i.d samples...")
        mask_ND = np.reshape(mask_vec, (n_x, n_y, n_z), order="F")
        sub_depth = len(idx)
        sub_iid_sp = np.zeros((sub_depth,))
        for i in range(sub_depth):
            x_single = np.zeros(n_x * n_y * n_z)
            x_single[mask_vec == 1] = dataN[:, idx[i]]
            x_single = np.reshape(x_single, (n_x, n_y, n_z), order="F")
            sub_iid_sp[i] = utils._est_indp_sp(x_single)[0] + 1
            if i > 6:
                tmp_sub_sp = sub_iid_sp[0:i]
                tmp_sub_median = np.round(np.median(tmp_sub_sp))
                if np.sum(tmp_sub_sp == tmp_sub_median) > 6:
                    sub_iid_sp = tmp_sub_sp
                    break
            dim_n = x_single.ndim

        sub_iid_sp_median = int(np.round(np.median(sub_iid_sp)))
        if np.floor(np.power(n_samples / n_timepoints, 1 / dim_n)) < sub_iid_sp_median:
            sub_iid_sp_median = int(np.floor(np.power(n_samples / n_timepoints, 1 / dim_n)))
        N = np.round(n_samples / np.power(sub_iid_sp_median, dim_n))

        if sub_iid_sp_median != 1:
            mask_s = utils._subsampling(mask_ND, sub_iid_sp_median)
            mask_s_1d = np.reshape(mask_s, np.prod(mask_s.shape), order="F")
            dat = np.zeros((int(np.sum(mask_s_1d)), n_timepoints))
            LGR.info("Generating subsampled i.i.d. data...")
            for i_vol in range(n_timepoints):
                x_single = X[:, i_vol]
                x_single = np.reshape(x_single, (n_x, n_y, n_z), order="F")
                dat0 = utils._subsampling(x_single, sub_iid_sp_median)
                dat0 = np.reshape(dat0, np.prod(dat0.shape), order="F")
                dat[:, i_vol] = dat0[mask_s_1d == 1]

            # Perform Variance Normalization
            dat = scaler.fit_transform(dat)

            # (completed)
            LGR.info("Performing SVD on subsampled i.i.d. data...")
            V, eigenvalues = utils._icatb_svd(dat, n_timepoints)
            LGR.info("SVD done on subsampled i.i.d. data")
            eigenvalues = eigenvalues[::-1]

        LGR.info("Effective number of i.i.d. samples %d" % N)

        # Make eigen spectrum adjustment
        LGR.info("Perform eigen spectrum adjustment ...")
        eigenvalues = utils._eigensp_adj(eigenvalues, N, eigenvalues.shape[0])
        # (completed)
        if np.sum(np.imag(eigenvalues)):
            raise ValueError("Invalid eigenvalue found for the subsampled data.")

        # Correction on the ill-conditioned results (when tdim is large,
        # some least significant eigenvalues become small negative numbers)
        if eigenvalues[np.real(eigenvalues) <= np.finfo(float).eps].shape[0] > 0:
            eigenvalues[np.real(eigenvalues) <= np.finfo(float).eps] = np.min(
                eigenvalues[np.real(eigenvalues) >= np.finfo(float).eps]
            )
        LGR.info("Estimating the dimensionality ...")
        p = n_timepoints
        aic = np.zeros(p - 1)
        kic = np.zeros(p - 1)
        mdl = np.zeros(p - 1)

        for k_idx, k in enumerate(np.arange(1, p)):
            LH = np.log(np.prod(np.power(eigenvalues[k:], 1 / (p - k))) / np.mean(eigenvalues[k:]))
            mlh = 0.5 * N * (p - k) * LH
            df = 1 + 0.5 * k * (2 * p - k + 1)
            aic[k_idx] = (-2 * mlh) + (2 * df)
            kic[k_idx] = (-2 * mlh) + (3 * df)
            mdl[k_idx] = -mlh + (0.5 * df * np.log(N))

        itc = np.row_stack([aic, kic, mdl])

        if self.criteria == "aic":
            criteria_idx = 0
        elif self.criteria == "kic":
            criteria_idx = 1
        elif self.criteria == "mdl":
            criteria_idx = 2

        dlap = np.diff(itc[criteria_idx, :])
        a = np.where(dlap > 0)[0] + 1  # Plus 1 to
        if a.size == 0:
            n_components = itc[criteria_idx, :].shape[0]
        else:
            n_components = a[0]

        LGR.info("Estimated number of components is %d" % n_components)

        # PCA with estimated number of components
        ppca = PCA(n_components=n_components, svd_solver="full", copy=False, whiten=False)
        ppca.fit(X)

        # Assign attributes from model
        self.components_ = ppca.components_
        self.explained_variance_ = ppca.explained_variance_
        self.explained_variance_ratio_ = ppca.explained_variance_ratio_
        self.singular_values_ = ppca.singular_values_
        self.mean_ = ppca.mean_
        self.n_components_ = ppca.n_components_
        self.n_features_ = ppca.n_features_
        self.n_samples_ = ppca.n_samples_
        self.noise_variance_ = ppca.noise_variance_

    def fit(self, X, shape_3d, mask_vec):
        """Fit the model with X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(X, shape_3d, mask_vec)
        return self

    def fit_transform(self, X, shape_3d, mask_vec):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Transformed values.
        """
        self._fit(X, shape_3d, mask_vec)
        return self.transform(X)

    def transform(self, X):
        """Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted from a training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples and n_features
            is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        X_new = np.dot(np.dot(X, self.components_.T), np.diag(1.0 / self.explained_variance_))
        return X_new

    def inverse_transform(self, X):
        """Transform data back to its original space.

        In other words, return an input X_original whose transform would be X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data, where n_samples is the number of samples and n_components
            is the number of components.

        Returns
        -------
        X_original : array-like, shape (n_samples, n_features)
        """
        X_orig = np.dot(np.dot(X, np.diag(self.explained_variance_)), self.components_)
        return X_orig


def ma_pca(img, mask_img, criteria="mdl", normalize=False):
    """
    Run Singular Value Decomposition (SVD) on input data,
    automatically select components based on a Moving Average
    (stationary Gaussian) process. Finally perform PCA with
    selected number of components.

    Parameters
    ----------
    img : 4D niimg_like
        Unmasked data to compute the PCA on.
    mask_img : 3D niim_like
        Mask to apply on ``img``.
    criteria : {'aic', 'kic', mdl'}, optional
        Criteria to select the number of components; default='mdl'.

    Returns
    -------
    u : (S [*E] x C) array-like
        Component weight map for each component.
    s : (C,) array-like
        Variance explained for each component.
    varex_norm : (n_components,) array-like
        Explained variance ratio.
    v : (T x C) array-like
        Component timeseries.

    Notes
    -----
    aic : Akaike Information Criterion. Least aggressive option.
    kic : Kullback-Leibler Information Criterion. Stands in the
          middle in terms of aggressiveness.
    mdl : Minimum Description Length. Most aggressive
          (and recommended) option.
    """
    from nilearn import masking

    data = masking.apply_mask(img, mask_img).T  # not sure about transpose
    mask_vec = np.reshape(mask_img.get_fdata(), np.prod(mask_img.shape), order="F")
    pca = MovingAveragePCA(criteria=criteria, normalize=normalize)
    u = pca.fit_transform(data, shape_3d=img.shape, mask_vec=mask_vec)
    v = pca.components_.T
    varex_norm = pca.explained_variance_ratio_
    s = pca.explained_variance_
    return u, s, varex_norm, v
