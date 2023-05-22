"""PCA based on Moving Average (stationary Gaussian) process.

MAPCA: Moving Average Principal Components Analysis
Copyright (C) 2003-2009  GIFT developers

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""
import logging

import nibabel as nib
import numpy as np
from nilearn._utils import check_niimg_3d, check_niimg_4d
from scipy.stats import kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import utils

LGR = logging.getLogger(__name__)


class MovingAveragePCA:
    """Scikit-learn-style moving-average PCA estimator.

    The moving-average method estimates the underlying dimensionality
    of the data, after which a standard sklearn PCA is used to perform
    the decomposition.

    Parameters
    ----------
    criterion : {'mdl', 'aic', 'kic'}, optional
        Criterion used to select the number of components. Default is "mdl".
        ``mdl`` refers to Minimum Description Length, which is the most aggressive
        (and recommended) option.
        ``aic`` refers to the Akaike Information Criterion, which is the least aggressive option.
        ``kic`` refers to the Kullback-Leibler Information Criterion, which is the middle option.
    normalize : bool, optional
        Whether to normalize (zero mean and unit standard deviation) or not. Default is False.

    Attributes
    ----------
    components_ : :obj:`numpy.ndarray`, shape (n_components, n_features)
        Principal axes in feature space, representing the directions of maximum
        variance in the data. The components are sorted by explained_variance_.
    u_ : :obj:`numpy.ndarray`, shape (n_components, n_mask)
        Component weight maps, limited to voxels in the mask.
    u_nii_ : 4D nibabel.nifti1.Nifti1Image
        Component weight maps, stored as a 4D niimg.
    explained_variance_ : :obj:`numpy.ndarray`, shape (n_components,)
        The amount of variance explained by each of the selected components.

        Equal to n_components largest eigenvalues of the covariance matrix of X.
    explained_variance_ratio_ : :obj:`numpy.ndarray`, shape (n_components,)
        Percentage of variance explained by each of the selected components.

        If n_components is not set then all components are stored and the sum of the
        ratios is equal to 1.0.
    singular_values_ : :obj:`numpy.ndarray`, shape (n_components,)
        The singular values corresponding to each of the selected components.
        The singular values are equal to the 2-norms of the n_components
        variables in the lower-dimensional space.
    mean_ : :obj:`numpy.ndarray`, shape (n_features,)
        Per-feature empirical mean, estimated from the training set.

        Equal to X.mean(axis=0).
    n_components_ : int
        The estimated number of components.
        When n_components is set to ‘mle’ or a number between 0 and 1
        (with svd_solver == ‘full’) this number is estimated from input data.
        Otherwise it equals the parameter n_components, or the lesser value of
        n_features and n_samples if n_components is None.
    n_features_ : int
        Number of features in the training data.
    n_samples_ : int
        Number of samples in the training data
    aic_ : dict
        Dictionary containing the Akaike Information Criterion results:
            - 'n_components': The number of components chosen by the AIC criterion.
            - 'value': The AIC curve values.
            - 'explained_variance_total': The total explained variance of the components.
    kic_ : dict
        Dictionary containing the Kullback-Leibler Information Criterion results:
            - 'n_components': The number of components chosen by the KIC criterion.
            - 'value': The KIC curve values.
            - 'explained_variance_total': The total explained variance of the components.
    mdl_ : dict
        Dictionary containing the Minimum Description Length results:
            - 'n_components': The number of components chosen by the MDL criterion.
            - 'value': The MDL curve values.
            - 'explained_variance_total': The total explained variance of the components.
    varexp_90_ : dict
        Dictionary containing the 90% variance explained results:
            - 'n_components': The number of components chosen by the 90% variance explained
                              criterion.
            - 'explained_variance_total': The total explained variance of the components.
    varexp_95_ : dict
        Dictionary containing the 95% variance explained results:
            - 'n_components': The number of components chosen by the 95% variance explained
                              criterion.
            - 'explained_variance_total': The total explained variance of the components.
    all_ : dict
        Dictionary containing the results for all possible components:
            - 'n_components': Total number of possible components.
            - 'explained_variance_total': The total explained variance of the components.

    References
    ----------
    * Li, Y. O., Adali, T., & Calhoun, V. D. (2007). Estimating the number of
      independent components for functional magnetic resonance imaging data.
      Human Brain Mapping, 28(11), 1251–1266. https://doi.org/10.1002/hbm.20359

    Translated from the MATLAB code available in GIFT. https://trendscenter.org/software/gift/
    """

    def __init__(self, criterion="mdl", normalize=True):
        self.criterion = criterion
        self.normalize = normalize

    def _fit(self, img, mask, IIDsubsample=None):
        LGR.info(
            "Performing dimensionality reduction based on GIFT "
            "(https://trendscenter.org/software/gift/) and Li, Y. O., Adali, T., "
            "& Calhoun, V. D. (2007). Estimating the number of independent components "
            "for functional magnetic resonance imaging data. Human Brain Mapping, 28(11), "
            "1251–1266. https://doi.org/10.1002/hbm.20359"
        )

        img = check_niimg_4d(img)
        mask = check_niimg_3d(mask)
        data = img.get_fdata()
        mask = mask.get_fdata()

        [n_x, n_y, n_z, n_timepoints] = data.shape
        data_nib_V = np.reshape(data, (n_x * n_y * n_z, n_timepoints), order="F")
        mask_vec = np.reshape(mask, n_x * n_y * n_z, order="F")
        X = data_nib_V[mask_vec == 1, :]

        n_samples = np.sum(mask_vec)

        self.scaler_ = StandardScaler(with_mean=True, with_std=True)
        if self.normalize:
            # TODO: determine if tedana is already normalizing before this
            X = self.scaler_.fit_transform(X.T).T  # This was X_sc
            # X = ((X.T - X.T.mean(axis=0)) / X.T.std(axis=0)).T

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
        kurt = kurtosis(dataN, axis=0, fisher=True)
        kurt[kurt < 0] = 0
        kurt = np.expand_dims(kurt, 1)

        kurt[eigenvalues > np.mean(eigenvalues)] = 1000
        idx_gauss = np.where(
            ((kurt[:, 0] < 0.3) & (kurt[:, 0] > 0) & (eigenvalues > np.finfo(float).eps)) == 1
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
        LGR.info(f"Esimated subsampling depth for effective i.i.i samples: {sub_iid_sp_median}")

        # Always save the calculated IID subsample value, but, if there is a user provide value, assign that to sub_iid_sp_median
        #   and use that instead
        calculated_sub_iid_sp_median = sub_iid_sp_median
        if IIDsubsample:
            if (isinstance(IIDsubsample, int) or (isinstance(IIDsubsample, float) and IIDsubsample == int(IIDsubsample))) and (1 <= IIDsubsample <= n_samples):        
                sub_iid_sp_median = IIDsubsample
            else:
                raise ValueError(f"IIDsubsample must be an integer between 1 and the number of samples. It is {IIDsubsample}")




        if np.floor(np.power(n_samples / n_timepoints, 1 / dim_n)) < sub_iid_sp_median:
            sub_iid_sp_median = int(np.floor(np.power(n_samples / n_timepoints, 1 / dim_n)))
        N = np.round(n_samples / np.power(sub_iid_sp_median, dim_n))

        if sub_iid_sp_median != 1:
            mask_s = utils._subsampling(mask_ND, sub_iid_sp_median)
            mask_s_1d = np.reshape(mask_s, np.prod(mask_s.shape), order="F")
            dat = np.zeros((int(np.sum(mask_s_1d)), n_timepoints))
            LGR.info("Generating subsampled i.i.d. data...")
            for i_vol in range(n_timepoints):
                x_single = np.zeros(n_x * n_y * n_z)
                x_single[mask_vec == 1] = X[:, i_vol]
                x_single = np.reshape(x_single, (n_x, n_y, n_z), order="F")
                dat0 = utils._subsampling(x_single, sub_iid_sp_median)
                dat0 = np.reshape(dat0, np.prod(dat0.shape), order="F")
                dat[:, i_vol] = dat0[mask_s_1d == 1]

            # Perform Variance Normalization
            temp_scaler = StandardScaler(with_mean=True, with_std=True)
            dat = temp_scaler.fit_transform(dat.T).T

            # (completed)
            LGR.info("Performing SVD on subsampled i.i.d. data...")
            V, eigenvalues = utils._icatb_svd(dat, n_timepoints)
            LGR.info("SVD done on subsampled i.i.d. data")
            eigenvalues = eigenvalues[::-1]

        LGR.info("Effective number of i.i.d. samples %d from %d total voxels" % (N, n_samples))

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

        dlap = np.diff(itc, axis=1)

        # Calculate optimal number of components with each criterion
        # AIC
        a_aic = np.where(dlap[0, :] > 0)[0] + 1
        if a_aic.size == 0:
            n_aic = itc[0, :].shape[0]
        else:
            n_aic = a_aic[0]

        # KIC
        a_kic = np.where(dlap[1, :] > 0)[0] + 1
        if a_kic.size == 0:
            n_kic = itc[1, :].shape[0]
        else:
            n_kic = a_kic[0]

        # MDL
        a_mdl = np.where(dlap[2, :] > 0)[0] + 1
        if a_mdl.size == 0:
            n_mdl = itc[2, :].shape[0]
        else:
            n_mdl = a_mdl[0]

        if self.criterion == "aic":
            n_components = n_aic
        elif self.criterion == "kic":
            n_components = n_kic
        elif self.criterion == "mdl":
            n_components = n_mdl

        LGR.info("Performing PCA")

        # PCA with all possible components (the estimated selection is made after)
        ppca = PCA(n_components=None, svd_solver="full", copy=False, whiten=False)
        ppca.fit(X)

        # Get cumulative explained variance as components are added
        cumsum_varexp = np.cumsum(ppca.explained_variance_ratio_)

        # Calculate number of components for 90% varexp
        n_comp_varexp_90 = np.where(cumsum_varexp >= 0.9)[0][0] + 1

        # Calculate number of components for 95% varexp
        n_comp_varexp_95 = np.where(cumsum_varexp >= 0.95)[0][0] + 1

        LGR.info("Estimated number of components is %d" % n_components)

        # Save results of each criterion into dictionaries
        self.aic_ = {
            "n_components": n_aic,
            "value": aic,
            "explained_variance_total": cumsum_varexp[n_aic - 1],
        }
        self.kic_ = {
            "n_components": n_kic,
            "value": kic,
            "explained_variance_total": cumsum_varexp[n_kic - 1],
        }
        self.mdl_ = {
            "n_components": n_mdl,
            "value": mdl,
            "explained_variance_total": cumsum_varexp[n_mdl - 1],
        }
        self.varexp_90_ = {
            "n_components": n_comp_varexp_90,
            "explained_variance_total": cumsum_varexp[n_comp_varexp_90 - 1],
        }
        self.varexp_95_ = {
            "n_components": n_comp_varexp_95,
            "explained_variance_total": cumsum_varexp[n_comp_varexp_95 - 1],
        }
        self.all_ = {
            "n_components": ppca.n_components_,
            "explained_variance_total": cumsum_varexp,
        }
        self.subsampling_ = {
            "calculated_IID_subsample_depth": calculated_sub_iid_sp_median,
            "used_IID_subsample_depth": sub_iid_sp_median,
            "effective_num_IID_samples": N,
            "total_num_samples": n_samples,
        }

        # Assign attributes from model
        self.components_ = ppca.components_[:n_components, :]
        self.explained_variance_ = ppca.explained_variance_[:n_components]
        self.explained_variance_ratio_ = ppca.explained_variance_ratio_[:n_components]
        self.singular_values_ = ppca.singular_values_[:n_components]
        self.mean_ = ppca.mean_
        self.n_components_ = n_components
        self.n_features_ = ppca.n_features_
        self.n_samples_ = ppca.n_samples_
        # Commenting out noise variance as it depends on the covariance of the estimation
        # self.noise_variance_ = ppca.noise_variance_
        component_maps = np.dot(
            np.dot(X, self.components_.T), np.diag(1.0 / self.explained_variance_)
        )
        component_maps_3d = np.zeros((n_x * n_y * n_z, n_components))
        component_maps_3d[mask_vec == 1, :] = component_maps
        component_maps_3d = np.reshape(component_maps_3d, (n_x, n_y, n_z, n_components), order="F")
        self.u_ = component_maps
        self.u_nii_ = nib.Nifti1Image(component_maps_3d, img.affine, img.header)

    def fit(self, img, mask, IIDsubsample=None):
        """Fit the model with X.

        Parameters
        ----------
        img : 4D niimg_like
            Data on which to apply PCA.
        mask : 3D niimg_like
            Mask to apply on ``img``.
        IIDsubsample : int
            The subsampling value so that the voxels are assumed to be
            independent and identically distributed (IID).
            Default=None (use estimated value)


        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(img, mask, IIDsubsample=IIDsubsample)
        return self

    def fit_transform(self, img, mask, IIDsubsample=None):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        img : 4D niimg_like
            Data on which to apply PCA.
        mask : 3D niimg_like
            Mask to apply on ``img``.
        IIDsubsample : int
            The subsampling value so that the voxels are assumed to be independent
            and identically distributed (IID)
            2 would mean using every other voxel in 3D space would mean the
            remaining voxels are considered IID. 3 would mean every 3rd voxel.
            Default=None (use estimated value)

        Returns
        -------
        X_new : 4D niimg_like
            Component weight maps.

        Notes
        -----
        The transformation step is different from scikit-learn's approach,
        which ignores explained variance.

        IIDsubsample is always calculated automatically, but it should be consistent
        across a dataset with the sample acquisition parameters. In practice, it sometimes
        gives a different value and causes problems. That is, for a dataset with 100 runs,
        it is 2 in most runs, but when it is 3 substantially fewer components are estimated
        and when it is 1, there is almost no dimensionality reduction. This has been added
        as an option user provided parameter to use with caution. If mapca seems to be having
        periodic mis-estimates, then this parameter should make it possible to set the IID
        subsample size to be consistent across a dataset.
        """
        self._fit(img, mask, IIDsubsample=IIDsubsample)
        return self.transform(img)

    def transform(self, img):
        """Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted from a training set.

        Parameters
        ----------
        img : 4D niimg_like
            Data on which to apply PCA.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        Notes
        -----
        This is different from scikit-learn's approach, which ignores explained variance.
        """
        # X = self.scaler_.fit_transform(X.T).T
        # X_new = np.dot(np.dot(X, self.components_.T), np.diag(1.0 / self.explained_variance_))
        return self.u_nii_

    def inverse_transform(self, img, mask):
        """Transform data back to its original space.

        In other words, return an input X_original whose transform would be X.

        Parameters
        ----------
        img : 4D niimg_like
            Component weight maps.
        mask : 3D niimg_like
            Mask to apply on ``img``.

        Returns
        -------
        img_orig : 4D niimg_like
            Reconstructed original data, with fourth axis corresponding to time.

        Notes
        -----
        This is different from scikit-learn's approach, which ignores explained variance.
        """
        img = check_niimg_4d(img)
        mask = check_niimg_3d(mask)
        data = img.get_fdata()
        mask = mask.get_fdata()

        [n_x, n_y, n_z, n_components] = data.shape
        data_nib_V = np.reshape(data, (n_x * n_y * n_z, n_components), order="F")
        mask_vec = np.reshape(mask, n_x * n_y * n_z, order="F")
        X = data_nib_V[mask_vec == 1, :]

        X_orig = np.dot(np.dot(X, np.diag(self.explained_variance_)), self.components_)
        if self.normalize:
            X_orig = self.scaler_.inverse_transform(X_orig.T).T

        n_t = X_orig.shape[1]
        out_data = np.zeros((n_x * n_y * n_z, n_t))
        out_data[mask_vec == 1, :] = X_orig
        out_data = np.reshape(out_data, (n_x, n_y, n_z, n_t), order="F")
        img_orig = nib.Nifti1Image(out_data, img.affine, img.header)
        return img_orig


def ma_pca(img, mask, criterion="mdl", normalize=False, IIDsubsample=None):
    """Perform moving average-based PCA on imaging data.

    Run Singular Value Decomposition (SVD) on input data,
    automatically select components based on a Moving Average
    (stationary Gaussian) process. Finally perform PCA with
    selected number of components.

    Parameters
    ----------
    img : 4D niimg_like
        Data on which to apply PCA.
    mask : 3D niimg_like
        Mask to apply on ``img``.
    criterion : {'mdl', 'aic', 'kic'}, optional
        Criterion used to select the number of components. Default is "mdl".
        ``mdl`` refers to Minimum Description Length, which is the most aggressive
        (and recommended) option.
        ``aic`` refers to the Akaike Information Criterion, which is the least aggressive option.
        ``kic`` refers to the Kullback-Leibler Information Criterion, which is the middle option.
    normalize : bool, optional
        Whether to normalize (zero mean and unit standard deviation) or not. Default is False.
    IIDsubsample : int, optional
            The subsampling value so that the voxels are assumed to be independent
            and identically distributed (IID).
            2 would mean using every other voxel in 3D space would mean the
            remaining voxels are considered IID. 3 would mean every 3rd voxel.
            Default=None (use estimated value)

    Returns
    -------
    u : array-like, shape (n_samples, n_components)
        Component weight map for each component, after masking.
    s : array-like, shape (n_components,)
        Variance explained for each component.
    varex_norm : array-like, shape (n_components,)
        Explained variance ratio.
    v : array-like, shape (n_timepoints, n_components)
        Component timeseries.
    """
    pca = MovingAveragePCA(criterion=criterion, normalize=normalize)
    _ = pca.fit_transform(img, mask, IIDsubsample=IIDsubsample)
    u = pca.u_
    s = pca.explained_variance_
    varex_norm = pca.explained_variance_ratio_
    v = pca.components_.T
    return u, s, varex_norm, v
