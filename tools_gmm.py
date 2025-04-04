import numpy as np
from scipy.io import loadmat
from skimage.util import view_as_windows
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky


def load_background_gmm(path):
    data = loadmat(path)['GS']
    dim = data['dim'][0, 0][0, 0]
    nmodels = data["nmodels"][0, 0][0, 0]
    means = data["means"][0, 0]
    covs = data["covs"][0, 0]
    invcovs = data["invcovs"][0, 0]
    mixweights = data["mixweights"][0, 0][:, 0]
    return dim, nmodels, means, covs, invcovs, mixweights


def min_var_region(image, region_size=(100, 100)):
    regions = view_as_windows(image, region_size)
    var = np.var(regions, axis=(2, 3))
    min_index = np.unravel_index(np.argmin(var), var.shape)
    return regions[min_index]


def fit_gmm(region, n_clusters, patch_size=(8, 8)):
    patches = view_as_windows(region, patch_size)\
        .reshape(-1, patch_size[0]*patch_size[1])
    patches -= np.mean(patches, axis=1, keepdims=True)
    gmm_model = GaussianMixture(n_clusters)
    gmm_model.fit(patches)
    return gmm_model, 64, n_clusters, gmm_model.means_.T, gmm_model.covariances_.T, gmm_model.precisions_.T, gmm_model.weights_


def fit_rain_gmm(image, region_size=(100, 100), patch_size=(8, 8), n_clusters=20):
    region = min_var_region(image, region_size)
    return fit_gmm(region, n_clusters, patch_size), region


def init_gmm(n, means, covs, invcovs, mixweights):
    gmm = GaussianMixture(n_components=n)
    gmm.means_ = means.T
    gmm.covariances_ = covs.T
    gmm.precisions_ = invcovs.T
    gmm.weights_ = mixweights.T
    gmm.precisions_cholesky_ = _compute_precision_cholesky(covs.T, "full")
    return gmm
