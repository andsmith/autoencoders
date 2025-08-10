from fileinput import filename
import os
import ipdb
import numpy as np
import matplotlib.pyplot as plt
from util import make_test_data
import logging
import pickle

WORKING_DIR = "PCA"


class PCA(object):

    def __init__(self, dims, whiten=True):
        """

        :param pca_dims: Number of PCA dimensions for pre-processing:
            0 == pca_dims     --> all data (no PCA)
            0 < pca_dims < 1  --> (float) fraction of variance retained
            pca_dims > 1      --> (int) number of dimensions to keep
        """
        self._dims = dims
        self.whiten = whiten
        self._n_train = None
        self.pca_dims = None
        self.components = None
        self.scales = None
        self.means = None
        logging.info("PCA(dims=%s, whiten=%s) initialized.", self._dims, self.whiten)

    def get_name(self, n_train=None, n_dims=None):
        nt = self._n_train if n_train is None else n_train
        nd = n_dims if n_dims is not None else self.pca_dims
        return "PCA(dim=%s_whiten=%s_n-train=%i)" % (nd, "T" if self.whiten else "F", nt)

    def _cache_name(self, n_train, n_dims):
        file = self.get_name(n_train, n_dims) + ".pkl"
        return os.path.join(WORKING_DIR, file)

    def _write_cache(self, clobber=False):
        if not os.path.exists(WORKING_DIR):
            os.makedirs(WORKING_DIR)
        filename = self._cache_name(self._n_train, n_dims=self.pca_dims)
        if os.path.exists(filename) and not clobber:
            raise ValueError("Cache file %s already exists. Use clobber=True to overwrite.", filename)
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        logging.info("PCA Cache file written: %s", filename)

    def _load_cache(self, n_dims):
        filename = self._cache_name(self._n_train, n_dims=n_dims)
        if not os.path.exists(filename):
            logging.info("PCA Cache file not found: %s", filename)
            return False
        with open(filename, 'rb') as f:
            pca = pickle.load(f)
            logging.info("PCA Cache file loaded: %s", filename)
        self.__dict__.update(pca.__dict__)
        return True

    def encode(self, points):
        codes = points - self.means

        if self.pca_dims == 0:
            codes = codes / self.scales if self.whiten else codes
        else:
            codes = codes @ self.components
            codes = codes / self.scales if self.whiten else codes
        return codes

    def decode(self, codes):

        if self.pca_dims == 0:
            points = (codes * self.scales if self.whiten else codes)
        else:
            points = codes * self.scales if self.whiten else codes
            points = points @ self.components.T

        return points + self.means

    def fit_transform(self, points, use_cache=False):
        self._n_train = points.shape[0]

        if use_cache:
            if 0 < self._dims < 1:
                raise ValueError("Can't use cache when deciding pca-dim by variance fraction.")
            n_dims = self._dims if self._dims != 0 else 784

            if self._load_cache(n_dims):
                return self.encode(points)

        logging.info("PCA(dims=%s, whiten=%s) training ...", self._dims, self.whiten)
        self.means = np.mean(points, axis=0)
        if self._dims == 0:
            self.pca_dims = 0  # no pca, just the data
            self.scales = np.std(points, axis=0)
            return self.encode(points)

        cov = np.cov(points, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        total_variance = np.sum(eigvals)

        if self._dims >= 1:
            self.pca_dims = self._dims
            current_variance = np.sum(eigvals[-self.pca_dims:])

        if 0 < self._dims < 1:
            # Keep enough components to explain the desired variance
            current_variance = 0
            num_components = 0
            while current_variance / total_variance < self._dims:
                current_variance += eigvals[-(num_components + 1)]
                num_components += 1
            self.pca_dims = num_components

        self.components = eigvecs[:, -self.pca_dims:]
        self.scales = eigvals[-self.pca_dims:] ** 0.5

        # Project the data onto the selected eigenvectors
        encoded = self.encode(points)

        logging.info("\tPCA training complete....")
        logging.info("\t\tPCA dims: %i,", self.pca_dims)
        logging.info("\t\tCaptured %.3f of the variance,", 100 * current_variance / total_variance)

        if use_cache:
            self._write_cache(clobber=True)

        return encoded


def _test_pca_modes(pca, data, labels, plot=True):
    codes = pca.fit_transform(data)
    decodes = pca.decode(codes)

    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(9, 4))
        ax = ax.flatten()

        def _show_data(points, title, ax):
            ax.scatter(points[:, 0], points[:, 1], s=1, c=labels, alpha=0.5)
            ax.set_title(title)
            ax.set_aspect('equal')

        _show_data(data, "Original Data", ax[0])
        _show_data(codes, "Encoded Data", ax[1])
        _show_data(decodes, "Decoded Data", ax[2])
        plt.suptitle(pca.get_name())
        plt.tight_layout()


def test_pca_modes():
    # cov = np.array([[1, 0.1], [0.1, .02]])
    # data = np.random.multivariate_normal([0, 0], cov, size=5000)  + 2
    data, labels = make_test_data(10, 5000, n_clusters=5, separability=2.0)
    data[:, 0] *= 2
    data[:, 1] += 25

    _test_pca_modes(PCA(dims=0), data, labels)
    _test_pca_modes(PCA(dims=0, whiten=False), data, labels)
    _test_pca_modes(PCA(dims=15, whiten=True), data, labels)
    _test_pca_modes(PCA(dims=15, whiten=False), data, labels)
    _test_pca_modes(PCA(dims=.95, whiten=True), data, labels)
    _test_pca_modes(PCA(dims=.95, whiten=False), data, labels)

    plt.show()


def test_pca_layer():
    d = 100
    data, labels = make_test_data(d, 5000, n_clusters=5, separability=1.0)
    pca = PCA(dims=5)
    reduced_data = pca.fit_transform(data)
    fig, ax = plt.subplots(1, 2, figsize=(8, 5))
    ax[0].scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, s=2)
    ax[0].set_title("PCA-reduced data")
    ax[1].scatter(data[:, 0], data[:, 1], c=labels)
    ax[1].set_title("Original data (first 2 of %i dims)" % (d,))
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_pca_layer()
    test_pca_modes()
