import numpy as np
import matplotlib.pyplot as plt

from util import make_test_data


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

        self.pca_dims = None
        self.components = None
        self.scales = None
        self.means = None

    def get_name(self):
        return "PCA(%d%s)" % (self.pca_dims, ", w" if self.whiten else "")

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

    def fit_transform(self, points):
        self.means = np.mean(points, axis=0)

        if self._dims == 0:
            self.pca_dims = 0  # no pca, just the data
            self.scales = np.std(points, axis=0)
            return self.encode(points)

        cov = np.cov(points, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)

        if self._dims >= 1:
            self.pca_dims = self._dims

        if 0 < self._dims < 1:
            # Keep enough components to explain the desired variance
            total_variance = np.sum(eigvals)
            current_variance = 0
            num_components = 0
            while current_variance / total_variance < self._dims:
                current_variance += eigvals[-(num_components + 1)]
                num_components += 1
            self.pca_dims = num_components

        self.components = eigvecs[:, -self.pca_dims:]
        self.scales = eigvals[-self.pca_dims:] ** 0.5

        # Project the data onto the selected eigenvectors
        return self.encode(points)


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
    data[:,0]*=2
    data[:,1] +=25

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
    ax[0].scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels,s=2)
    ax[0].set_title("PCA-reduced data")
    ax[1].scatter(data[:, 0], data[:, 1], c=labels)
    ax[1].set_title("Original data (first 2 of %i dims)" % (d,))
    plt.show()


if __name__ == "__main__":
    test_pca_layer()
    test_pca_modes()
