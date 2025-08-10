import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import logging
import seaborn as sns


class PointSet2d(object):
    def __init__(self, min_separation):
        self._min_sep = min_separation
        self.points = np.array([]).reshape(0, 2)

    def add_points(self, points):
        """
        Add multiple points to the set, ensuring they are at least min_separation apart.
        :param points: array of shape (n_samples, 2)
        :returns: array mask, which points were added,
        """
        points = points.reshape(-1, 2)

        if len(self.points) == 0:
            self.points = points
            return np.ones(len(points)).astype(bool)

        dists = np.linalg.norm(self.points - points[:, np.newaxis], axis=2)
        valid = np.all(dists >= self._min_sep, axis=1)
        self.points = np.vstack([self.points, points[valid]])

        return valid

    def add_point(self, point):
        return self.add_points(point)[0]

    def well_separated(self, point):
        """
        Check if a point is far enough away from any existing points in the set.
        :param point: array of shape (2,)
        :returns: boolean indicating if the point is far enough away
        """
        if len(self.points) == 0:
            return True
        dists = np.linalg.norm(self.points - point, axis=1)
        return np.min(dists) >= self._min_sep


def _make_orthonormal_basis(dims):
    # Make a random orthornormal basis in d-dimensional space
    basis = np.random.randn(dims, dims)
    q, r = np.linalg.qr(basis)
    return q


def _make_random_cov(dims, scale=1.0):
    """
    Create a random covariance matrix for a given number of dimensions.
    :param dims: Number of dimensions
    :param scale: Scale factor for the covariance matrix
    :return: Random covariance matrix of shape (dims, dims)
    """
    min_dim = (scale * .02) ** 2
    max_dim = (scale) ** 2
    eigen_values = np.random.rand(dims) * (max_dim - min_dim)
    eigen_values[0] = max_dim
    # eigen_values[-1] = min_dim
    eigen_vectors = _make_orthonormal_basis(dims)
    cov = eigen_vectors @ np.diag(eigen_values) @ eigen_vectors.T
    return cov


def test_make_random_cov(n_points=10000, plot=False):
    test_scales = [0.1, 1.0, 10.0, 100.0]
    covs = [_make_random_cov(2, scale) for scale in test_scales]
    points = [np.random.multivariate_normal([0, 0], cov, n_points) for cov in covs]
    est_covs = [np.cov(point, rowvar=False) for point in points]
    for i in range(len(test_scales)):
        logging.info(f"Scale: {test_scales[i]}, Covariance Matrix:\n{covs[i]}\nEstimated Covariance:\n{est_covs[i]}")
        diff = np.abs(covs[i] - est_covs[i])
        rel_diff = np.max(diff / np.max(np.abs(covs[i])))
        assert rel_diff < 0.1, \
            f"Covariance matrices do not match for scale {test_scales[i]}\n" +\
            f"Test Cov:\n{covs[i]}\nEstimated Cov:\n{est_covs[i]}\n"\
            f"Relative Difference:\n{rel_diff} \nAbsolute Difference:\n{diff}"
    if plot:
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        for i, point in enumerate(points):
            ax[i // 2, i % 2].scatter(point[:, 0], point[:, 1], alpha=0.5, s=1)
            ax[i // 2, i % 2].set_title(f"Scale: {test_scales[i]}")
            ax[i // 2, i % 2].axis('equal')
        # plt.tight_layout()
        plt.show()


def make_test_data(d, n_points, n_clusters=10, separability=3.0):
    scale = 10.0
    cluster_means = np.random.uniform(-scale*separability, scale*separability, (n_clusters, d))
    cluster_covariances = np.array([_make_random_cov(d, scale) for _ in range(n_clusters)])
    cluster_sizes = np.random.rand(n_clusters)
    cluster_sizes /= np.sum(cluster_sizes)  # Normalize to sum to 1
    cluster_sizes *= n_points  # Scale to total number of points
    cluster_sizes = cluster_sizes.astype(int)  # Convert to integers
    cluster_sizes[0] += n_points - np.sum(cluster_sizes)  # Ensure total size matches n_points
    labels = np.concatenate([[i] * size for i, size in enumerate(cluster_sizes)])
    points = []

    for i in range(n_clusters):
        points.append(np.random.multivariate_normal(cluster_means[i],
                                                    cluster_covariances[i],
                                                    cluster_sizes[i]))
    return np.vstack(points), labels


def test_make_data(d=3, plot=True):
    seps = [0.0, 0.2, 1.0, 3.0, 6.0, 10.0]
    points, labels = [], []
    n_clusters = 10
    for sep in seps:
        logging.info(f"Generating test data with separability {sep}")
        p, l = make_test_data(d=d, n_points=20000, n_clusters=n_clusters, separability=sep)
        points.append(p)
        labels.append(l)
    colors = (np.array(plt.cm.gist_ncar(np.linspace(0, 1, n_clusters)))*255.).astype(int)
    if plot:
        fig, ax = plt.subplots(2, 3, figsize=(10, 6))
        ax = ax.flatten()
        for i, (point_set, label_set) in enumerate(zip(points, labels)):
            color_set = colors[label_set]
            ax[i].scatter(point_set[:, 0], point_set[:, 1], alpha=0.3,
                          c=color_set, s=2, label="cluster %i" % i)
            ax[i].set_title(f"Separability: {seps[i]}")
            ax[i].axis('equal')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_make_random_cov(n_points=10000, plot=True)
    # test_make_data(d=2, plot=True)
