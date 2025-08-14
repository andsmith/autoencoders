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


def fit_spaced_intervals(extent, n_intervals, spacing_fraction, min_spacing=1, fill_extent=True):
    """
    Fit intervals within a given extent, ensuring specified spacing between them, but not on either end,
    so the first interval starts at extent[0] and the last ends at extent[1]
    :param extent: The total extent (range) within which to fit intervals.
    :param n_intervals: The number of intervals to fit.
    :param spacing_fraction: The fraction of the interval size to use as spacing.
    :param fill_extent: 
        True: (default) intervals have integer sizes, but spacing can vary to fill the entire extent. 
        False:  items will be evenly spaced, but may not fill the entire extent.
    :return: A list of tuples representing the fitted intervals.
    """
    if n_intervals == 0:
        return []
    intervals = []
    total_size = extent[1] - extent[0]
    interval_size = int(total_size / (n_intervals + spacing_fraction * (n_intervals - 1)))
    spacing = (total_size - interval_size * n_intervals) / (n_intervals - 1) if n_intervals > 1 else 0
    if spacing < min_spacing:
        spacing = min_spacing
        interval_size = (total_size - spacing * (n_intervals - 1)) / n_intervals
        #interval_starts = (np.arange(n_intervals) * (interval_size + spacing) + extent[0])
    if not fill_extent:
        spacing = int(spacing)
        interval_size = int(interval_size)
    interval_starts = (np.arange(n_intervals) * (interval_size + spacing) + extent[0]).astype(int)
    interval_ends = interval_starts + int(interval_size)

    return [(start, end) for start, end in zip(interval_starts, interval_ends)]


def test_fit_spaced_intervals():
    ext = [0, 50]
    ext_range = ext[1]-ext[0]
    n_intervals = [1, 2, 3, 5, 7, 10, 12, 15]

    spacing_frac = [0.0, 0.1, 0.2, 0.5]
    n_rows, n_cols = len(n_intervals), len(spacing_frac)
    fix, ax = plt.subplots(n_rows, n_cols, figsize=(12, 8), sharex=True, sharey=True)
    ax = np.array(ax).reshape(n_rows, n_cols)
    for i, n_int in enumerate(n_intervals):
        for j, s_frac in enumerate(spacing_frac):
            logging.info(f"Testing intervals: {n_int}, spacing: {s_frac}")

            intervals = fit_spaced_intervals(ext, n_int, s_frac)
            intervals_even = fit_spaced_intervals(ext, n_int, s_frac, fill_extent=False)
            interval_size = intervals[0][1] - intervals[0][0] if intervals else 0
            def _plot_at_y(intv, y):
                xcoords = np.array([(start, end, np.nan) for (start, end) in intv]).flatten()
                ycoords = np.zeros_like(xcoords) + y
                coords = np.stack([xcoords[:, ...], ycoords[:, ...]], axis=-1)
                ax[i, j].plot(coords[:, 0], coords[:, 1], 'o-',markersize=2)
                ax[i, j].axis('off')
            _plot_at_y(intervals, -0.1)
            _plot_at_y(intervals_even, 0.1)
            ax[i, j].plot([ext[0], ext[0]], [-.3, .3], 'k-')
            ax[i, j].plot([ext[1], ext[1]], [-.3, .3], 'k-')
            ax[i, j].set_title(f"Intervals: {n_int}, size={interval_size}, spacing_frac={s_frac}\n"+
                               f"blue: {intervals[0][0]} .. {intervals[-1][1]},  " +
                               f"orange: {intervals_even[0][0]} .. {intervals_even[-1][1]}",
                               fontsize=10)
            ax[i, j].set_xlim(ext[0]-ext_range*.05, ext[1]+ext_range*.05)
            ax[i, j].set_ylim(-.3, .3)

    plt.suptitle('Extent: [%s, %s],  \nblue: fill-extent=True (uneven spacing, even endpoints),\n'
                 'orange: fill-extent=False (even spacing, uneven endpoints)' % (ext[0], ext[1]))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_make_random_cov(n_points=10000, plot=True)
    # test_make_data(d=2, plot=True)
    test_fit_spaced_intervals()
    logging.info("Tests completed successfully.")