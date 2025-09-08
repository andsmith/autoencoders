import numpy as np
from shapely import points
from sklearn.cluster import KMeans
from abc import ABC, abstractmethod
from util import get_good_point_size, write_lines
import logging
import cv2
        

class ClusteringAlgorithm(ABC):
    """
    Abstract class for clustering algorithms, plugins to the cluster creator.
    """

    def __init__(self, name, k):
        self._name = name
        self.k = k
        self._fit = False

    @abstractmethod
    def fit(self, x):
        """
        Cluster the points.
        :param x: N x 2 array of points
        :returns: N x 1 array of cluster assignments
        """
        pass

    @abstractmethod
    def assign(self, x):
        """
        Assign clusters to new points
        :param x: N x 2 array of points
        :returns: N x 1 array of cluster assignments
        """
        pass

    def is_fit(self):
        return self._fit

    def get_k(self):
        return self.k

    def set_k(self, k):
        if self.k != k:
            self.k = k
            self._fit = False
            self.means = None
            self.loss = None

def render_clustering(img, points, cluster_ids, colors, clip_unit=True, margin_px=5):
    """
    Render the clustering.
    :param img: the image to render on
    :param points: N x 2 array of points
    :param cluster_ids: N x 1 array of cluster assignments
    :param colors: list of colors for each cluster
    :param clip_unit: if True, draws only points in unit square, else draws all points scaled to img size
    :param margin_px: don't put points in the margin
    :param pt_size: size of points (pt_size x pt_size boxes)
    """
    pt_size = get_good_point_size(points.shape[0], None)
    points_scaled = (points * img.shape[1::-1]).astype(int)
    if cluster_ids is None:
        cluster_ids = np.zeros(points.shape[0], dtype=np.int32)

    if clip_unit:
        valid = (points_scaled[:, 0] >= margin_px) & (points_scaled[:, 0] < img.shape[1] - margin_px) & \
                (points_scaled[:, 1] >= margin_px) & (points_scaled[:, 1] < img.shape[0] - margin_px)
        points = points[valid]
        cluster_ids = cluster_ids[valid]
        points_scaled = points_scaled[valid]

    if isinstance(colors, np.ndarray):
        colors = colors.tolist()
    for i, (x, y) in enumerate(points_scaled):
        color = colors[cluster_ids[i]]

        img[y:y + pt_size, x:x + pt_size] = color


class KMeansAlgorithm(ClusteringAlgorithm):
    def __init__(self, k, distance_metric='euclidean', max_iter=300, n_init=10, random_state=42):
        """
        :param k: number of clusters
        :param distance_metric: 'euclidean' or 'cosine'
        :param max_iter: maximum number of iterations
        :param n_init: number of initializations to try
        :param random_state: random seed

        """
        super().__init__('KMeans', k)
        # self._kmeans = KMeans(n_clusters=k)
        self.which = distance_metric
        self.means = None
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.loss = None


    def fit(self, x, verbose=False):
        if self._fit:
            logging.warning("Model has already been fit, refitting.")
        if verbose:
            logging.info("Fitting KMeans with k=%i, distance_metric=%s, n_samples=%i, dim=%i",
                         self.k, self.which, x.shape[0], x.shape[1])
        self._fit = True
        best_loss = np.inf
        best_means = None
        d = x.shape[1]
        old_cluster_ids = None
        np.random.seed(self.random_state)
        for trial in range(self.n_init):
            random_indices = np.random.choice(x.shape[0], self.k, replace=False)
            means = np.array([self._calc_mean(x[random_indices[i]].reshape(1, d)) for i in range(self.k)])

            for iteration in range(self.max_iter):
                cluster_ids,_ = self._find_closest_means(x, means=means)
                n_changed = cluster_ids.size if old_cluster_ids is None else np.sum(cluster_ids != old_cluster_ids)
                old_cluster_ids = cluster_ids
                counts = np.bincount(cluster_ids, minlength=self.k)
                if np.any(counts == 0):
                    if verbose:
                        logging.info("\t\tconverged in %i iterations (empty cluster)!!!!!!!!!!!!", iteration)
                    break
                means = np.array([self._calc_mean(x[cluster_ids == i].reshape(counts[i], d)) for i in range(self.k)])
                if iteration % 50 == 0 and verbose:
                    logging.info("\t\titeration %i had %i cluster assignment changes.", iteration, n_changed)
                    logging.info("\t\titeration %i, cluster sizes: %s", iteration, counts   )
                if n_changed == 0:
                    if verbose:
                        logging.info("\t\tconverged in %i iterations.", iteration)
                    break
            # Compute loss
            loss = np.sum((x - means[cluster_ids]) ** 2)
            if loss < best_loss:
                best_loss = loss
                best_means = means

            if verbose:
                logging.info("\ttrial %i/%i converged in %i iterations, loss: %.4f",
                             trial + 1, self.n_init, iteration, loss)
        self.means = best_means
        self.loss = best_loss
        if verbose:
            logging.info("Best loss after %i trials: %.4f", self.n_init, self.loss)
        return self.loss

    def _calc_mean(self, points):
        if len(points) == 0:
            return None
        mean = np.mean(points, axis=0)
        if self.which == 'cosine':
            mean /= np.linalg.norm(mean)
        return mean

    def _find_closest_means(self, samples, means=None):
        if means is None:
            means = self.means
        if self.which == 'euclidean':
            dists = np.linalg.norm(samples[:, np.newaxis, :] - means[np.newaxis, :, :], axis=2)
        elif self.which == 'cosine':
            # Cosine distance
            samples_norm = samples / np.linalg.norm(samples, axis=1, keepdims=True)
            means_norm = means  # Will already be normalized
            dists = 1 - np.dot(samples_norm, means_norm.T)
        closest_means = np.argmin(dists, axis=1)
        closest_dists = dists[np.arange(samples.shape[0]), closest_means]
        return closest_means, closest_dists

    def assign(self, x):
        if not self._fit:
            raise ValueError("Model has not been fit yet.")
        return self._find_closest_means(x)
    
    def draw_stats(self, image, bbox, color):
        if not self._fit:
            lines = ["Cluster for stats"]
        else:
            lines = [f"KMeans: k={self.k}",
                     f'loss={self.loss:.2f}']
        write_lines(image, bbox, lines, 5, color=color)


class SpectralAlgorithm(ClusteringAlgorithm):
    def __init__(self, n_clusters, normalize=False):
        super().__init__('Spectral', None)
        self._n_clusters = n_clusters
        self._normalize = normalize
        self._kmeans = None

    def _solve(self, g):
        w = g.get_matrix().copy()
        # set diagonal to zero
        np.fill_diagonal(w, 0)
        # compute the Laplacian matrix:
        degree_vec = np.sum(w, axis=1)
        degree_mat = np.diag(degree_vec)
        laplacian = degree_mat - w
        if self._normalize:
            # normalize
            degree_mat_sqrt = np.diag(1 / np.sqrt(degree_vec))
            laplacian = np.dot(degree_mat_sqrt, np.dot(laplacian, degree_mat_sqrt))

        eigvals, eigvecs = np.linalg.eigh(laplacian)

        # sort by eigenvalues
        idx = eigvals.argsort()
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        return eigvals, eigvecs

    def fit(self, sim_graph ,verbose=True):
        self._tree = sim_graph.get_tree()  # to cluster new points

        self._eigvals, self._eigvecs = self._solve(sim_graph)
        if verbose:
            logging.info("Fitting Spectral clustering with %i clusters, %i features, normalize=%s.  Calculating eigenvectors...",
                         self._n_clusters, self._n_clusters, self._normalize)
        eig_features = self._eigvecs[:, :self._n_clusters]
        if self._normalize:
            # normalize
            eig_features /= np.linalg.norm(eig_features, axis=1)[:, np.newaxis]

        # kmeans on eigenvectors
        logging.info("Fitting KMeans to %s eigenvectors, with %i clusters...", eig_features.shape, self._n_clusters)
        self._kmeans = KMeans(n_clusters=self._n_clusters)
        self._kmeans.fit(eig_features)
        self._kmeans_dists = self._kmeans.transform(eig_features)
        self._fit = True

        # cluster
        cluster_ids = self._kmeans.labels_

        return cluster_ids

    def get_eigens(self):
        return self._eigvals, self._eigvecs

    def assign(self, x):
        if self._kmeans is None:
            raise ValueError("Model has not been fit() yet.")
        # get index of nearest neighbor to x
        n_ind = self._tree.query(x, k=1)[1]
        assignments = self._kmeans.labels_[n_ind]
        distances = self._kmeans_dists[n_ind, assignments]
        return assignments, distances


def test_render_clustering():
    import cv2
    img = np.zeros((480, 640, 3), np.uint8)
    points = np.random.rand(100, 2)
    cluster_ids = np.random.randint(0, 3, 100)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    render_clustering(img, points, cluster_ids, colors)
    cv2.imshow('test_render_clustering', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_kmeans():
    import matplotlib.pyplot as plt

    n_clusters = 3
    n_dims = [2, 4, 8, 16, 32, 64]
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax = ax.flatten()

    def test_dim(d, ax):
        points = np.random.rand(1000, d)
        kmeans = KMeansAlgorithm(k=n_clusters, distance_metric='cosine', max_iter=100, n_init=5)
        loss = kmeans.fit(points, verbose=True)
        cluster_ids = kmeans.assign(points)
        ax.scatter(points[:, 0], points[:, 1], c=cluster_ids)
        ax.set_title("Dim=%i, Loss=%.2f" % (d, loss))
    for i, d in enumerate(n_dims):
        test_dim(d, ax[i])
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # test_render_clustering()
    test_kmeans()
