import numpy as np
from shapely import points
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from abc import ABC, abstractmethod
from util import get_good_point_size, write_lines
import logging
import cv2
from util import split_bbox
from tiny_plot import tiny_plot


class ClusteringAlgorithm(ABC):
    """
    Abstract class for clustering algorithms, plugins to the cluster creator.
    """

    def __init__(self, k):
        self.k = k
        self._fit = False
        self.labels_ = None

    @abstractmethod
    def fit(self, x):
        """
        Cluster the points.
        set:
           * self._fit = True
           * self.labels_ = N x 1 array of cluster assignments
           * ... other attributes as needed

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

    
    @abstractmethod
    def draw_stats(self, image, bbox, color):
        """
        Draw stats about the clustering in the given bbox on the image.
        :param image: the image to draw on
        :param bbox: the bbox to draw in, dict with 'x' and 'y' keys, each a (min, max) tuple
        :param color: color to use for text/lines
        """
        pass

    @abstractmethod
    def _get_stat_lines(self):
        """
        Returns line of text strings to print in the stats box.
        """
        pass
    
    def _draw_txt_stats(self, image, bbox, color):
        """
        if just text lines, call this as implementation of draw_stats()
        """
        if not self._fit or (self.labels_ is None):
            lines = ["Not clustered",
                     '(no stats available)']
        else:
            lines = self._get_stat_lines()
        write_lines(image, bbox, lines, 5, color=color)


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
        super().__init__(k)
        # self._kmeans = KMeans(n_clusters=k)
        self.which = distance_metric
        self.means = None
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.loss = None
        self.labels_ = None

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
                cluster_ids, _ = self._find_closest_means(x, means=means)
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
                    logging.info("\t\titeration %i, cluster sizes: %s", iteration, counts)
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
        self.labels_, _ = self.assign(x)
        return self.labels_

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

    def _get_stat_lines(self):
        lines = [f"KMeans: k={self.k}",
                 f'loss={self.loss:.2f}']
        return lines
    
    def draw_stats(self, image, bbox, color):
        """
        Just print text stats.
        """
        self._draw_txt_stats(image, bbox, color)

class DBScanAlgorithm(ClusteringAlgorithm):

    def __init__(self, min_nn_samples=3, metric='euclidean', maxiter=50):
        """
        Find epsilon that achieves highest number of clusters.

        Assign outliers to the last cluster.

        :param metric: distance metric, 'euclidean' or 'cosine'
        :param min_samples: minimum samples in a cluster
        """
        super().__init__(k=-1)
        self.min_samples = min_nn_samples
        self.metric = metric
        if metric not in ['euclidean', 'cosine']:
            raise ValueError("metric must be 'euclidean' or 'cosine'")
        self._dbscan = None
        self.epsilon_ = None
        self.maxiter = maxiter
        self._n_clust = {}  # map from epsilon to number of clusters found
        
        self.labels_ = None
        self.iterations_ = 0
        self.epsilon_ = None
        self.epsilon_rel = None

    def _get_eps_range(self, x):
        dists = pairwise_distances(x, metric=self.metric)
        np.fill_diagonal(dists, 0)
        dists = dists.flatten()
        dist_min, dist_max = np.min(dists[dists > 0]), np.max(dists)
        return dist_min, dist_max

    def fit(self, x, verbose=False):
        """
        Ternary search on epsilon until we get the most clusters.
        """
        self._eps_low, self._eps_high = self._get_eps_range(x)
        if verbose:
            logging.info("DBScan(MinSample=%i) - ternary search for max clusters, eps range: [%.4f, %.4f]",
                         self.min_samples, self._eps_low, self._eps_high)
            logging.info("\tsampling range...")
        self._n_clust = {}
        epsilon = (self._eps_high + self._eps_low) / 2

        def _test(eps):
            model = DBSCAN(eps=eps, min_samples=self.min_samples, metric=self.metric)
            labels = model.fit_predict(x)
            n_clusters = len(set(labels)) 
            self._n_clust[eps] = n_clusters
            return n_clusters

        high, low = self._eps_high, self._eps_low
        # Sample 20 values before searching
        test_eps = np.linspace(low, high, num=20)
        for eps in test_eps:
            self._n_clust[eps] = _test(eps)

        iteration=0
        best_epsilon = None
        max_clusters = -1
        if verbose:
            logging.info("\tstarting ternary search...")
        while (high - low >= 1e-3) and (iteration < self.maxiter):
            mid1 = low + (high - low) / 3
            mid2 = high - (high - low) / 3
            n1 = _test(mid1)
            n2 = _test(mid2)
            self._n_clust[mid1] = n1
            self._n_clust[mid2] = n2
            if n1 < n2:
                low = mid1
                if n2 >= max_clusters:
                    max_clusters = n2
                    best_epsilon = mid2
            else:
                high = mid2
                if n1 >= max_clusters:
                    max_clusters = n1
                    best_epsilon = mid1
            iteration += 1
            if verbose:
                logging.info("\tternary iter %i: eps range [%.4f, %.4f], mid1=%.4f (%i clusters), mid2=%.4f (%i clusters)",
                             iteration, low, high, mid1, n1, mid2, n2)
        epsilon = best_epsilon if best_epsilon is not None else (self._eps_high + self._eps_low) / 2
        self.epsilon_rel = (epsilon - self._eps_low) / (self._eps_high - self._eps_low)
        self.epsilon_ = epsilon
        self.iterations_ = iteration
        self._fit = True
        if verbose:
            logging.info("DBScan ternary search done in %i iterations, using eps=%.4f with %i clusters.",
                         iteration, epsilon, max_clusters)
            
        return self._fit_eps(x, epsilon, verbose=verbose)

    def _fit_eps(self, x, epsilon, verbose=False):
        if self._fit:
            logging.warning("Model has already been fit, refitting.")
        if verbose:
            logging.info("Fitting DBScan with min_samples=%i, n_samples=%i, dim=%i",
                         self.min_samples, x.shape[0], x.shape[1])
        if verbose:
            logging.info("\tusing eps=%.4f", epsilon)
        self._dbscan = DBSCAN(eps=epsilon, min_samples=self.min_samples, metric=self.metric)
        self.labels_ = self._dbscan.fit_predict(x)

        n_clusters = len(set(self.labels_[self.labels_ != -1]))  
        n_outliers = np.sum(self.labels_ == -1)

        if verbose:
            logging.info("\tDBScan found %i clusters, %i outliers", n_clusters, n_outliers)


        # Move outliers to a new cluster if there are any
        outlier_mask = self.labels_ == -1

        if n_outliers > 0:
            if verbose: 
                logging.info("\tfound %i outliers, assinging to cluster %i.", n_outliers, n_clusters + 1)
            self.labels_[outlier_mask] = n_clusters 
            n_clusters += 1

        if verbose:
            logging.info("DBScan classified %i points into %i clusters, sized: ", 
                         len(self.labels_), n_clusters)
            unique, counts = np.unique(self.labels_, return_counts=True)
            tot=0
            for u, c in zip(unique, counts):
                logging.info("\tcluster %i: %i points", u, c)
                tot+=c
            logging.info("\ttotal: %i points", tot)

        self.k = n_clusters

        self._core_samples = self._get_core_samples(self._dbscan, x)
        return self.labels_
    
    def _get_core_samples(self, dbscan,data, n_max=30):
        """
        We'll use distance to nearest core sample as our "cluster distance" metric.
        :param dbscan: the fitted DBSCAN model
        :returns:  list of arrays of core samples, one array per cluster
        """
        core_inds = dbscan.core_sample_indices_
        core_samples = []
        for cluster_id in range(self.k-1):  # won't include outliers, who have no core samples
            inds = core_inds[dbscan.labels_[core_inds] == cluster_id]
            core_samps = data[inds]
            if len(core_samps) > n_max:
                # randomly sample
                rand_inds = np.random.choice(len(core_samps), n_max, replace=False)
                core_samps = core_samps[rand_inds]
            core_samples.append(core_samps.reshape(-1, data.shape[1]))
        return core_samples
    
    def _get_distance(self, x, labels):
        """
        Get distance to nearest core sample in assigned cluster.
        :param x: N x d array of points
        :param labels: N x 1 array of cluster assignments
        """
        print("Max labels:", np.max(labels))
        print("Num core samples:", len(self._core_samples))

        distances = np.full(x.shape[0], np.inf)
        logging.info("Calculating distances to core samples for %i points...", x.shape[0])
        labs = np.unique(labels)
        for lab in labs:
            print("Processing label:", lab)
            sample_mask = labels == lab
            samples = x[sample_mask].reshape(-1, x.shape[1])
            core_samps = self._core_samples[lab]
            dists = pairwise_distances(samples, core_samps, metric=self.metric)
            min_dists = np.min(dists, axis=1)
            distances[sample_mask] = min_dists

        return distances

    def assign(self, x):
        if not self._fit:
            raise ValueError("Model has not been fit yet.")
        if self._dbscan is None:
            raise ValueError("Model has not been fit yet.")
        labels = self._dbscan.fit_predict(x)
        distances_inliers = self._get_distance(x[labels>-1], labels[labels>-1])
        distances_outliers = np.zeros(np.sum(labels==-1)) 
        distances = np.full(x.shape[0], np.inf)
        distances[labels>-1] = distances_inliers   
        distances[labels==-1] = distances_outliers
        labels[labels==-1] = self.k - 1
        return labels, distances

    def draw_stats(self, image, bbox, color):
        """
        split bbox in two vertically,
        in the upper:
           - plot n_clusters as a function of epsilon, calculated in binary search.
        in the lower:
           - print txt_stats
        """
        bbox_upper, bbox_lower = split_bbox(bbox, 0.5,orient='h')
        self._draw_eps_plot(image, bbox_upper, color)
        self._draw_txt_stats(image, bbox_lower, color)

    def _draw_eps_plot(self, image, bbox, color):
        if len(self._n_clust) < 2:
            lines = ["DBScan eps plot", "not enough data"]
            write_lines(image, bbox, lines, 5, color=color)
            return
        epsilons = np.array(sorted(self._n_clust.keys()))
        n_clusters = np.array([self._n_clust[eps] for eps in epsilons])
        size_wh = (bbox['x'][1] - bbox['x'][0], bbox['y'][1] - bbox['y'][0])
        img = tiny_plot(size_wh, x=epsilons, y=n_clusters,
                  x_label="epsilon", y_label="n clusters",adjust_params={'left':.3,'bottom':.35})
        image[bbox['y'][0]:bbox['y'][1], bbox['x'][0]:bbox['x'][1]] = img

    def _get_stat_lines(self):
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = np.sum(self.labels_ == -1)
        lines = [f"DBScan: min_samples={self.min_samples}",
                 f"eps={self.epsilon_:.4f}",
                 f"eps_rel={self.epsilon_rel:.2f}",
                 f"n clusters={n_clusters}",
                 f"n noise pts={n_noise}",
                 f"n iters={self.iterations_}"]
        return lines


class DBScanManualAlgorithm(DBScanAlgorithm):
    def __init__(self, epsilon_rel=0.1, min_nn_samples=3, metric='euclidean'):
        """
        Don't search for epsilon, use the given (relative) value.
        :param epsilon_rel: relative epsilon, in [0, 1], relative to the max pairwise distance
        :param metric: distance metric, 'euclidean' or 'cosine'
        :param min_nn_samples: number of nearest neighbors to use for distance graph
        :param k: expected number of clusters
        :param min_samples: minimum samples in a cluster
        """
        super().__init__(min_nn_samples=min_nn_samples, metric=metric)
        self.epsilon_rel = epsilon_rel

    def fit(self, x, verbose=False):
        dist_min, dist_max = self._get_eps_range(x)
        epsilon = dist_min + self.epsilon_rel * (dist_max - dist_min)
        if verbose:
            logging.info("DBScanManual(MinSample=%i, eps_rel=%.2f) - using eps=%.4f",
                         self.min_samples, self.epsilon_rel, epsilon)
        labels = self._fit_eps(x, epsilon, verbose=verbose)
        # set K to number of clusters found
        self.k = len(set(labels)) - (1 if -1 in labels else 0)

    def _get_stat_lines(self):
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = np.sum(self.labels_ == -1)
        lines = [f"DBScanManual: min_samples={self.min_samples}",
                 f"eps_rel={self.epsilon_rel:.2f}",
                 f"eps={self.epsilon_:.4f}",
                 f"n clusters={n_clusters}",
                 f"n noise pts={n_noise}"]
        return lines


class SpectralAlgorithm(ClusteringAlgorithm):
    def __init__(self, k, normalize=False):
        super().__init__(k)
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

    def fit(self, sim_graph, verbose=True):
        self._tree = sim_graph.get_tree()  # to cluster new points

        self._eigvals, self._eigvecs = self._solve(sim_graph)
        if verbose:
            logging.info("Fitting Spectral clustering with %i clusters, %i features, normalize=%s.  Calculating eigenvectors...",
                         self.k, self.k, self._normalize)
        eig_features = self._eigvecs[:, :self.k]
        if self._normalize:
            # normalize
            eig_features /= np.linalg.norm(eig_features, axis=1)[:, np.newaxis]

        # kmeans on eigenvectors
        logging.info("Fitting KMeans to %s eigenvectors, with %i clusters...", eig_features.shape, self.k)
        self._kmeans = KMeans(n_clusters=self.k)
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
    def _get_stat_lines(self):
        pass  # handled by simgraphs
    def draw_stats(self, image, bbox, color):
        pass # handled by simgraphs

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


def test_dbscan():
    import matplotlib.pyplot as plt

    n_clusters = 15
    spread = 2.0
    centers = np.random.rand(n_clusters, 2)*10 * spread
    covs = [np.eye(2) * (0.5 + np.random.rand()*2) for _ in range(n_clusters)]
    points = []
    for i in range(n_clusters):
        pts = np.random.multivariate_normal(centers[i], covs[i], size=300)
        points.append(pts)

    points = np.vstack(points)

    dbscan = DBScanAlgorithm(k=n_clusters, metric='euclidean')
    cluster_ids = dbscan.fit(points, verbose=True)

    plt.scatter(points[:, 0], points[:, 1], c=cluster_ids)
    plt.title("DBScan Clustering")
    plt.show()


def test_spectral():
    import matplotlib.pyplot as plt
    from similarity import NNSimGraph

    dim = 10
    n_clusters = 15
    spread = 3.0
    centers = np.random.rand(n_clusters, dim)*10 * spread

    def make_rand_covariance():
        A = np.random.randn(dim, dim)
        return np.dot(A, A.T)
    covs = [make_rand_covariance() for _ in range(n_clusters)]
    points = []
    for i in range(n_clusters):
        pts = np.random.multivariate_normal(centers[i], covs[i], size=300)
        points.append(pts)

    points = np.vstack(points)

    sim_graph = NNSimGraph(k=5, distance_metric='euclidean')
    sim_graph.fit(points)
    print(sim_graph._get_graph_stats())
    spectral = SpectralAlgorithm(k=n_clusters, normalize=True)
    cluster_ids = spectral.fit(sim_graph, verbose=True)

    plt.scatter(points[:, 0], points[:, 1], c=cluster_ids)
    plt.title("Spectral Clustering")
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # test_render_clustering()
    # test_kmeans()
    # test_dbscan()
    test_spectral()
