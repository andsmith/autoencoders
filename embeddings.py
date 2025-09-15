"""
Classes for embedding the distribution of latent representations of MNIST digits.

I.E.   MNIST(28x82 images, 8bit grayscale)
             |
             V
       VAE-Encoding(D dimensional latent values)
             |
             V
       Embedding2D(X,Y coordinates)

Embeddings find the D-dimensional to 2-dimensional mapping to preserve various
properties of the original (latent) distribution.
"""

import numpy as np
from shapely import points
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
import umap
from abc import ABC, abstractmethod
import logging
import pickle


class Embedding(ABC):
    """
    Find a 2d embedding function for points in a higher-dimensional latent space.
    Calculated embeddings will map all initial points so they fill the unit square.
    """

    def __init__(self):  # , points, inputs=None, class_labels=None):
        """
        Initialize the embedding with a set of points.
        :param points: array of shape (n_samples, n_features), points in latent space.

        :param class_labels: optional array of shape (n_samples,), int in range 0 -- n_classes-1.
        """
        self.embedder, self.scale = None, None

    def fit_embed(self, points):
        self.embedder, points_2d_unscaled = self._calc_embedding(points)
        self.scale = self._calc_scale(points_2d_unscaled)
        points_2d = self._scale_points(points_2d_unscaled)
        return points_2d

    @abstractmethod
    def get_name(self):
        """
        Return a short string (can be used as part of a filename) with all relevant parameters
        to reconstruct.
        """
        pass

    @abstractmethod
    def _calc_embedding(self, points):
        """
        Calculate the 2d embedding of the points.

        :return: tuple with
          - the embedder object, and
          - array of shape (n_samples, 2), embedded points in 2D space.
        """
        pass

    @abstractmethod
    def _embed_points(self, points):
        """
        Embed new points into the existing embedding.
        :param points: array of shape (n_samples, n_features)
        :return: array of shape (n_samples, 2)
        """
        pass

    def _check(self):
        if self.embedder is None:
            raise Exception("Don't call this before fitting.")

    def get_file_suffix(self):
        return "embedding=%s.pkl" % (self.get_name(),)

    def save(self, file_root):
        self._check()
        filename = "%s_%s" % (file_root, self.get_file_suffix())
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        return filename

    @staticmethod
    def from_file(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def embed_points(self, points):
        self._check()
        return self._scale_points(self._embed_points(points))

    def _calc_scale(self, points):
        mins, maxes = points.min(axis=0), points.max(axis=0)
        offsets, scales = mins, maxes - mins
        return offsets, scales

    def _scale_points(self, points):
        return (points - self.scale[0]) / self.scale[1]

    def interp_path(self, start, end, n_points):
        """
        Interpolate a path between two points in the embedding space & compute the
        corresponding embedded path.

        :param start: starting point in the latent space
        :param end: ending point in the latent space
        :param n_points: number of points to interpolate
        :return: array of shape (n_points, 2), embedded points along the path in the embedding space
                 array of shape (n_points, n_features), points along the path in latent space
        """
        self._check()
        t = np.linspace(0, 1, n_points)
        path = start + t[:, np.newaxis] * (end - start)
        return self.embed_points(path), path

    def extrap_point_dir(self, start, vec, n_points, length_factor=1.5):
        """
        Extrapolate a point in the direction of a given vector.

        Add the vector to the starting point, and extend it by the length factor.
        Sample n_points on this line, find embedded locations.

        :param start: starting point in the latent space

        :param vec: vector to extrapolate in
        :param n_points: number of points to sample along the extrapolated line
        :param length_factor: factor to extend the vector by
        :return: array of shape (n_points, 2), embedded points along the extrapolated line
                 array of shape (n_points, n_features), feature-space points along the extrapolated line
        """
        self._check()
        length = np.linalg.norm(vec)
        direction = vec / length
        endpoint = start + direction * length_factor * length
        t = np.linspace(0, 1, n_points)
        path = start + t[:, np.newaxis] * (endpoint - start)
        return self.embed_points(path), path

    def scale_to_bbox(self, points, bbox, img_size_wh=None):
        """
        "Zoom-in" on the points so the bounding box is mapped to the unit square, or
        an image's pixel space.
        :param points: array of shape (n_samples, 2), points in the embedding space
        :param bbox: bounding box in the embedding space, dict with 'x' and 'y' keys
                    each containing a tuple (min, max)
        :param img_size_wh: optional size of the image to map to (width, height)
        :returns: float array, shape (n_samples, 2), scaled points in the target space
        """
        if img_size_wh is not None:
            # Map to image pixel space
            scale_x = img_size_wh[0] / (bbox['x'][1] - bbox['x'][0])
            scale_y = img_size_wh[1] / (bbox['y'][1] - bbox['y'][0])
        else:
            # Map to unit square
            scale_x = 1.0 / (bbox['x'][1] - bbox['x'][0])
            scale_y = 1.0 / (bbox['y'][1] - bbox['y'][0])
        points = points.reshape(-1, 2)
        scaled_points = np.zeros_like(points).reshape(-1, 2)
        scaled_points[:, 0] = (points[:, 0] - bbox['x'][0]) * scale_x
        scaled_points[:, 1] = (points[:, 1] - bbox['y'][0]) * scale_y
        return scaled_points


class PassThroughEmbedding(Embedding):
    """
    A trivial embedding that does not change the points, uses first two dimensions
    of latent space as the 2d embedding.
    Useful for testing and debugging.
    """

    def get_name(self):
        return "First2Dims()"

    def _calc_embedding(self, points):
        # Can't return None as the embedder since we will look uninitialized.
        return 0, points[:, :2]

    def _embed_points(self, points):
        return points[:, :2]


class RandomProjectionEmbedding(Embedding):
    """
    Embedding using random projection to reduce the dimensionality of the points to 2D.
    """

    def get_name(self):
        return "RandomProjection(2)"

    def _calc_embedding(self, points):
        """
        Get 2 random directions (parent class will scale everything)
        """
        vecs = np.random.randn(points.shape[1], 2).reshape(-1, 2)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        rp = points @ vecs
        return vecs, rp

    def _embed_points(self, points):
        return points @ self.embedder


class PCAEmbedding(Embedding):
    """
    Embedding using PCA to reduce the dimensionality of the points to 2D.
    """

    def get_name(self):
        return "PCA(2)"

    def _calc_embedding(self, points):
        logging.info("Calculating PCA embedding for %d points", points.shape[0])
        pca = PCA(n_components=2)
        return pca, pca.fit_transform(points)

    def _embed_points(self, points):
        return self.embedder.transform(points)


class UMAPEmbedding(Embedding):
    """
    Embedding using UMAP to reduce the dimensionality of the points to 2D.
    """

    def __init__(self, neighbors=5):
        self.neighbors = neighbors
        super().__init__()

    def get_name(self):
        return "UMAP(2)"

    def _calc_embedding(self, points):
        logging.info("Calculating UMAP embedding for %d points", points.shape[0])
        reducer = umap.UMAP(n_neighbors=self.neighbors, n_components=2)
        logging.info("\tFinished UMAP embedding... embedding training set...")
        pts_2d = reducer.fit_transform(points)
        logging.info("\tdone.")
        return reducer, pts_2d

    def _embed_points(self, points):
        return self.embedder.transform(points)


class TSNEEmbedding(Embedding):
    """
    Embedding using t-SNE to reduce the dimensionality of the points to 2D.
    """

    def __init__(self, perplexity=30.0):
        self.perplexity = perplexity
        super().__init__()

    def get_name(self):
        return "t-SNE(perplexity=%.1f)" % (self.perplexity,)

    def _calc_embedding(self, points):
        logging.info("Calculating t-SNE embedding for %d points with perplexity %.2f" %
                     (points.shape[0], self.perplexity))
        logging.info("\tEmbedded training set.")
        tsne = TSNE(n_components=2, perplexity=self.perplexity)
        pts_2d = tsne.fit_transform(points)
        logging.info("\tFinished t-SNE embedding.")

        return tsne, pts_2d

    def _embed_points(self, points):
        return self.tsne.transform(points)
