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
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
import umap
from abc import ABC, abstractmethod
import logging


class Embedding(ABC):
    """
    Find a 2d embedding function for points in a high-dimensional feature space.
    Calculated embeddings will map all initial points so they fill the unit square.
    """

    def __init__(self, points):
        """
        Initialize the embedding with a set of points.
        :param points: array of shape (n_samples, n_features), points in feature space.
        """
        self.points = points
        points_2d_unscaled = self._calc_embedding()
        self.scale = self._calc_scale(points_2d_unscaled)
        self.points_2d = self._scale_points(points_2d_unscaled)

    @abstractmethod
    def _calc_embedding(self):
        """
        Calculate the embedding of the points.
        This method should be implemented by subclasses.
        :return: array of shape (n_samples, 2), embedded points in 2D space.
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

    def embed_points(self, points):
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

        :param start: starting point in the feature space
        :param end: ending point in the feature space
        :param n_points: number of points to interpolate
        :return: array of shape (n_points, 2), embedded points along the path in the embedding space
                 array of shape (n_points, n_features), points along the path in feature space
        """
        t = np.linspace(0, 1, n_points)
        path = start + t[:, np.newaxis] * (end - start)
        return self.embed_points(path), path

    def extrap_point_dir(self, start, vec, n_points, length_factor=1.5):
        """
        Extrapolate a point in the direction of a given vector.

        Add the vector to the starting point, and extend it by the length factor.
        Sample n_points on this line, find embedded locations.

        :param start: starting point in the feature space

        :param vec: vector to extrapolate in
        :param n_points: number of points to sample along the extrapolated line
        :param length_factor: factor to extend the vector by
        :return: array of shape (n_points, 2), embedded points along the extrapolated line
                 array of shape (n_points, n_features), feature-space points along the extrapolated line
        """
        length = np.linalg.norm(vec)
        direction = vec / length
        endpoint = start + direction * length_factor * length
        t = np.linspace(0, 1, n_points)
        path = start + t[:, np.newaxis] * (endpoint - start)
        return self.embed_points(path), path


class PassThroughEmbedding(Embedding):
    """
    A trivial embedding that does not change the points, uses first two dimensions
    of feature space as the 2d embedding.
    Useful for testing and debugging.
    """

    def _calc_embedding(self):
        return self.points[:, :2]

    def _embed_points(self, points):
        return points[:, :2]


class PCAEmbedding(Embedding):
    """
    Embedding using PCA to reduce the dimensionality of the points to 2D.
    """

    def _calc_embedding(self):
        logging.info("Calculating PCA embedding for %d points", self.points.shape[0])
        pca = PCA(n_components=2)
        return pca.fit_transform(self.points)

    def _embed_points(self, points):
        pca = PCA(n_components=2)
        return pca.fit_transform(points)

class UMAPEmbedding(Embedding):
    """
    Embedding using UMAP to reduce the dimensionality of the points to 2D.
    """

    def _calc_embedding(self):
        logging.info("Calculating UMAP embedding for %d points", self.points.shape[0])
        self.reducer = umap.UMAP(n_components=2)
        logging.info("\tFinished UMAP embedding.")
        pts_2d = self.reducer.fit_transform(self.points)
        logging.info("\tEmbedded training set.")
        return pts_2d

    def _embed_points(self, points):
        return self.reducer.transform(points)

class TSNEEmbedding(Embedding):
    """
    Embedding using t-SNE to reduce the dimensionality of the points to 2D.
    """
    def __init__(self, points, perplexity=30.0):
        self.perplexity = perplexity
        super().__init__(points)

    def _calc_embedding(self):
        logging.info("Calculating t-SNE embedding for %d points with perplexity %.2f"%( self.points.shape[0], self.perplexity))
        self.tsne = TSNE(n_components=2, perplexity=self.perplexity)
        logging.info("\tFinished t-SNE embedding.")
        pts_2d = self.tsne.fit_transform(self.points)
        logging.info("\tEmbedded training set.")

        return pts_2d

    def _embed_points(self, points):
        return self.tsne.transform(points)