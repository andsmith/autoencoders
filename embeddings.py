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
        self.points_2d = self._calc_embedding()

    @abstractmethod
    def _calc_embedding(self):
        """
        Calculate the embedding of the points.
        This method should be implemented by subclasses.
        :return: array of shape (n_samples, 2), embedded points in 2D space.
        """
        pass

    @abstractmethod
    def embed_points(self, points):
        """
        Embed new points into the existing embedding.
        :param points: array of shape (n_samples, n_features)
        :return: array of shape (n_samples, 2)
        """
        pass

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
        t=np.linspace(0, 1, n_points)
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
        return self.embed_points(self.points)

    def embed_points(self, points):
        return points[:, :2]
    


