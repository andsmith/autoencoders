import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt


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
    
    