from clustering import ClusteringAlgorithm
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import cv2
from enum import IntEnum
from util import image_from_floats, apply_colormap
from abc import ABC, abstractmethod
from scipy.spatial import KDTree
import logging
from util import fit_spaced_intervals, get_font_size, get_best_font_size
from scipy.sparse.csgraph import connected_components


class SimilarityGraphTypes(IntEnum):
    EPSILON = 0
    NN = 1
    FULL = 2
    SOFT_NN = 3


def get_kind_from_name(names, name):
    for kind, kind_name in names.items():
        if kind_name == name:
            return kind
    raise ValueError(f"Invalid name: {name}")


class SimilarityGraph(object):
    """
    Build a similarity graph from a set of points using euclidean distances.
    Adjust parameters before extracting laplacian.

    """

    def __init__(self, points):
        """
        Construct a similarity graph from points in the unit square.
        :param points: 2D numpy array of points
        :param colormap: colormap to use for the similarity matrix image
        """
        self._points = points
        self._mat = None
        # print("Built similarit matrix, weights in range [%f, %f], frac_nonzero=%.5f" % (
        # np.min(self._mat), np.max(self._mat), np.count_nonzero(self._mat) / self._mat.size))

    @abstractmethod
    def _build(self):
        """
        Build the similarity matrix.
        :return: similarity matrix, image of the similarity matrix
        """
        pass

    @abstractmethod
    def set_param(self, **kwargs):
        """
        Set parameters for the similarity graph and rebuild the matrix.
        :param kwargs: parameters to set.  (After setting, call draw_stats to inspect effects.)
        """
        pass

    @abstractmethod
    def draw_stats(self, img, bbox, pad_px=5):
        """
        Draw stats about the similarity graph in the given bounding box.
        :param img: image to draw on
        :param bbox: bounding box to draw in
        :param pad_px: padding in pixels
        """
        pass

    def _write_lines(self, img, bbox, lines, pad_px):
        """
        """
        y_span = bbox['y'][0] + pad_px, bbox['y'][1]-pad_px
        x_span = bbox['x'][0] + pad_px, bbox['x'][1]-pad_px
        txt_w, txt_h = x_span[1]-x_span[0], y_span[1]-y_span[0]

        n_lines = len(lines)
        line_y = fit_spaced_intervals(y_span, n_lines, spacing_fraction=0, fill_extent=False)
        line_wh = (txt_w, line_y[0][1]-line_y[0][0])

        font_size, font_thick = get_best_font_size(lines, line_wh, font=cv2.FONT_HERSHEY_SIMPLEX, )
        for i, line in enumerate(lines):
            org = (x_span[0], int(line_y[i][0] + (line_y[i][1]-line_y[i][0]+font_size)//2))
            cv2.putText(img, line, org, cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), font_thick, cv2.LINE_AA)

    def draw_graph(self, img):
        """
        Draw an edge between points with nonzero entries in the similarity matrix.
        """
        lines = []
        nonzero = np.nonzero(np.triu(self._mat))
        for i, j in zip(*nonzero):
            lines.append(np.array([self._points[i], self._points[j]], dtype=np.int32))

        cv2.polylines(img, lines, False, (128, 128, 128), 1, cv2.LINE_AA)

    def get_matrix(self):
        if self._mat is None:
            self._mat = self._build()
        return self._mat

    @abstractmethod
    def make_img(self, colormap=None):
        pass

    def get_tree(self):
        return KDTree(self._points)

    def _get_graph_stats(self, wt_threshold=0.0):
        mat_thresh = self._mat > wt_threshold
        n = self._mat.shape[0]
        m = np.count_nonzero(mat_thresh) // 2
        degrees = np.sum(mat_thresh, axis=1)
        dmin, dmed, dmean, dmax = np.min(degrees), np.median(degrees), np.mean(degrees), np.max(degrees)
        dstd = np.std(degrees)
        # components
        n_components, labels = connected_components(mat_thresh)
        return [' n_components=%i'% n_components,
                ' n_nodes=%i'% n,
                ' n_edges=%i'% m,
                ' deg_mean=%.1f'% dmean,
                ' deg_median=%i'% dmed,
                ' deg_range=[%i, %i]'% (dmin, dmax)]

class EpsilonSimGraph(SimilarityGraph):

    def __init__(self, points, epsilon_rel=0.5):
        super().__init__(points)
        self._dists = squareform(pdist(self._points))
        self._epsilon_rel = epsilon_rel
        self._epsilon = None

    def set_param(self, epsilon_rel=None):
        if epsilon_rel is not None:
            self._epsilon_rel = epsilon_rel
        self._mat = self._build()

    def _build(self):
        """
        Build the similarity matrix using epsilon distance.
        I.e., two points are connected if their distance is less than epsilon, all 
        weight 1 or 0.
        """
        dmin, dmax = np.min(self._dists[self._dists > 0]), np.max(self._dists[self._dists<np.Inf])
        self._epsilon = dmin + self._epsilon_rel * (dmax - dmin)
        logging.info("Setting epsilon=%.5f using rel param %.3f to interpolate range[%.3f, %.3f]" % (
            self._epsilon, self._epsilon_rel, dmin, dmax))
        sim_matrix = np.zeros(self._dists.shape)
        sim_matrix[self._dists <= self._epsilon] = 1
        # remove self-loops
        np.fill_diagonal(sim_matrix, 0)
        return sim_matrix

    def make_img(self, colormap=None):
        img = image_from_floats(self._mat, 0, 1)
        img = cv2.merge([img, img, img])
        return img

    def draw_stats(self, img, bbox, pad_px=5):
        """
        Stats:
           * Number of nodes, edges
           * Number of components
           * min, median, mean, max degree of nodes

        :param img: image to draw on
        :param bbox: bounding box to draw stats in
        """
        if self._mat is None:
            raise ValueError("Similarity matrix not built yet, set parameters first.")
        graph_stat_lines = self._get_graph_stats()
        lines = ['Epsilon Similarity Graph',
                 ' epsilon-rel:  %.2f' % (self._epsilon_rel, ),
                 ' epsilon:      %.3f' % (self._epsilon, ),] + graph_stat_lines
        self._write_lines(img, bbox, lines, pad_px)


def test_epsilon_param_update():
    """
    From random points, show the "draw_stats" image as the value is adusted by the keyboard.
    """
    points = np.random.rand(1500, 10)
    sim_graph = EpsilonSimGraph(points)
    epsilon_rel = 0.5
    img_size_wh = (300, 350)
    blank = np.zeros((img_size_wh[1], img_size_wh[0], 3), dtype=np.uint8)
    blank[:] = 250, 250, 250
    bbox = {'x':(0, img_size_wh[0]), 'y':(0, img_size_wh[1])}

    sim_graph.set_param(epsilon_rel=epsilon_rel)

    while True:
        frame = blank.copy()
        sim_graph.draw_stats(frame, bbox)
        cv2.imshow("Epsilon Similarity Graph", frame)
        sim_mat = sim_graph.make_img()
        cv2.imshow("Similarity Matrix", sim_mat)
        key = cv2.waitKey(1)
        new_epsilon = None
        if key == ord('q'):
            break
        elif key == ord('['):
            new_epsilon = epsilon_rel + 0.01
        elif key == ord(']'):
            new_epsilon = epsilon_rel - 0.01
        if new_epsilon is not None:
            epsilon_rel = new_epsilon
            epsilon_rel = max(0, min(1, epsilon_rel))
            print("Setting epsilon_rel=%.3f" % (epsilon_rel, ))

            sim_graph.set_param(epsilon_rel=epsilon_rel)   

class FullSimGraph(SimilarityGraph):
    def __init__(self, points,sigma_rel =0.5):
        """
        Construct a similarity graph using the full similarity function.
        """        
        super().__init__(points)
        self._dists = squareform(pdist(self._points))
        self._sigma_rel = sigma_rel
        self._sigma = None

    def _build(self):
        """
        Build the similarity matrix using the full similarity function.
        Interpolate between 0.0
        """
        _, dmax = np.min(self._dists[self._dists > 0]), np.max(self._dists)
        dmin=0.001
        self._sigma = dmin + self._sigma_rel * (dmax - dmin)
        logging.info("Setting sigma=%.5f using rel param %.3f to interpolate range[%.3f, %.3f]" % (
            self._sigma, self._sigma_rel, dmin, dmax))
        sim_matrix = np.exp(-self._dists**2 / (2*self._sigma**2))
        np.fill_diagonal(sim_matrix, 0)
        return sim_matrix

    def make_img(self, colormap=None):
        img = apply_colormap(self._mat, colormap)
        return img
    
    def set_param(self, sigma_rel=None):
        if sigma_rel is not None:
            self._sigma_rel = sigma_rel
        self._mat = self._build()

    def draw_stats(self, img, bbox, pad_px=5, tol_rel=0.1):
        """
        Stats:
           * Number of nodes, edges
           * Number of components
           * min, median, mean, max degree of nodes

        :param img: image to draw on
        :param bbox: bounding box to draw stats in
        :param pad_px: padding in pixels
        :param tol_rel: relative tolerance for considering an edge to exist (wrt sd. of )
        """
        if self._mat is None:
            raise ValueError("Similarity matrix not built yet, set parameters first.")
        tol_abs = tol_rel * np.std(self._mat)
        graph_stat_lines = self._get_graph_stats(wt_threshold=tol_abs)
        lines = ['Full Similarity Graph',
                 ' sigma-rel:  %.2f' % (self._sigma_rel, ),
                 ' sigma:      %.3f' % (self._sigma, )] + graph_stat_lines
        self._write_lines(img, bbox, lines, pad_px)

def test_full_param_update():
    """
    From random points, show the "draw_stats" image as the value is adusted by the keyboard.
    """
    points = np.random.rand(500, 10)
    sim_graph = FullSimGraph(points, sigma_rel=0.0)
    sigma_rel = 0.5
    img_size_wh = (300, 350)
    blank = np.zeros((img_size_wh[1], img_size_wh[0], 3), dtype=np.uint8)
    blank[:] = 250, 250, 250
    bbox = {'x':(0, img_size_wh[0]), 'y':(0, img_size_wh[1])}

    sim_graph.set_param(sigma_rel=sigma_rel)

    while True:
        frame = blank.copy()
        sim_graph.draw_stats(frame, bbox)
        cv2.imshow("Full Similarity Graph", frame)
        sim_mat = sim_graph.make_img()
        cv2.imshow("Similarity Matrix", sim_mat)
        key = cv2.waitKey(1)
        new_sigma = None
        if key == ord('q'):
            break
        elif key == ord('['):
            new_sigma = sigma_rel + 0.01
        elif key == ord(']'):
            new_sigma = sigma_rel - 0.01
        if new_sigma is not None:
            sigma_rel = new_sigma
            sigma_rel = max(0, min(1, sigma_rel))
            print("Setting sigma_rel=%.3f" % (sigma_rel, ))

            sim_graph.set_param(sigma_rel=sigma_rel)



class NNSimGraph(SimilarityGraph):
    def __init__(self, points):
        """
        Construct a similarity graph using K-nearest neighbors.
        :param points: 2D numpy array of points
        :param k: number of nearest neighbors
        :param mutual: if True, only connect if both points are among each other's K-nearest neighbors,
            otherwise connect if either is among the other's K-nearest neighbors.
        """
        super().__init__(points)

    def set_param(self, k, mutual):
        self._k = k
        self._mutual = mutual
        self._mat = self._build()

    def _build(self):
        """
        Build the similarity matrix using K-nearest neighbors, i.e.
        two points are connected if they are among each other's K-nearest neighbors.
        :param k: number of nearest neighbors
        :param mutual: if True, only connect if both points are among each other's K-nearest neighbors,
            otherwise connect if either is among the other's K-nearest neighbors.
        """
        # add 1 to k to include self
        logging.info("Building %s-NN graph with k=%i, mutual=%s",
                     "Mutual" if self._mutual else "Asymmetric", self._k, self._mutual)
        nbrs = NearestNeighbors(n_neighbors=self._k+1, algorithm='ball_tree').fit(self._points)
        logging.info("Fitted NearestNeighbors modelm, computing graph...")
        edge_mat = nbrs.kneighbors_graph(self._points, mode='connectivity').toarray()

        if self._mutual:
            edge_mat = np.logical_and(edge_mat, edge_mat.T)
        else:
            edge_mat = np.logical_or(edge_mat, edge_mat.T)
        np.fill_diagonal(edge_mat, 0)
        return edge_mat

    def make_img(self, colormap=None):
        img = image_from_floats(self._mat, 0, 1)
        img = cv2.merge([img, img, img])
        return img
    
    def draw_stats(self, img, bbox, pad_px=5):
        """
        Stats:
           * Number of nodes, edges
           * Number of components
           * min, median, mean, max degree of nodes

        :param img: image to draw on
        :param bbox: bounding box to draw stats in
        """
        if self._mat is None:
            raise ValueError("Similarity matrix not built yet, set parameters first.")
        graph_stat_lines = self._get_graph_stats()
        lines = ['K-NN Similarity Graph',
                 ' K: %d' % (self._k, ),
                 ' mutual: %s' % (self._mutual, )] + graph_stat_lines
        self._write_lines(img, bbox, lines, pad_px) 
def test_soft_nn_sim():
    points = np.random.rand(500, 10)
    sim_graph = NNSimGraph(points)
    img_size_wh = (300, 350)
    blank = np.zeros((img_size_wh[1], img_size_wh[0], 3), dtype=np.uint8)
    blank[:] = 250, 250, 250
    bbox = {'x':(0, img_size_wh[0]), 'y':(0, img_size_wh[1])}
    k = 5
    mututal=True
    sim_graph.set_param(k=k, mutual=mututal)

    while True:
        frame = blank.copy()
        sim_graph.draw_stats(frame, bbox)
        cv2.imshow("Soft-NN Similarity Graph", frame)
        sim_mat = sim_graph.make_img()
        cv2.imshow("Similarity Matrix", sim_mat)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break   
        elif key == ord('m'):
            mututal = not mututal
            print("Setting mutual=%s" % (mututal, ))
            sim_graph.set_param(k=k, mutual=mututal)
        elif key == ord('['):
            k += 1
            print("Setting k=%i" % (k, ))
            sim_graph.set_param(k=k, mutual=mututal)
        elif key == ord(']'):
            k = max(1, k-1)
            print("Setting k=%i" % (k, ))
            sim_graph.set_param(k=k, mutual=mututal)

# labels for slider param for different simgraph types
SIMGRAPH_PARAM_NAMES = {SimilarityGraphTypes.NN: "N-nearest",
                        SimilarityGraphTypes.EPSILON: "Epsilon",
                        SimilarityGraphTypes.FULL: "Sigma"}
# simgraph type menu options
SIMGRAPH_KIND_NAMES = {SimilarityGraphTypes.NN: "N-nearest",
                       SimilarityGraphTypes.EPSILON: "Epsilon",
                       SimilarityGraphTypes.FULL: "Full"}



if __name__ == "__main__":
    test_soft_nn_sim()
    # test_soft_nn_sim()
