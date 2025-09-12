from tkinter import font
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import logging
import seaborn as sns
import cv2


def image_from_floats(floats, small=None, big=None):
    small = floats.min() if small is None else small
    big = floats.max() if big is None else big

    values = (floats - small) / (big - small) * 255
    return values.astype(np.uint8)


def apply_colormap(floats, colormap=cv2.COLORMAP_JET):
    """
    Apply a colormap to an image.
    :param floats: 2D array of floats
    :param colormap: cv2 colormap
    :return: 3D array of uint8
    """
    image = image_from_floats(floats)
    return cv2.applyColorMap(image, colormap)


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


def get_good_point_size(n_points, bbox):
    # Copied from github:andsmith/ml_demos/spectral_clustering/util.py
    # returns: number of pixels
    if n_points > 10000:
        pts_size = 2
    elif n_points > 1000:
        pts_size = 3
    elif n_points > 100:
        pts_size = 4
    else:
        pts_size = 5
    return pts_size


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


def split_bbox(bbox, weight=0.5, orient='v'):
    """
    Split a bounding box into two parts.
    :param bbox: The bounding box to split.
    :param weight: The relative weight of the first part.
    :param orient: The orientation of the split ('v' for vertical, 'h' for horizontal).
    :return: Two bounding boxes representing the split.
    """
    if orient == 'v':
        # Vertical split
        (x1, x2), (y1, y2) = bbox['x'], bbox['y']
        split_x = int(x1 + (x2 - x1) * weight)
        rv = {'x': (x1, split_x), 'y': (y1, y2)}, {'x': (split_x, x2), 'y': (y1, y2)}
    else:
        # Horizontal split
        (x1, x2), (y1, y2) = bbox['x'], bbox['y']
        split_y = int(y1 + (y2 - y1) * weight)
        rv = {'x': (x1, x2), 'y': (y1, split_y)}, {'x': (x1, x2), 'y': (split_y, y2)}
    return rv


def scale_bbox(bbox, scale):
    """
    Scale a bounding box by a given factor.
    :param bbox: Dictionary with 'x' and 'y' keys containing tuples (min, max)
    :param scale: Tuple (scale_x, scale_y) to scale the bbox
    :return: Scaled bounding box
    """
    x_min, x_max = bbox['x']
    y_min, y_max = bbox['y']
    new_bbox = {
        'x': (int(x_min * scale[0]), int(x_max * scale[0])),
        'y': (int(y_min * scale[1]), int(y_max * scale[1]))
    }
    return new_bbox


def get_font_size(text, size_wh, incl_baseline=False, max_scale=10.0, pad=5, font=cv2.FONT_HERSHEY_DUPLEX):
    """
    Shrink font until it just fits,
    return font size, pos_xy, thickness, such that cv2.putText will put the text at the right place.
    """
    def _get_h_w(font_scale):
        (width, height), baseline = cv2.getTextSize(text, font, font_scale, 1)
        if not incl_baseline:
            return (width, height), baseline
        return (width, height+baseline), baseline

    font_scale = max_scale
    incr = 0.1
    text_wh, baseline = _get_h_w(font_scale)
    while text_wh[0] > size_wh[0]-pad*2 or text_wh[1] > size_wh[1]-pad*2:
        font_scale -= incr
        text_wh, baseline = _get_h_w(font_scale)
    text_x = (size_wh[0] - text_wh[0]) // 2
    if not incl_baseline:
        text_y = (size_wh[1] + text_wh[1]) // 2
    else:
        text_y = (size_wh[1] + text_wh[1]) // 2 - baseline
    thickness = max(1, int(font_scale*1.25))

    return font_scale, (text_x, text_y), thickness


def get_best_font_size(lines, size_wh, *args, **kwargs):
    """
    Get the largest font that will fit all lines in the given box size.
    Call get_font_size on each line and return the smallest font size, thickness.
    """
    best_scale = kwargs.get('max_scale', 10.0)
    best_thickness = 1
    for line in lines:
        scale, _, thickness = get_font_size(line, size_wh, *args, **kwargs)
        if scale < best_scale:
            best_scale = scale
            best_thickness = thickness
    return best_scale, best_thickness


def test_get_font_size():
    strings = ['#', 'test_string 1']
    box_sizes = [(100, 30), (50, 20), (200, 100), (400, 100), (300, 40)]
    pad = 10
    h = np.sum([b[1] + pad for b in box_sizes]) + pad
    w = np.max([b[0] + pad for b in box_sizes])*2 + pad
    mid = w//2
    blank_img = np.zeros((h, w, 3), dtype=np.uint8)
    boxes = {}
    x = pad
    for string in strings:
        y = pad
        boxes[string] = []
        for w, h in box_sizes:
            boxes[string].append({'x': (x, x + w), 'y': (y, y + h)})
            y += h + pad
        x = mid + pad//2

    for string in boxes:
        for box in boxes[string]:
            draw_bbox(blank_img, box, 2, color=(255, 255, 255))
            box_w, box_h = box['x'][1] - box['x'][0], box['y'][1] - box['y'][0]
            # import ipdb; ipdb.set_trace()
            incl_baseline = len(string) == 1
            font_scale, pos_xy_rel, thickness = get_font_size(string, (box_w, box_h), incl_baseline=incl_baseline)
            pos_xy = (pos_xy_rel[0] + box['x'][0], pos_xy_rel[1] + box['y'][0])
            cv2.putText(blank_img, string, pos_xy, cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    plt.imshow(blank_img)
    plt.axis('off')
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
        # interval_starts = (np.arange(n_intervals) * (interval_size + spacing) + extent[0])
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
                ax[i, j].plot(coords[:, 0], coords[:, 1], 'o-', markersize=2)
                ax[i, j].axis('off')
            _plot_at_y(intervals, -0.1)
            _plot_at_y(intervals_even, 0.1)
            ax[i, j].plot([ext[0], ext[0]], [-.3, .3], 'k-')
            ax[i, j].plot([ext[1], ext[1]], [-.3, .3], 'k-')
            ax[i, j].set_title(f"Intervals: {n_int}, size={interval_size}, spacing_frac={s_frac}\n" +
                               f"blue: {intervals[0][0]} .. {intervals[-1][1]},  " +
                               f"orange: {intervals_even[0][0]} .. {intervals_even[-1][1]}",
                               fontsize=10)
            ax[i, j].set_xlim(ext[0]-ext_range*.05, ext[1]+ext_range*.05)
            ax[i, j].set_ylim(-.3, .3)

    plt.suptitle('Extent: [%s, %s],  \nblue: fill-extent=True (uneven spacing, even endpoints),\n'
                 'orange: fill-extent=False (even spacing, uneven endpoints)' % (ext[0], ext[1]))
    plt.tight_layout()
    plt.show()


def write_lines(img, bbox, lines, pad_px, font=cv2.FONT_HERSHEY_SIMPLEX, color=(128, 128, 128)):
    """
    """
    y_span = bbox['y'][0] + pad_px, bbox['y'][1]-pad_px
    x_span = bbox['x'][0] + pad_px, bbox['x'][1]-pad_px
    txt_w, txt_h = x_span[1]-x_span[0], y_span[1]-y_span[0]

    n_lines = len(lines)
    line_y = fit_spaced_intervals(y_span, n_lines, spacing_fraction=0, fill_extent=False)
    line_wh = (txt_w, line_y[0][1]-line_y[0][0])

    font_size, font_thick = get_best_font_size(lines, line_wh, font=font)
    for i, line in enumerate(lines):
        org = (x_span[0], int(line_y[i][0] + (line_y[i][1]-line_y[i][0]+font_size)//2))
        cv2.putText(img, line, org, font, font_size, color, font_thick, cv2.LINE_AA)


def draw_bbox(image, bbox, thickness=1, inside=True, color=(128, 128, 128)):
    """
    Set pixels just inside/outside the specified bounding box to the color indicated.
    :param image:  H x W x 3  uint8 array
    :param bbox:   Dictionary with 'x' and 'y' keys containing tuples (min, max)
    :param thickness: Thickness of the bounding box lines
    :param inside: If True, color the inside of the bbox; if False, color the outside
    :param color: Color to use for the bounding box
    """

    x_min, x_max = bbox['x']
    y_min, y_max = bbox['y']
    if inside:
        x_max -= thickness-1
        y_max -= thickness-1
        x_max += thickness-1
        y_max += thickness-1
    else:
        x_min -= thickness
        y_min -= thickness
        x_max += thickness
        y_max += thickness

    image[y_min:y_min+thickness, x_min:x_max] = color
    image[y_min:y_max, x_min:x_min+thickness] = color
    image[y_min:y_max, x_max-thickness:x_max] = color
    image[y_max-thickness:y_max, x_min:x_max] = color
    return image


def test_draw_bbox():
    blank = np.zeros((10, 10, 3), dtype=np.uint8)
    bboxes = [{'bbox': {'x': (2, 7), 'y': (2, 7)}, 'thickness': 1, 'inside': True},
              {'bbox': {'x': (2, 7), 'y': (2, 7)}, 'thickness': 1, 'inside': False},
              {'bbox': {'x': (2, 7), 'y': (2, 7)}, 'thickness': 2, 'inside': True},
              {'bbox': {'x': (2, 7), 'y': (2, 7)}, 'thickness': 2, 'inside': False}]
    fig, ax = plt.subplots(2, 2, figsize=(12, 3))
    ax = ax.flatten()
    for i, bbox in enumerate(bboxes):
        image = draw_bbox(blank.copy(), **bbox)
        ax[i].imshow(image)
        ax[i].set_title(f"bbox: {bbox['bbox']}, thickness: {bbox['thickness']}, inside: {bbox['inside']}")
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_make_random_cov(n_points=10000, plot=True)
    # test_make_data(d=2, plot=True)
    # test_fit_spaced_intervals()
    test_get_font_size()
    logging.info("Tests completed successfully.")
