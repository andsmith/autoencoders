"""
Class for rendering views of the embedding space and
interacting with it:
   * show a visually clean representation of the embedding
   * 


Usage:   

      er = EmbeddingRenderer(embedding_function, images, locations, labels, min_dist=1.0)

      frame1 = er.render_embedding(blank_img, view_bbox1)
      frame2 = er.render_embedding(blank_img, view_bbox2, keep_visible_icons=True)

      start, end = # two points in the embedding space (in view_bbox1,and 2)

      frame_between_points, sampled_points = er.render_interpolated_path(frame1, start, end, density=0.5)

      transf = end - start

      test_point = # apply the transformation to this new test point

      frame_extrapolated = er.render_extrapolated_path(frame2, test_point, transf, density=0.5)

      indices = er.nearest_neighbors(point, n=3)

      indices_in_hull = er.inside_hull(point_list)
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
from scipy.spatial import ConvexHull
from mnist import MNISTData
from embeddings import PassThroughEmbedding, PCAEmbedding, TSNEEmbedding, UMAPEmbedding
from util import PointSet2d


class Icon(object):
    def __init__(self, image, x_in, x_latent, x_embed):
        self.image = image
        self.c = np.array((image.shape[1] / 2, image.shape[0] / 2))  # Center of the icon
        self.x_in = x_in
        self.x_latent = x_latent
        self.x_embedded = x_embed


class MNISTIcon(Icon):
    def __init__(self, x_in, x_latent, x_embed, digit, color=(255, 255, 255)):
        """
        Create an icon from a MNIST digit.
        :param x: array of shape (d_input,), where d_input is the flattened image size.
        """
        self.digit = digit
        self.color = color
        image = x_in.reshape(28, 28)  # don't cast yet, use for coloring RGB image
        super().__init__(image, x_in, x_latent, x_embed)

    @staticmethod
    def icons_from_data(data, latent_data, embed_data, digit_labels):
        """
        Create icons from a dataset of images.
        :param data: array of shape (n_samples, d_input), where d_input is the flattened image size.
        :param latent_data: array of shape (n_samples, d_latent), where d_latent is the latent space size.
        :param embed_data: array of shape (n_samples, d_embed), where d_embed is the embedding space size.
        :return: list of Icon objects
        """
        color_palette = plt.cm.gist_ncar(np.linspace(0, 1, 10))[:, :3]  # 10 colors for digits 0-9

        colors = (color_palette[digit_labels] * 255).astype(np.uint8)  # Scale to 0-255
        embed_data = np.zeros((data.shape[0], 2)) if embed_data is None else embed_data
        icons = [MNISTIcon(data[i], latent_data[i], embed_data[i], digit_labels[i], colors[i])
                 for i in range(data.shape[0])]
        return icons


class EmbeddingRenderer(object):
    """
    Renders the embedding space and provides interaction methods.

    Strategy:
        - Draw as many icons as possible within a given view of the embedding space.
        - Don't draw overlapping icons, up to the avg. max density determined by the min_dist.
        - For a given view (bounding box):
            - Optionally remember a list of icons drawn in the previous view, for continuity.
            - Filter this list for the new view & minimum distance (since the scale can change).
            - Extend the list with remaining icons in bounds, adding greedily when not within the
              minimum distance of any existing icon.
            - Render all icons, return the list of rendered icons (indices).
    """

    def __init__(self, embedder, icons):
        logging.info("Creating EmbeddingRenderer with %d icons.", len(icons))
        self.embedder = embedder
        self.icons = icons
        self._draw_order_lut = {}
        self._draw_priority = -1 + np.zeros(len(icons), dtype=np.int32)
        self._embed_locs = np.array([icon.x_embedded for icon in icons])

        self._icons_displayed_last = None  # set


class MNISTEmbedRenderer(EmbeddingRenderer):
    """
    Renderer for MNIST digits, using the embedding of the digits.
    """

    def __init__(self, embedder, x_train):
        icons = MNISTIcon.icons_from_data(x_train, embedder.points, embedder.points_2d, embedder.class_labels)
        super().__init__(embedder, icons)
        self._embed_locs = np.array([icon.x_embedded for icon in icons])
        self._raw_locs = np.array([icon.x_in for icon in icons])
        self._digits = np.array([icon.digit for icon in icons])
        self._images = [icon.image for icon in icons]

    def render_embedding(self, image, view_bbox, keep_visible_icons=False, spread=1.0):
        """
        Render the embedding space in the given view bounding box.
        :param image: input image to draw on, or a tuple (width, height)
        :param view_bbox: bounding box in the embedding space (dict with 'x' and 'y' keys)
        :param keep_visible_icons: if True, keep icons that are already visible in the previous view
        :param min_dist: minimum distance between icons in the embedding space
            NOTE:  This is wrt the full view, will be decreased to keep min_dist/width constant.
        :return: rendered frame with icons drawn
        """

        def min_dist_from_width(width_px):
            # calculate the minimum separation so images won't overlap when fully zoomed out
            n_imgs = width_px/28
            min_dist = 1.0 / n_imgs if n_imgs > 0 else 0
            return min_dist

        image = np.zeros((image[1], image[0], 3), dtype=np.uint8) if isinstance(image, tuple) else image
        width, height = image.shape[1], image.shape[0]
        view_w, view_h = view_bbox['x'][1] - view_bbox['x'][0], view_bbox['y'][1] - view_bbox['y'][0]
        zoom = 1.0 / max(view_w, view_h)
        min_dist = min_dist_from_width(width) / zoom
        old_icon_inds = self._icons_displayed_last if keep_visible_icons and self._icons_displayed_last is not None else set()
        all_icon_inds = set(range(len(self.icons)))
        usable_icon_inds = all_icon_inds - old_icon_inds
        usable_icon_inds, _ = self._filter_bbox(view_bbox, usable_icon_inds)
        old_icon_inds, _ = self._filter_bbox(view_bbox, old_icon_inds)
        old_icon_inds, _ = self._filter_min_dist(old_icon_inds, min_dist)

        icons_to_draw = list(self._add_icons(old_icon_inds, usable_icon_inds, min_dist)[0])

        for icon_ind in icons_to_draw:
            icon = self.icons[icon_ind]
            loc = self.embedder.scale_to_bbox(icon.x_embedded, view_bbox, (width, height)).flatten() - icon.c
            loc = loc.astype(np.int32)
            loc[0] = np.clip(loc[0], 0, width - icon.image.shape[1])
            loc[1] = np.clip(loc[1], 0, height - icon.image.shape[0])
            image[loc[1]:loc[1] + icon.image.shape[0], loc[0]:loc[0] + icon.image.shape[1]] = (icon.image[:, :, np.newaxis]
                                                                                               * icon.color.reshape(1, 1, 3)).astype(np.uint8)
        self._icons_displayed_last = set(icons_to_draw)
        return image

    def _filter_bbox(self, bbox, icon_inds):
        """
        Filter icons that are outside the bounding box.
        :param bbox: bounding box in the embedding space (dict with 'x' and 'y' keys)
        :param icon_inds: set of icon indices to filter
        :return: tuple of (used icon indices, removed bounding box)
        """
        if len(icon_inds) == 0:
            return set(), set()
        good_inds = set()

        icon_inds = np.array(list(icon_inds))

        icon_embed_locs = np.array([self.icons[i].x_embedded for i in icon_inds])
        valid = (icon_embed_locs[:, 0] >= bbox['x'][0]) & (icon_embed_locs[:, 0] <= bbox['x'][1]) & \
                (icon_embed_locs[:, 1] >= bbox['y'][0]) & (icon_embed_locs[:, 1] <= bbox['y'][1])
        good_inds = set((icon_inds[valid]).tolist())
        removed_inds = set((icon_inds[~valid]).tolist())

        return good_inds, removed_inds

    def _filter_min_dist(self, icon_inds, min_dist):
        """
        Filter icons that are too close to each other based on the minimum distance ratio.
        :param icon_inds: set of icon indices to filter
        :return: tuple of (used icon indices, removed distance)
        """
        if len(icon_inds) == 0:
            return set(), set()
        good_inds, unused_inds = self._add_icons(set(), icon_inds, min_dist)
        return good_inds, unused_inds

    def _add_icons(self, icon_set, candidates, min_dist):
        """
        Add icons from the candidates set to the icon_set, ensuring they are not too close to existing icons.
        """
        icon_inds = np.array(list(icon_set))
        cand_inds = np.array(list(candidates))
        existing_locs = np.array([self.icons[i].x_embedded for i in icon_inds]).reshape(-1, 2)
        unused = set(candidates)
        point_set = PointSet2d(min_dist)
        point_set.add_points(existing_locs)
        # TODO: Randomize?
        pts_added = []
        for cand_ind in cand_inds:
            icon = self.icons[cand_ind]

            if len(icon_set) > 0:
                existing_locs = np.array([self.icons[i].x_embedded for i in icon_set]).reshape(-1, 2)
                dists = np.linalg.norm(existing_locs - icon.x_embedded.reshape(1, 2), axis=1)

            if point_set.add_point(icon.x_embedded):
                icon_set.add(cand_ind)
                unused.remove(cand_ind)

        return icon_set, unused


def test_embedding_renderer(n_sample=10000):
    blank = np.zeros((1000, 1000, 3), dtype=np.uint8)

    data = MNISTData()
    sample_inds = np.random.choice(len(data.x_train), n_sample, replace=False)
    x_train = data.x_train[sample_inds].reshape(n_sample, -1)
    y_train = data.y_train[sample_inds]
    labels = np.argmax(y_train, axis=1)
    x_latent = x_train  # For simplicity, use the input as latent space
    embed = PCAEmbedding(x_latent, class_labels=labels)
    rend = MNISTEmbedRenderer(embed, x_train)

    def view(img):
        cv2.imshow("Embedding", img)
        cv2.waitKey(0)

    view(rend.render_embedding(blank.copy(), {'x': (0.0, 1.0), 'y': (0.0, 1.0)}, spread=1.0))
    view(rend.render_embedding(blank.copy(), {'x': (0.4, 0.6), 'y': (0.4, 0.6)}, spread=1.0))
    view(rend.render_embedding(blank.copy(), {'x': (0.45, 0.55), 'y': (0.45, 0.65)}, spread=1.0))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_embedding_renderer()  # Run the test function to verify the implementation
