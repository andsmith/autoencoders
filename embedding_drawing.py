import os
import sys
import time
import numpy as np
from colors import MPL_CYCLE_COLORS, COLORS, COLOR_SCHEME
import cv2
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from threading import Lock, Thread
from color_blit import draw_color_tiles_cython as color_blit
from util import draw_bbox
# import keyboard


class EmbeddingPanZoom(object):
    """
    Show colored tiles at the embedded locations.  Be responsive enough to run at 30 fps.

    Expose a bounding-box defining the extent/view of the current rendered image and provide methods to move it,
            pan_to(offset_xy)
            zoom(factor)

    Allow callbacks for mouse events when they happen over one or more of the tiles
        - click/unclick 
        - mouse-over/-out


    """
    _MAX_TILES_PER_UPDATE = 8000

    def __init__(self, size, embed_xy, images_gray, labels, image_colors=None):
        """
        Initialize the EmbeddingPanZoom object.

        :param size: (width, height) of the display window
        :param embed_xy: N x 2 array of embedded (x, y) coordinates, in the unit square (to fill image @ full zoom).
        :param images_gray: List of N grayscale images corresponding to the embeddings
        :param labels: List of labels for the embeddings (strings)
        :param image_colors: List of C colors for the embeddings, where C is the number of unique labels and each 
                       color is an (r,g,b)-tuple of ints in [0, 255]
        """
        self.bg_color = np.array((COLOR_SCHEME['bkg']), dtype=np.uint8)
        self.images_gray = np.array(images_gray).astype(np.float32)  # N x H x W
        self.tile_size = np.array((self.images_gray.shape[2], self.images_gray.shape[1]))
        self._pad_size = self.tile_size
        self._size_orig = size
        # shrink controls tile size
        self._shrink_level = 0  # subtract 1 pixel from tile size until minimum
        self._min_tile_size = 4
        self.shrink(0)  # initialize frame

        # zoom controls view window:
        self._zoom_rate = 1.1
        self._zoom_level = 1.0  # current zoom level
        n_tiles = self.images_gray.shape[0]
        self._sample_step = 0.02
        self._sample_rate = min(1.0, max(self._sample_step, 50000 / n_tiles)) # downsample for display
        logging.info(f"Initial sample rate {self._sample_rate}, showing {int(n_tiles*self._sample_rate)} tiles.")
        self._sample = np.arange(self.images_gray.shape[0], dtype=int)
        self._unused = np.array([], dtype=int)
        if self._sample_rate<1.0:
            self._sample = np.random.choice(self._sample, size=int(n_tiles*self._sample_rate), replace=False)
            self._unused = np.setdiff1d(self._unused, self._sample)

        # logical displayed area:
        self.bbox = {'x': (0.0, 1.0), 'y': (0.0, 1.0)}
        self.embed_xy = embed_xy  # 2x locations of tiles in logical coords (unit square)
        self._last_offset_px = None

        self.labels = labels
        self.label_classes = sorted(list(set(labels)))

        self.int_labels = np.array([self.label_classes.index(lbl) for lbl in labels])
        self.colors = image_colors if image_colors is not None else (
            np.array(sns.color_palette("husl", len(self.label_classes)))*255.0).astype(int).tolist()
        self.colors = np.array(self.colors, dtype=np.uint8)
        self._update_lock = Lock()

        self._frame = None
        self._shutdown = False  # set True to kill
        self._tree = KDTree(self.embed_xy)
        logging.info("Embedding pan-zoom initialized with %i points.", len(self.embed_xy))

    def zoom_at(self, direction, pos_xy=None, pos_px=None):
        """
        :param pos_xy: logical (x, y) coordinates to zoom into/away from (must be not none if pos_px is None)
        :param pos_px: pixel (x, y) coordinates to zoom into/away from  (must be not none if pos_xy is None)
        :param direction: 'in' to zoom in, 'out' to zoom out
        """
        self._frame = None  # force redraw
        if direction > 0:
            zoom_mul = 1.0 / self._zoom_rate
        elif direction < 0:
            zoom_mul = self._zoom_rate
        self._zoom_level *= zoom_mul
        pos_xy = pos_xy if pos_xy is not None else self._pixel_to_embed(pos_px)
        left_len, right_len = pos_xy[0] - self.bbox['x'][0], self.bbox['x'][1] - pos_xy[0]
        top_len, bottom_len = pos_xy[1] - self.bbox['y'][0], self.bbox['y'][1] - pos_xy[1]
        self.bbox = {'x': (pos_xy[0] - left_len * zoom_mul, pos_xy[0] + right_len * zoom_mul),
                     'y': (pos_xy[1] - top_len * zoom_mul, pos_xy[1] + bottom_len * zoom_mul)}
        # logging.info(f"Zooming {direction} at {pos_xy}, new bbox: {self.bbox}, current level: {self._zoom_level}")

    def get_frame(self, px_offset=None, color_boxes=None, moused_over=None):
        """
        Get the current frame for display.
        :param px_offset: Pixel offset to apply to the frame, user has dragged the view by this many pixels,
            determine the corresponding shift in logical coords, apply it to the bbox for _make_frame.
            NOTE:  draws new frame if different from previous value
        :param boxed:  Dict w/ colors as keys, list indices into tiles to draw boxes around in each color as values
        :param moused_over:  Which tile(s) are currently being hovered over, drawn in mouseover color.
        """
        px_offset = (0, 0) if px_offset is None else px_offset
        if not np.all(self._last_offset_px == px_offset) or self._frame is None:
            self._frame = self._make_frame(px_offset, color_boxes=color_boxes, moused_over=moused_over)
            self._last_offset_px = px_offset
        frame_out = self._frame.copy()

        if moused_over is not None:
            self._draw_box_around_tile(frame_out, moused_over, color=COLOR_SCHEME['mouseover'], thickness=3)

        return frame_out

    def pan_by(self, offset_px):
        """
        Pan the view by a pixel offset.
        """
        self.bbox = self._get_bbox(offset_px)

    def get_moused_over(self, pos_px):
        """
        Get the index of the point(s) currently moused over, if any.
        :param pos_px:  (x,y) pixel coordinates of the mouse position
        :return:  index (or list of indices) of the point(s) moused over, or None if none
        """
        pos_px -= self.tile_size // 2  # querying from tile corners
        pos_embed = self._pixel_to_embed(pos_px)
        query_shift = self._zoom_level * self.tile_size / self.size / 2
        _, inds = self._tree.query(pos_embed+query_shift, k=1)
        if inds.size > 0:
            sample_pos_px = self._embed_to_pixel(self.embed_xy[inds])[0]
            if (np.abs(sample_pos_px[0] - pos_px[0]) < self.tile_size[0]//2 and
               np.abs(sample_pos_px[1] - pos_px[1]) < self.tile_size[1]//2):
                return inds
        return None

    def _get_bbox(self, px_offset):
        """
        Given current focus and zoom, what are the x,y translations corresponding to the xy pixel offsets in px_offset
        :param px_offset:  (x, y) pixels (wrt the current view) to shift the current bounding box view.
        """
        if px_offset is None:
            return self.bbox

        x_shift = px_offset[0] * self._zoom_level / self.size[0]
        y_shift = px_offset[1] * self._zoom_level / self.size[1]

        new_bbox = {
            'x': (self.bbox['x'][0] - x_shift, self.bbox['x'][1] - x_shift),
            'y': (self.bbox['y'][0] - y_shift, self.bbox['y'][1] - y_shift)
        }
        return new_bbox

    def _embed_to_pixel(self, locs, bbox=None):
        """
        Map embedding locations to pixel coordinates in the image,
        putting the unit square in the bounding box of the image.

        :param locs:  N x 2 array of xy embedded locations, all in the unit square
        :param img_shape:  Width, Height
        :return: N x 2 array of pixel coordinates
        """
        bbox = bbox if bbox is not None else self.bbox
        bbox_width, bbox_height = bbox['x'][1] - bbox['x'][0], bbox['y'][1] - bbox['y'][0]
        img_coords = np.array(locs).reshape(-1, 2) - np.array((bbox['x'][0], bbox['y'][0]))
        img_coords /= np.array((bbox_width, bbox_height))  # now in unit square
        img_coords = img_coords * self.size - self.tile_size/2  # scaled up and centered
        return (img_coords.round()).astype(int)

    def _pixel_to_embed(self, locs_px, bbox=None):
        """
        Map pixel coordinates in the image back to embedding locations.

        :param locs_px:  N x 2 array of pixel coordinates
        :param img_shape:  Width, Height
        :return: N x 2 array of xy embedded locations, all in the unit square
        """
        bbox = bbox if bbox is not None else self.bbox
        bbox_width, bbox_height = bbox['x'][1] - bbox['x'][0], bbox['y'][1] - bbox['y'][0]
        locs = (np.array(locs_px)) / np.array(self.size)  # now in unit square
        locs = locs * np.array((bbox_width, bbox_height)) + np.array((bbox['x'][0], bbox['y'][0]))
        return locs

    def _get_valid_tiles(self, bbox):
        """
        Get a set of tiles in the current bounding box.
        """
        sampled_mask = (self.embed_xy[self._sample, 0] >= bbox['x'][0]) & (self.embed_xy[self._sample, 0] < bbox['x'][1]) & \
            (self.embed_xy[self._sample, 1] >= bbox['y'][0]) & (self.embed_xy[self._sample, 1] < bbox['y'][1])
        mask = np.zeros(self.embed_xy.shape[0], dtype=bool)
        mask[self._sample] = sampled_mask
        return mask

    def shrink(self, direction):
        if direction == 0:
            self.size = self._size_orig
            # Internal image is bigger, what's returned is a view into the internal buffer.
            self._blank = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
            self._blank[:] = self.bg_color

        if direction > 0:
            self._shrink_level = min(self._shrink_level + 1, self.tile_size[0] - 2)
        else:
            self._shrink_level = max(self._shrink_level - 1, 0)
        logging.info(f"Shrink level now {self._shrink_level}, tile size {self.tile_size[0] - self._shrink_level}")
        # TODO: implement shrinking of tiles (drawing to larger image then scaling down)

    def sample(self, direction):
        if direction < 0:
            self._sample_rate = max(self._sample_step, self._sample_rate - self._sample_step)
            n_keep = int(self.images_gray.shape[0]*self._sample_rate)
            self._sample = np.random.choice(self._sample, size=n_keep, replace=False)
        else:
            self._sample_rate = min(1.0, self._sample_rate + self._sample_step)
            n_add = int(self.images_gray.shape[0]*self._sample_rate) - self._sample.size
            unused = np.setdiff1d(np.arange(self.images_gray.shape[0]), self._sample)
            self._sample = np.concatenate([self._sample, np.random.choice(unused, size=n_add, replace=False)])
        self._frame = None  # force redraw
        logging.info(f"Sample rate now {self._sample_rate}, showing {self._sample.size} tiles.")

    def _make_frame(self, px_offset, color_boxes=None, moused_over=None):
        """
        Create a new frame for the current view.
        """
        bbox = self._get_bbox(px_offset)
        frame = self._blank.copy()
        valid_inds = np.where(self._get_valid_tiles(bbox))[0]
        color_labels = self.int_labels[valid_inds]
        images = self.images_gray[valid_inds]

        # check nothing overlaps
        embed_locs = self._embed_to_pixel(self.embed_xy[valid_inds], bbox=bbox).reshape(-1, 2)
        valid_mask = (((embed_locs[:, 0] >= 0) & (embed_locs[:, 0] < self.size[0] - self._pad_size[0]) &
                      (embed_locs[:, 1] >= 0) & (embed_locs[:, 1] < self.size[1] - self._pad_size[1])))
        valid_inds = valid_inds[valid_mask]
        embed_locs = embed_locs[valid_mask]
        images = images[valid_mask]
        color_labels = color_labels[valid_mask]
        color_blit(frame, embed_locs, images, color_labels, self.colors)

        boxed = {} if color_boxes is None else color_boxes
        valid_inds = set(valid_inds.tolist())
        for box_color, boxed_inds in boxed.items():
            for ind in boxed_inds:
                if ind in valid_inds:
                    self._draw_box_around_tile(frame, ind, box_color, thickness=2, offset=px_offset)
        return frame

    def _draw_box_around_tile(self, frame, ind, color, thickness, offset=(0, 0)):
        bbox_upper_left = self._embed_to_pixel(self.embed_xy[ind])[0]
        bbox = {'x': (bbox_upper_left[0]+offset[0], bbox_upper_left[0] + self.tile_size[0]+offset[0]),
                'y': (bbox_upper_left[1]+offset[1], bbox_upper_left[1] + self.tile_size[1]+offset[1])}
        draw_bbox(frame, bbox, color=color, thickness=thickness, inside=False)


class EmbedTester(object):
    def __init__(self):

        self._size = 1920, 1000  # 500,500
        self.size = (np.array([1920, 1000]) * 1.0).astype(int)

        from embed import LatentRepEmbedder
        self.samples = []

        self.embedder = LatentRepEmbedder.from_filename(sys.argv[1])

        self._box_colors = {COLOR_SCHEME['a_source']: [],
                            COLOR_SCHEME['a_dest']: [],
                            COLOR_SCHEME['a_input']: [],
                            COLOR_SCHEME['a_output']: []}
        self._box_fill_seq = ['a_source', 'a_dest','a_input']
        colors = MPL_CYCLE_COLORS
        self.epz = EmbeddingPanZoom(self.size, self.embedder._embedded_train_data,
                                    self.embedder._images.reshape((-1, 28, 28)), self.embedder._digits, colors)
        self.win_name = "Embedding Pan/Zoom test"
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        #cv2.resizeWindow(self.win_name, self.size[0], self.size[1])
        cv2.setMouseCallback(self.win_name, self._mouse_callback)

    def _get_real_data(self, load_fn, n_max=2000):

        from embeddings import PCAEmbedding
        pca = PCAEmbedding()

        (tiles, labels), _ = load_fn()
        if n_max is not None:
            tiles = tiles[:n_max]
            labels = labels[:n_max]
        x_embed = pca.fit_embed(tiles)
        return tiles, labels, x_embed

    def _set_sample(self, index):
        self.samples.append(index)
        box_name = self._box_fill_seq[len(self.samples) - 1]
        box_color= COLOR_SCHEME[box_name]
        self._box_colors[box_color].append(index)
        self.epz._frame = None  # force redraw
        if len(self.samples) == 3:
            self._do_analogy()

    def _pop_sample(self):
        if self.samples:
            index = self.samples.pop()
            box_name = self._box_fill_seq[len(self.samples)]
            box_color= COLOR_SCHEME[box_name]
            self._box_colors[box_color].remove(index)
            self.epz._frame = None  # force redraw
            
    def _do_analogy(self):
        a_source_code = self.embedder._codes[self.samples[0]]
        a_dest_code = self.embedder._codes[self.samples[1]]
        a_input_code = self.embedder._codes[self.samples[2]]
        a_output_code = a_input_code + (a_dest_code - a_source_code)
        # Perform analogy operation here
        a_source_img = self.embedder._images[self.samples[0]]   
        a_dest_img = self.embedder._images[self.samples[1]]
        a_input_img = self.embedder._images[self.samples[2]]
        a_output_img = self.embedder._autoencoder.decode_samples(a_output_code.reshape(1,-1))
        fig, ax = plt.subplots(2,2)
        ax = ax.flatten()

        ax[0].imshow(a_source_img.squeeze().reshape(28,28), cmap='gray')
        ax[0].set_title('A')
        ax[1].imshow(a_dest_img.squeeze().reshape(28,28), cmap='gray')
        ax[1].set_title('AA')
        ax[2].imshow(a_input_img.squeeze().reshape(28,28), cmap='gray')
        ax[2].set_title('B')
        ax[3].imshow(a_output_img.squeeze().reshape(28,28), cmap='gray')
        ax[3].set_title('??')
        fig.suptitle('Analogy: A is to AA as B is to ??', fontsize=16)
        plt.show()

    def run(self):

        self._click_px = None
        self._pan_offset = None
        self._moused_over = None  # index into points

        while True:
            # if not np.all([len(self._box_colors[color]) == 0 for color in self._box_colors]):
            #     import ipdb; ipdb.set_trace()
            frame = self.epz.get_frame(self._pan_offset, moused_over=self._moused_over, color_boxes=self._box_colors)
            cv2.imshow(self.win_name, frame[:, :, ::-1])
            k = cv2.waitKey(1)
            if k & 0xFF == ord('q'):
                break
            elif k & 0xFF == ord(','):
                self.epz.shrink(-1)
            elif k & 0xFF == ord('.'):
                self.epz.shrink(1)

            elif k & 0xFF == ord(';'):
                self.epz.sample(-1)
            elif k & 0xFF == ord('\''):
                self.epz.sample(1)

            elif k & 0xFF == ord('c'):
                self._pop_sample()

    def _mouse_callback(self, event, x, y, flags, param):
        pos_px = np.array((x, y))
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click_px = pos_px
            if self._moused_over is not None:
                self._set_sample(self._moused_over)

        elif event == cv2.EVENT_LBUTTONUP:
            self._click_px = None
            self.epz.pan_by(self._pan_offset)
            self._pan_offset = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._click_px is not None:
                self._pan_offset = pos_px - self._click_px
            else:
                self._moused_over = self.epz.get_moused_over(pos_px)

        elif event == cv2.EVENT_MOUSEWHEEL:
            direction = int(np.sign(flags))
            self.epz.zoom_at(direction, pos_px=pos_px)


def _make_fake_data(n=6):
    embed_locs_xy = np.array([(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.6, 0.6), (0.5, 0.5), (0.9, 0.9)])
    n_tiles = embed_locs_xy.shape[0]
    labels = np.array(["%i" % (l % 2,) for l in range(n_tiles)])

    tile_size = 28
    tiles = np.random.rand(tile_size**2 * n_tiles).reshape((n_tiles, tile_size, tile_size))

    return tiles[:n], labels[:n], embed_locs_xy[:n]


def test_mouseover():
    tiles, labels, embed_locs_xy = _make_fake_data()
    colors = np.array(((0, 128, 128), (0, 255, 0)))
    epz = EmbeddingPanZoom((150, 150), embed_locs_xy, tiles, labels, colors)
    p1 = (75, 75)
    mo = epz.get_moused_over(p1)
    assert mo is not None and mo == 4, f"Expected to mouse over center tile, got {mo}"
    p2 = (85, 94)
    mo = epz.get_moused_over(p2)
    assert mo is not None and mo == 3, f"Expected to mouse over bottom-left tile, got {mo}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    #test_mouseover()
    EmbedTester().run()
