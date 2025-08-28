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

    def __init__(self, size, embed_xy, images_gray, labels, colors=None, bg_color=None):
        """
        Initialize the EmbeddingPanZoom object.

        :param size: (width, height) of the display window
        :param embed_xy: N x 2 array of embedded (x, y) coordinates, in the unit square (to fill image @ full zoom).
        :param images_gray: List of N grayscale images corresponding to the embeddings
        :param labels: List of labels for the embeddings (strings)
        :param colors: List of C colors for the embeddings, where C is the number of unique labels and each 
                       color is an (r,g,b)-tuple of ints in [0, 255]
        :param bg_color: Background color for the embedding image, (r,g,b) ints in [0, 255]
        :param zoom_level:  Magnification factor (1.0 puts all points in view, larger zooms in)
        """
        self.bg_color = np.array((bg_color or COLORS['OFF_WHITE_RGB']), dtype=np.uint8)
        self.size = size
        self.images_gray = np.array(images_gray).astype(np.float32)  # N x H x W
        self.tile_size = np.array((self.images_gray.shape[2], self.images_gray.shape[1]))
        self._pad_size = self.tile_size
        # Internal image is bigger, what's returned is a view into the internal buffer.
        self._blank = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        self._blank[:] = self.bg_color

        self._zoom_rate = 1.1
        self._zoom_level = 1.0  # current zoom level
        self._shutdown = False  # set True to kill
        # logical displayed area:
        self.bbox = {'x': (0.0, 1.0), 'y': (0.0, 1.0)}
        self.embed_xy = embed_xy  # 2x locations of tiles in logical coords (unit square)
        self._last_offset_px = None

        self.labels = labels
        self.label_classes = sorted(list(set(labels)))

        self.int_labels = np.array([self.label_classes.index(lbl) for lbl in labels])
        self.colors = colors if colors is not None else (
            np.array(sns.color_palette("husl", len(self.label_classes)))*255.0).astype(int).tolist()
        self.colors = np.array(self.colors, dtype=np.uint8)
        self._update_lock = Lock()

        self._frame = None
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
        logging.info(f"Zooming {direction} at {pos_xy}, new bbox: {self.bbox}, current level: {self._zoom_level}")

    def get_frame(self, px_offset=None, boxed=None, moused_over=None):
        """
        Get the current frame for display.
        :param px_offset: Pixel offset to apply to the frame, user has dragged the view by this many pixels,
            determine the corresponding shift in logical coords, apply it to the bbox for _make_frame.
            NOTE:  draws new frame if different from previous value
        :param boxed:  Dict w/ colors as keys, list indices into tiles to draw boxes around in each color as values
        :param moused_over:  Which tile(s) are currently being hovered over, drawn in mouseover color.
        """
        if not np.all(self._last_offset_px == px_offset) or self._frame is None:
            self._frame = self._make_frame(bbox=self._get_bbox(px_offset))
            self._last_offset_px = px_offset
        frame_out = self._frame.copy()
        boxed = {} if boxed is None else boxed

        def _draw_box_around_tile(ind, color):
            bbox_upper_left = self._embed_to_pixel(self.embed_xy[ind])[0]
            bbox = {'x': (bbox_upper_left[0], bbox_upper_left[0] + self.tile_size[0]),
                    'y': (bbox_upper_left[1], bbox_upper_left[1] + self.tile_size[1])}
            draw_bbox(frame_out, bbox, color=color, thickness=1)

        for box_color, boxed_inds in boxed.items():
            for ind in boxed_inds:
                _draw_box_around_tile(ind, (0,255,0))
                
        if moused_over is not None:
            _draw_box_around_tile(moused_over, color=COLOR_SCHEME['mouseover'])

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
        pos_px -= self.tile_size //2 # querying from tile corners
        pos_embed = self._pixel_to_embed(pos_px)  
        query_shift = self._zoom_level * self.tile_size / self.size /2
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
        img_coords = img_coords * self.size  - self.tile_size/2# scaled up and centered
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
        mask = (self.embed_xy[:, 0] >= bbox['x'][0]) & (self.embed_xy[:, 0] < bbox['x'][1]) & \
            (self.embed_xy[:, 1] >= bbox['y'][0]) & (self.embed_xy[:, 1] < bbox['y'][1])
        return mask

    def _make_frame(self, bbox):
        """
        Create a new frame for the current view.
        """
        frame = self._blank.copy()
        valid_mask = self._get_valid_tiles(bbox)
        color_labels = self.int_labels[valid_mask]
        images = self.images_gray[valid_mask]

        # check nothing overlaps
        embed_locs = self._embed_to_pixel(self.embed_xy[valid_mask], bbox=bbox)  
        #print(embed_locs)
        valid_mask = ((embed_locs[:, 0] >= 0) & (embed_locs[:, 0] < self.size[0] - self._pad_size[0]) &
                      (embed_locs[:, 1] >= 0) & (embed_locs[:, 1] < self.size[1] - self._pad_size[1]))
        embed_locs = embed_locs[valid_mask]
        images = images[valid_mask]
        color_labels = color_labels[valid_mask]

        color_blit(frame, embed_locs, images, color_labels, self.colors)
        return frame


class EmbedTester(object):
    def __init__(self):
        tiles, labels, embed_xy = _make_fake_data()  # self._get_real_data()

        self._selected = np.random.choice(labels.size,size=3)
        self.size =150,150#1200, 970
        colors = MPL_CYCLE_COLORS
        self.epz = EmbeddingPanZoom(self.size, embed_xy,
                                    tiles.reshape((-1, 28, 28)), labels, colors)
        self.win_name = "Embedding Pan/Zoom test"
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_name, self.size[0], self.size[1])
        cv2.setMouseCallback(self.win_name, self._mouse_callback)

    def _get_real_data(self):

        from tests import load_mnist
        from embeddings import PCAEmbedding
        pca = PCAEmbedding()

        (tiles, labels), _ = load_mnist()
        
        x_embed = pca.fit_embed(tiles)
        return tiles, labels, x_embed

    def run(self):

        self._click_px = None
        self._pan_offset = None
        self._moused_over = None  # index into points

        while True:
            frame = self.epz.get_frame(self._pan_offset, moused_over=self._moused_over)
            cv2.imshow(self.win_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def _mouse_callback(self, event, x, y, flags, param):
        pos_px = np.array((x, y))
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click_px = pos_px
            self._moused_over = None

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


def _make_fake_data():
    n_tiles = 5
    tile_size = 28
    tiles = np.random.rand(tile_size**2 * n_tiles).reshape((n_tiles, tile_size, tile_size))
    embed_locs_xy = np.array([(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9), (0.5, 0.5)])
    labels = np.array(["%i"%l for l in [0, 1, 0, 1, 0]])
    return tiles, labels, embed_locs_xy


def test_mouseover():
    tiles, labels, embed_locs_xy = _make_fake_data()
    colors = np.array(((0, 128,128), (0, 255, 0)))
    epz = EmbeddingPanZoom((150, 150), embed_locs_xy, tiles, labels, colors=colors)
    p1=(75, 75)
    mo = epz.get_moused_over(p1)
    #assert mo is not None and mo == 4, f"Expected to mouse over center tile, got {mo}"
    p2=(85, 94)
    #import ipdb; ipdb.set_trace()
    mo = epz.get_moused_over(p2)
    #assert mo is not None and mo == 3, f"Expected to mouse over bottom-left tile, got {mo}"
    frame = epz.get_frame()

    def draw_pt(pt):
        frame[pt[0]: pt[0]+4, pt[1]:pt[1]+4] = (0,255,0)

    draw_pt(p1)
    draw_pt(p2)
    cv2.imshow("Mouseover test", frame[:,:,::-1])
    cv2.waitKey(0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_mouseover()
    EmbedTester().run()
