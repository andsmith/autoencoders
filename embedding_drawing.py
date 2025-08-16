import time
import numpy as np
from colors import MPL_CYCLE_COLORS, COLORS
import cv2
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from color_blit import draw_color_tiles_cython as color_blit


def embed_to_pixel(locs, img_bbox):
    """
    Map embedding locations to pixel coordinates in the image,
    putting the unit square in the bounding box of the image.

    :param locs:  N x 2 array of xy embedded locations, all in the unit square
    :param img_shape:  Width, Height
    :param img_bbox: {'x': (min_x, max_x), 'y': (min_y, max_y)}
    :return: N x 2 array of pixel coordinates
    """
    bbox_width, bbox_height = img_bbox['x'][1] - img_bbox['x'][0], img_bbox['y'][1] - img_bbox['y'][0]
    locs = np.clip(locs, 0, 1)
    img_coords = np.zeros_like(locs)
    img_coords[:, 0] = locs[:, 0] * bbox_width + img_bbox['x'][0]
    img_coords[:, 1] = locs[:, 1] * bbox_height + img_bbox['y'][0]
    return (img_coords).astype(int)


def test_embed_to_pixels():
    locs = np.random.rand(1000).reshape(-1, 2)

    img_shape = (640, 480)
    bbox_1 = {'x': (20, 200), 'y': (20, 200)}
    bbox_2 = {'x': (300, 400), 'y': (300, 400)}
    bbox_3 = {'x': (500, 550), 'y': (600, 630)}
    img_coords_1 = embed_to_pixel(locs, bbox_1)
    img_coords_2 = embed_to_pixel(locs, bbox_2)
    img_coords_3 = embed_to_pixel(locs, bbox_3)
    plt.scatter(img_coords_1[:, 0], img_coords_1[:, 1], c='r', label='bbox 1')
    plt.scatter(img_coords_2[:, 0], img_coords_2[:, 1], c='g', label='bbox 2')
    plt.scatter(img_coords_3[:, 0], img_coords_3[:, 1], c='b', label='bbox 3')
    plt.legend()
    plt.axis('equal')
    plt.show()


def draw_embedding(image, embedding, images_gray, labels, colors=None):
    # Create a blank canvas
    size_wh = np.array((image.shape[1], image.shape[0]))
    colors = colors if colors is not None else (np.array(sns.color_palette("husl", 10))*255.0).astype(int).tolist()
    colors = np.array(colors, dtype=np.uint8)

    tile_size_wh = np.array((images_gray[0].shape[1], images_gray[0].shape[0]))

    # draw in this margin so nothing is out of bounds
    draw_bbox = {'x': (tile_size_wh[0]//2, size_wh[0]-tile_size_wh[0]//2),
                 'y': (tile_size_wh[1]//2, size_wh[1]-tile_size_wh[1]//2)}
    pixel_locs = embed_to_pixel(embedding, draw_bbox) - tile_size_wh // 2

    # Draw the embeddings
    drew, skipped = 0, 0
    logging.info("Drawing %d embedded tiles...", len(pixel_locs))
    for i, (loc, img) in enumerate(zip(pixel_locs, images_gray)):
        color_blit(image, loc, img, colors[labels[i]])
    logging.info("\tdrew %d tiles, skipped %d", drew, skipped)

    return image, draw_bbox


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

    def __init__(self, size, embed_xy, images_gray, labels, colors=None, bg_color=None, zoom_rate=1.0):
        """
        Initialize the EmbeddingPanZoom object.

        :param size: (width, height) of the display window
        :param embed_xy: N x 2 array of embedded (x, y) coordinates
        :param images_gray: List of grayscale images corresponding to the embeddings
        :param labels: List of labels for the embeddings
        :param colors: List of C colors for the embeddings, where C is the number of unique labels and each 
                       color is an (r,g,b)-tuple of ints in [0, 255]
        :param bg_color: Background color for the embedding image, (r,g,b) ints in [0, 255]
        """
        self.bg_color = np.array((bg_color or COLORS['OFF_WHITE_RGB']), dtype=np.uint8)
        # Internal image is bigger, what's returned is a view into the internal buffer.
        self._pad_size = np.array((images_gray.shape[1], images_gray.shape[0]))
        self._img = np.zeros((size[1] + self._pad_size[1]*2,
                              size[0] + self._pad_size[0]*2, 3), dtype=np.uint8)
        

        self.bbox = {'x': (0.0, 1.0), 'y': (0.0, 1.0)}
        self.size = size
        self.embed_xy = embed_xy
        self.images_gray = images_gray.reshape(-1, 28, 28)
        self.labels = labels
        self.colors = colors if colors is not None else (
            np.array(sns.color_palette("husl", 10))*255.0).astype(int).tolist()
        self.colors = np.array(self.colors, dtype=np.uint8)


def test_draw_embedding():
    from tests import load_mnist
    from embeddings import PCAEmbedding
    pca = PCAEmbedding()
    (x_train, y_train), _ = load_mnist()
    x_embed = pca.fit_embed(x_train)

    size = 1200, 970
    colors = MPL_CYCLE_COLORS

    epz = EmbeddingPanZoom(size, x_embed, x_train, y_train, colors)

    win_name = "Embedding Pan/Zoom test"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, size[0], size[1])
    cv2.setMouseCallback(epz.mouse_callback)

    while True:
        frame = epz.get_frame()
        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_draw_embedding()
