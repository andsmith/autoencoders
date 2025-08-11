import numpy as np
from colors import MPL_CYCLE_COLORS, COLORS
import cv2
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from test_embeddings import make_test_data


def color_blit(image, loc_px, gray, color):
    """
    Draw the imagen color with transparent background.
    :param gray: h x w floating point image (in 0, 1)
    :param color: (r,g,b) color tuple (also uint8)
    """
    h, w = gray.shape
    gray = gray.reshape((h, w, 1))
    img_y_low, img_y_high = loc_px[1], loc_px[1] + h
    img_x_low, img_x_high = loc_px[0], loc_px[0] + w
    orig_patch = image[img_y_low:img_y_high, img_x_low:img_x_high]
    patched = (1 - gray) * orig_patch + gray * color.reshape((1, 1, 3))
    image[img_y_low:img_y_high, img_x_low:img_x_high] = patched.astype(np.uint8)    


def _make_tile(size=(28, 28)):
    x, y = np.meshgrid(np.linspace(0, 1, size[0]) - .5, np.linspace(0, 1, size[1]) - .5)
    gray = 1 - 2 * np.sqrt(x**2 + y**2)
    gray = np.clip(gray, 0, 1)**.5
    return gray


def test_color_blit():
    size_wh = np.array((640, 480))
    image = np.zeros((size_wh[1], size_wh[0], 3), dtype=np.uint8)
    gray = _make_tile()
    tile_size_wh = np.array((gray.shape[1], gray.shape[0]))
    for _ in range(100):
        loc = np.random.rand(2) * (np.array(size_wh) - tile_size_wh)
        loc = loc.astype(int)
        color = np.ascontiguousarray(np.random.randint(50, 255, size=(3,), dtype=np.uint8))
        color_blit(image, loc, gray, color)
    # Check that the image has been modified
    plt.imshow(image)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


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


def test_draw_embedding():
    size = 640,480
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    image[:] = COLORS['OFF_WHITE_RGB']
    data, labels = make_test_data(d=10, n_points=300)
    from embeddings import PCAEmbedding
    pca = PCAEmbedding()
    locations = pca.fit_embed(data)
    gray_tile = _make_tile()
    n_samples = locations.shape[0]
    images_gray = [gray_tile for _ in range(n_samples)]

    image = draw_embedding(image, locations, images_gray, labels=labels)
    plt.imshow(image[:,:,::-1])
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_color_blit()
    test_draw_embedding()
    # test_embed_to_pixels()
