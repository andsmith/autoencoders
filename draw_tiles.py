"""
C extension for fast digit drawing
Run stand-alone for speed tests.
"""
from email.mime import image
from turtle import color
import numpy as np
from tests import load_mnist
from colors import COLORS, MPL_CYCLE_COLORS
from pca import MNISTPCA
import logging
import time
import matplotlib.pyplot as plt
from color_blit import draw_color_tiles_cython
import jax
import jax.numpy as jnp


@jax.jit
def draw_color_tiles_jax_blend(image, locs_px, gray_tiles, color_labels, colors):
    """
    Correct alpha blending even for overlapping tiles.
    """
    image = jnp.asarray(image, jnp.float32)  # Work in float
    locs_px = jnp.asarray(locs_px)
    gray_tiles = jnp.asarray(gray_tiles, jnp.float32)
    color_labels = jnp.asarray(color_labels)
    colors = jnp.asarray(colors, jnp.float32)

    N, Th, Tw = gray_tiles.shape
    tile_colors = colors[color_labels]  # (N, 3)

    ys = locs_px[:, 1][:, None] + jnp.arange(Th, dtype=locs_px.dtype)  # (N, Th)
    xs = locs_px[:, 0][:, None] + jnp.arange(Tw, dtype=locs_px.dtype)  # (N, Tw)

    gray_exp = gray_tiles[..., None]            # (N, Th, Tw, 1)
    color_exp = tile_colors[:, None, None, :]   # (N, 1, 1, 3)

    # Gather the original pixels from the *current image*
    orig_patches = image[ys[:, :, None], xs[:, None, :], :]  # (N, Th, Tw, 3)

    # Blend for all tiles
    blended = (1.0 - gray_exp) * orig_patches + gray_exp * color_exp  # (N, Th, Tw, 3)

    # Scatter-add to combine overlapping tiles correctly
    out = image.at[ys[:, :, None], xs[:, None, :], :].set(blended)

    return jnp.asarray(out, jnp.uint8)


def draw_color_tiles_reference(image, locs_px, gray_tiles, color_labels, colors):
    """
    Color the each tile, put it in the image w/alpha blending so the tile background is transparent.
    Use the grayscale-image as the alpha channel, and grayscale_image * color as the RGB channels

    :param image:  H x W x 3 numpy image
    :param locs_px:  N x 2 array of pixel coordinates, upper-left corners of each tile's location in the image.
    :param gray_tiles: N x Th x Tw, the grayscale images, each height Th and width Tw
    :param color_labels: N array of indices into colors
    :param colors: N x 3 colors LUT to use, int8 (r,g,b) color tuples
    :return: None, modifies the image in place
    """
    h, w = gray_tiles.shape[1], gray_tiles.shape[2]
    for loc_px, grayscale, color_ind in zip(locs_px, gray_tiles, color_labels):
        img_y_low, img_y_high = loc_px[1], loc_px[1] + h
        img_x_low, img_x_high = loc_px[0], loc_px[0] + w
        gray = grayscale.reshape((h, w, 1))
        orig_patch = image[img_y_low:img_y_high, img_x_low:img_x_high]
        patched = (1 - gray) * orig_patch + gray * colors[color_ind].reshape((1, 1, 3))
        image[img_y_low:img_y_high, img_x_low:img_x_high] = patched.astype(np.uint8)


def draw_color_tiles_jax(image, locs_px, gray_tiles, color_labels, colors):
    """
    functional -> in-place
    """
    image[:] = draw_color_tiles_jax_blend(image, locs_px, gray_tiles, color_labels, colors)


def speed_comparison():
    logging.info("Loading data...")
    (x_train, labels), _ = load_mnist()
    image_size = (1024, 1024)
    blank = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    blank[:] = COLORS["OFF_WHITE_RGB"]
    logging.info("Data loaded, starting PCA...")
    pca = MNISTPCA('digits', dims=2)
    locs = pca.fit_transform(x_train)
    locs -= np.min(locs, axis=0)
    locs /= np.max(locs, axis=0)
    locs_px = (locs * np.array(image_size)).astype(np.int32)

    locs_px[:, 0] = np.clip(locs_px[:, 0], 0, image_size[1] - 28)
    locs_px[:, 1] = np.clip(locs_px[:, 1], 0, image_size[0] - 28)
    gray_tiles = x_train.reshape(-1, 28, 28)

    def _test_and_draw(ax, test_func, n_samp=0):
        """
        Test & time it, plot results, return theoretical max frames to get 30 FPS.
        """

        plot_locs = (locs_px[:n_samp] if n_samp > 0 else locs_px).astype(np.int32)
        plot_labels = (labels[:n_samp] if n_samp > 0 else labels).astype(np.int32)
        plot_tiles = (gray_tiles[:n_samp] if n_samp > 0 else gray_tiles).astype(np.float32)
        n_samp = plot_labels.size
        t0 = time.perf_counter()
        image = blank.copy()
        test_func(image=image, locs_px=plot_locs, gray_tiles=plot_tiles, color_labels=plot_labels,
                  colors=np.array(MPL_CYCLE_COLORS, dtype=np.uint8))
        t1 = time.perf_counter()
        duration = t1 - t0

        draw_rate = n_samp / duration
        max_for_30fps = int(draw_rate * (1.0/30))
        logging.info(f"Drawing {n_samp} images took {duration:.4f} seconds")
        logging.info(f"Draw rate: {draw_rate:.2f} images per second")
        logging.info("At 30 FPS, can draw %i tiles per frame." % (max_for_30fps))
        ax.imshow(image)
        ax.axis('off')
        ax.set_title("%s\nimages:  %i,duration:%.3f sec,\nFPS:  %.2f." %
                     (test_func.__name__, n_samp, duration, 1.0/(duration)))

        ax.set_xticks([])
        ax.set_yticks([])
        # turn off
        ax.set_frame_on(False)
        return max_for_30fps

    # prime jax
    _ = draw_color_tiles_jax(image=blank.copy(),
                             locs_px=locs_px[:10],
                             gray_tiles=gray_tiles[:10],
                             color_labels=labels[:10],
                             colors=np.array(MPL_CYCLE_COLORS, dtype=np.uint8))

    fig, ax = plt.subplots(2, 3, figsize=(10, 10))
    ax = ax.flatten()
    m_fps = _test_and_draw(ax[0], draw_color_tiles_reference)
    _ = _test_and_draw(ax[1], draw_color_tiles_reference, n_samp=int(m_fps))
    j_fps = _test_and_draw(ax[2], draw_color_tiles_jax)
    _ = _test_and_draw(ax[3], draw_color_tiles_jax, n_samp=int(j_fps))
    c_fps = _test_and_draw(ax[4], draw_color_tiles_cython)
    _ = _test_and_draw(ax[5], draw_color_tiles_cython, n_samp=int(c_fps))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    speed_comparison()
