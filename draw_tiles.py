"""
C extension for fast digit drawing
"""
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


def test_draw_color_tiles_reference():
    logging.info("Loading data...")
    (x_train, labels), _ = load_mnist()
    image_size = (1024, 1024)
    image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    image[:] = COLORS["OFF_WHITE_RGB"]
    logging.info("Data loaded, starting PCA...")
    pca = MNISTPCA('digits', dims=2)
    locs = pca.fit_transform(x_train)
    locs -= np.min(locs, axis=0)
    locs /= np.max(locs, axis=0)
    locs_px = (locs * np.array(image_size)).astype(np.int32)

    locs_px[:, 0] = np.clip(locs_px[:, 0], 0, image_size[1] - 28)
    locs_px[:, 1] = np.clip(locs_px[:, 1], 0, image_size[0] - 28)

    logging.info("Starting draw...")
    t0 = time.perf_counter()

    draw_color_tiles_reference(image=image,
                               locs_px=locs_px,
                               gray_tiles=x_train.reshape(-1, 28, 28),
                               color_labels=labels,
                               colors=np.array(MPL_CYCLE_COLORS, dtype=np.uint8))
    t1 = time.perf_counter()
    duration = t1 - t0

    draw_rate = labels.size / duration
    max_per_30fps = int(draw_rate * (1.0/30))
    logging.info(f"Drawing {labels.size} images took {duration:.4f} seconds")
    logging.info(f"Draw rate: {draw_rate:.2f} images per second")
    logging.info("At 30 FPS, can draw %i tiles per frame." % (max_per_30fps))

    fast_image = image*0 + COLORS['OFF_WHITE_RGB']
    t2 = time.perf_counter()
    draw_color_tiles_reference(image=fast_image,
                               locs_px=locs_px[:max_per_30fps,],
                               gray_tiles=x_train.reshape(-1, 28, 28),
                               color_labels=labels[:max_per_30fps],
                               colors=np.array(MPL_CYCLE_COLORS, dtype=np.uint8))
    fast_duration = time.perf_counter() - t2

    vec_image = image*0 + COLORS['OFF_WHITE_RGB']
    t2 = time.perf_counter()
    n_vec_draw = x_train.shape[0]
    draw_color_tiles_reference(image=vec_image,
                               locs_px=locs_px[:n_vec_draw],
                               gray_tiles=x_train.reshape(-1, 28, 28)[:n_vec_draw],
                               color_labels=labels[:n_vec_draw],
                               colors=np.array(MPL_CYCLE_COLORS, dtype=np.uint8))
    vec_duration = time.perf_counter() - t2

    jax_image = image*0 + COLORS['OFF_WHITE_RGB']

    # Run first to compile
    _ = draw_color_tiles_jax_blend(image=jax_image,
                             locs_px=locs_px[:10],
                             gray_tiles=x_train.reshape(-1, 28, 28)[:10],
                             color_labels=labels[:10],
                             colors=np.array(MPL_CYCLE_COLORS, dtype=np.uint8))
    

    t2 = time.perf_counter()
    n_jax_draw = 60000
    jax_image = draw_color_tiles_jax_blend(image=jax_image,
                                     locs_px=locs_px[:n_jax_draw],
                                     gray_tiles=x_train.reshape(-1, 28, 28)[:n_jax_draw],
                                     color_labels=labels[:n_jax_draw],
                                     colors=np.array(MPL_CYCLE_COLORS, dtype=np.uint8))
    jax_duration = time.perf_counter() - t2


    cython_image = (image*0+COLORS['OFF_WHITE_RGB']).astype(np.uint8)
    t0 = time.perf_counter()
    n_cython_draw = 8000
    gray_tiles = x_train[:n_cython_draw,:].astype(np.float32).reshape(-1, 28, 28)
    colors = np.array(MPL_CYCLE_COLORS, dtype=np.uint8)
    locs = locs_px[:n_cython_draw].astype(np.int32)
    c_labels = labels[:n_cython_draw].astype(np.int32)
    output = draw_color_tiles_cython(cython_image,
                                     locs_px=locs,
                                     gray_tiles=gray_tiles,
                                     color_labels=c_labels,
                                     colors=colors)
    cython_duration = time.perf_counter() - t0


    fig, ax = plt.subplots(2, 3, figsize=(10, 10))
    ax = ax.flatten()
    ax[0].imshow(image)
    ax[0].axis('off')
    ax[0].set_title("reference:%i images,\nduration:%.3f sec,\nFPS:  %.2f." %
                    (x_train.shape[0], duration, 1.0/(duration)))

    ax[1].imshow(fast_image)
    ax[1].axis('off')
    ax[1].set_title("reference_few: %i images\nduration:%.3f sec,\nFPS:  %.2f." %
                    (max_per_30fps, fast_duration, 1.0/(fast_duration)))

    ax[2].imshow(vec_image)
    ax[2].axis('off')
    ax[2].set_title("vectorized:  %i images,\nduration %.3f sec,\nFPS:  %.2f." %
                    (n_vec_draw, vec_duration, 1.0/(vec_duration)))

    ax[3].imshow(jax_image)
    ax[3].axis('off')
    ax[3].set_title("jax:  %i images,\nduration %.3f sec,\nFPS:  %.2f." %
                    (n_jax_draw, jax_duration, 1.0/(jax_duration)))

    ax[4].imshow(cython_image)
    ax[4].axis('off')
    ax[4].set_title("cython:  %i images,\nduration %.3f sec,\nFPS:  %.2f." %
                    (n_cython_draw, cython_duration, 1.0/(cython_duration)))
    
    for axis in ax:
        axis.set_xticks([])
        axis.set_yticks([])
        # turn off
        axis.set_frame_on(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_draw_color_tiles_reference()
