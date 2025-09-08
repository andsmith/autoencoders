
import numpy as np


def diff_img(digit_true, digit_hat, thresh=0.1):
    """
    Turn two monochrome images into a color difference image.
    False positive pixels are red, false negative pixels are blue.
    """
    digit_true = digit_true.reshape(-1)
    digit_hat = digit_hat.reshape(-1)

    agree = digit_true * digit_hat  # in white
    false_pos = np.maximum(digit_hat - digit_true, 0) > thresh  # in red
    false_neg = np.maximum(digit_true - digit_hat, 0) > thresh  # in blue

    img = np.zeros((28*28, 3), dtype=np.uint8)
    for c in range(3):
        img[:, c] = agree * 255  # White channel
    img[false_pos, 0] = 255  # Red channel
    img[false_neg, 2] = 255  # Blue channel
    img = img.reshape((28, 28, 3))
    return img


def make_img(digit):
    """
    Convert a monochrome image to RGB.
    """
    digit = digit.reshape((28, 28)) * 255  # Reshape and scale to 0-255
    digit = digit.astype(np.uint8)
    img = np.stack((digit, digit, digit), axis=-1)  # Convert to RGB
    return img


def get_mosaic_shape(n_imgs, mosaic_aspect=1.7):

    n_cols = int(np.ceil(np.sqrt(n_imgs * mosaic_aspect)))
    n_rows = int(np.ceil(n_imgs / n_cols))
    return n_rows, n_cols


def make_digit_mosaic(imgs, mosaic_aspect=1.7, bkg=None):
    n_rows, n_cols = get_mosaic_shape(len(imgs), mosaic_aspect)
    img_side = 28
    img_flat_size = img_side * img_side
    n_channels = imgs[0].shape[-1] if imgs[0].ndim == 3 else None
    img = np.zeros((n_rows*img_side, n_cols*img_side, n_channels), dtype=np.uint8)\
        if n_channels else np.zeros((n_rows*img_side, n_cols*img_side), dtype=np.uint8)
    if bkg is not None:
        img[:, :, ...] = bkg
    #print(f"Creating mosaic with {n_rows} rows and {n_cols} columns, img shape {img.shape}")

    for i, im in enumerate(imgs):
        x = i % n_cols
        y = i // n_cols
        if im.shape == (img_flat_size,):
            im = im.reshape((img_side, img_side))
        elif im.shape == (img_flat_size, 3):
            im = im.reshape((img_side, img_side, 3))
        # try:
        if n_channels is None:
            img[y*img_side:(y+1)*img_side, x*img_side:(x+1)*img_side] = im
        else:
            img[y*img_side:(y+1)*img_side, x*img_side:(x+1)*img_side, :] = im
        # except Exception:
        #    print(f"Error assembling image {i}: {im.shape}")
        #    continue
    img = np.clip(img, 0, 255)
    return img


def make_heterog_mosaic(size, imgs, pad_px=6, bkg_color=0):
    """
    Aragnge images of different sizes into a mosaic as tightly as possible using the following algorithm:
    1. Sort images by size (largest to smallest)
    2. Until all images are placed:  Go downwards placing images in order until the next one won't fit.
       skip images until one fits, continue downwards.
    3.  When nothing fits, move to the right and repeat step 2.

    If out of room, throw exception.

    :param imgs: list of images (numpy arrays)
    :param mosaic_aspect: desired aspect ratio of the mosaic (width/height)
    :returns img: combined image
             bboxes: list of bounding boxes for each image in the mosaic
    """
    n_imgs = len(imgs)
    if n_imgs == 0:
        raise ValueError("No images provided.")
    img_sides = [max(im.shape[0], im.shape[1]) for im in imgs]
    # Sort images by size (largest to smallest)
    sorted_indices = np.argsort(img_sides)[::-1]
    imgs = [imgs[i] for i in sorted_indices]

    # Determine mosaic size
    mosaic_width, mosaic_height = size
    print(f"Creating heterog mosaic with target size {mosaic_width}x{mosaic_height}.")
    if len(imgs[0].shape) == 2:
        img = np.ones((mosaic_height, mosaic_width), dtype=np.uint8) * 255
    else:
        img = np.ones((mosaic_height, mosaic_width, imgs[0].shape[2]), dtype=np.uint8) * 255  # White background

    img[:, :, ...] = bkg_color

    x, y = pad_px, pad_px
    row_height = 0
    placed = [False] * n_imgs
    bboxes = []
    while not all(placed):
        placed_in_row = False
        for i in range(n_imgs):
            if placed[i]:
                continue
            img_wh = imgs[i].shape[1], imgs[i].shape[0]
            if x + img_wh[0] + pad_px <= mosaic_width and y + img_wh[1] + pad_px <= mosaic_height:
                # Place image
                im = imgs[i]
                img[y:y+img_wh[1], x:x+img_wh[0], ...] = im[:, :, ...]
                placed[i] = True
                x += img_wh[0] + pad_px
                bboxes.append({'x': (x, x + img_wh[0]), 'y': (y, y + img_wh[1])})
                row_height = max(row_height, img_wh[1])
                placed_in_row = True
        if not placed_in_row:
            # Move to next row
            x = pad_px
            y += row_height + pad_px
            row_height = 0
            if y >= mosaic_height - pad_px:
                raise ValueError("Not enough room to place all images in the mosaic. Try increasing the mosaic size.")
    return img, bboxes


def make_heterog_mosaic_autosize(imgs, mosaic_aspect=1.7, pad_px=6, bkg_color=0):
    min_side = min(max(im.shape[0], im.shape[1]) for im in imgs)
    init_size = (int(min_side*mosaic_aspect+pad_px*2), int(min_side+pad_px*2))
    done = False
    inc_rate = 1.1
    while not done:
        try:
            img, bboxes = make_heterog_mosaic(init_size, imgs, pad_px=pad_px)
            done = True
        except ValueError:
            init_size = (int(init_size[0]*inc_rate), int(init_size[1]*inc_rate))
            #self.print(f"Increasing mosaic size to {init_size}.")
    
    return img, bboxes


def test_make_heterog_mosaic():
    import matplotlib.pyplot as plt
    import cv2
    n_tiles = 20
    tiles = []
    sides = []
    for i in range(n_tiles):
        side = np.random.randint(20, 100)
        tile = np.random.randint(0, 255, size=(side, side, 3), dtype=np.uint8)
        tile[:20, :20, ] = 0
        cv2.putText(tile, str(i), (1, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(tile)
    size = np.array((800, 600))
    done = False
    img, bboxes = make_heterog_mosaic_autosize(tiles, mosaic_aspect=1.7)
    plt.imshow(img)
    plt.show()


def make_cluster_image(tiles, distances, n_max=100, aspect=1.7, bg_color=None):
    """
    show central third, middle third, and outer-most third of tiles
    :param tiles: N x 768 array of intensities (floats in [0,1])
    :param distances: array of N distances of each tile to cluster center
    :returns img: combined image
    """
    # Sort tiles by distance
    sorted_indices = np.argsort(distances)

    # adjust n_Max so mosaic is full
    n_rows, n_cols = get_mosaic_shape(n_max, aspect)
    n_max = n_rows * n_cols

    tiles = tiles[sorted_indices[:n_max], :]
    if tiles.shape[0] < n_max:
        tiles = tiles
    else:
        best = tiles[:n_max//3, :]
        middle = tiles[tiles.shape[0]//3:tiles.shape[0]//3 + n_max//3, :]
        worst = tiles[-(n_max//3):, :]
        tiles = np.vstack((best, middle, worst))
    img = make_digit_mosaic(tiles.astype(np.uint8), mosaic_aspect=aspect, bkg=bg_color)
    return img


def make_assign_gallery(size, tiles, distances, assignments, n_max=200, pad_px=10, bgk_color=255):
    """
    Create a gallery of cluster images.
    :param size: (width, height) of output image
    :param tiles: N x 768 array of intensities (floats in [0,1])
    :param distances: array of N distances of each tile to cluster center
    :param assignments: array of N cluster assignments
    :param n_max: maximum number of tiles to show
    :param pad_px: padding between clusters
    :returns img: combined image, dict: cluster_id -> {'x': (x_min, x_max), 'y': (y_min, y_max)}
    """

    cluster_ids = np.unique(assignments)
    n_clusters = len(cluster_ids)
    images = [make_cluster_image(tiles[assignments == k], distances[assignments == k], n_max=n_max, aspect=1.0, bg_color=bgk_color)
              for k in cluster_ids]
    aspect = size[0]/size[1]

    img, cluster_bboxes = make_heterog_mosaic_autosize(images, pad_px=pad_px, mosaic_aspect=aspect, bkg_color=bgk_color)

    return img, cluster_bboxes


def test_make_cluster_image():
    import matplotlib.pyplot as plt
    n_tiles = 2500
    tiles = np.random.rand(n_tiles, 28*28)
    distances = np.random.rand(n_tiles)
    img1 = make_cluster_image(tiles, distances, n_max=100, aspect=1.7)
    img2 = make_cluster_image(tiles[:30], distances[:30], n_max=100, aspect=1.7)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    plt.show()


def test_make_assign_gallery():
    import matplotlib.pyplot as plt
    n_tiles = 2500
    tiles = np.random.rand(n_tiles, 28*28)
    distances = np.random.rand(n_tiles)
    assignments = np.random.randint(0, 10, size=(n_tiles,))
    img, bboxes = make_assign_gallery((800, 600), tiles, distances, assignments,
                                      n_max=200, pad_px=10)
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    # test_make_cluster_image()
    # test_make_heterog_mosaic()
    test_make_assign_gallery()
