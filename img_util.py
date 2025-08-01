
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


def make_digit_mosaic(imgs, mosaic_aspect=1.7):
    n_cols = int(np.ceil(np.sqrt(len(imgs)*mosaic_aspect)))
    n_rows = int(np.ceil(len(imgs) / n_cols))
    img_side = 28
    img_flat_size = img_side * img_side
    img = np.zeros((n_rows*img_side, n_cols*img_side, 3), dtype=np.uint8)
    for i, im in enumerate(imgs):
        x = i % n_cols
        y = i // n_cols
        if im.shape == (img_flat_size,):
            im = im.reshape((img_side, img_side))
            im = np.stack((im, im, im), axis=-1)  # Convert to RGB
        elif im.shape == (img_flat_size, 3):
            im = im.reshape((img_side, img_side, 3))
        # try:
        img[y*img_side:(y+1)*img_side, x*img_side:(x+1)*img_side, :] = im
        # except Exception:
        #    print(f"Error assembling image {i}: {im.shape}")
        #    continue
    img = np.clip(img, 0, 255)
    return img
