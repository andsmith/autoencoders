from pca import PCA
from tests import load_mnist, load_fashion_mnist
from load_typographyMNIST import load_alphanumeric, load_numeric, GOOD_CHAR_SET
import numpy as np
import matplotlib.pyplot as plt
from img_util import make_img, make_digit_mosaic
from colors import COLORS
import logging
import cv2
import argparse
import os
from multiprocessing import Pool, cpu_count


_OUTPUT_DIR = "PCA"


def _generate(images, samples, grid_shape, d=None, var_frac=None, orient='vertical'):
    """
    :param images: The input images.
    :param labels: The labels corresponding to the images.
    :param samples: The indices of the samples to use.
    :param grid_shape: The shape of the grid for displaying images.
    :param d: The number of dimensions to reduce to (if using PCA).
    :param var_frac: The fraction of variance to retain (if using PCA).
    :param orient: Are images ordered across rows first (horizontal) or columns first (vertical)?
    """
    if d is not None:
        pca = PCA(dims=d)
    elif var_frac is not None:
        pca = PCA(dims=var_frac)
    else:
        pca = None

    if pca is None:
        decoded_sample_images = [images[s].reshape(28, 28) for s in samples]
        title_parts = {'original': None}
    else:
        encoded_images = pca.fit_transform(images)
        decoded_images = pca.decode(encoded_images)
        train_mse = np.mean((images - decoded_images) ** 2)
        decoded_sample_images = [decoded_images[s].reshape(28, 28) for s in samples]

        var_exp, pca_dim = pca.variance_explained, pca.pca_dims
        title_parts = {"p_comps": "%i" % pca_dim, "var_pct": "%.2f %%" % (var_exp*100), "mse": "%.5f" % train_mse}

    out_img = np.zeros((grid_shape[0] * 28, grid_shape[1] * 28))
    for i, img in enumerate(decoded_sample_images):
        if orient == 'horizontal':
            col = i % grid_shape[1]
            row = i // grid_shape[1]
        else:
            col = i // grid_shape[0]
            row = i % grid_shape[0]
        out_img[row*28:(row+1)*28, col*28:(col+1)*28] = np.clip(img, 0, 1)

    return out_img, title_parts


def _draw_img_data(img_data_list, title):
    n_cols = 3
    n_rows = 3
    img_w, img_h = img_data_list[0][0].shape[1], img_data_list[0][0].shape[0]
    if n_cols * n_rows < len(img_data_list):
        n_cols = int(np.ceil(len(img_data_list) / n_rows))
    footer_space = 112
    side_space = 30
    img_h_total = img_h + footer_space
    footer = np.zeros((footer_space, img_w, 3), dtype=np.uint8)
    img_w_total = img_w + side_space
    side_spacer = np.zeros((img_h_total, side_space, 3), dtype=np.uint8)
    mosaic = np.zeros((img_h_total * n_rows, img_w_total * n_cols, 3), dtype=np.uint8)

    for col in range(n_cols):
        for row in range(n_rows):
            idx = row * n_cols + col
            if idx < len(img_data_list):
                img, title_info = img_data_list[idx]
                color_img = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                m_x = col*img_w_total
                m_y = row*img_h_total
                mosaic[m_y:m_y+img_h_total, m_x:m_x +
                       img_w_total] = np.concatenate([np.concatenate([color_img, footer], axis=0), side_spacer], axis=1)
                title_line_keys = ['original'] if 'original' in title_info else ['p_comps', 'var_pct', 'mse']
                y_top = (row+1)*img_h_total - footer_space - 10
                for i, title_key in enumerate(title_line_keys):
                    string = f"{title_key}: {title_info[title_key]}" if title_info[title_key] is not None else title_key
                    font_scale = 0.8
                    row_spacing = 34
                    txt_pos = (col*img_w_total, y_top + (i+1)*row_spacing)
                    cv2.putText(mosaic, string, txt_pos, cv2.FONT_HERSHEY_DUPLEX,
                                font_scale, COLORS['NEON_GREEN'], 1, lineType=cv2.LINE_AA)
    # add side-spacer to left and top
    mosaic = np.concatenate([np.zeros((mosaic.shape[0], side_space, 3), dtype=np.uint8), mosaic], axis=1)
    mosaic = np.concatenate([np.zeros((side_space, mosaic.shape[1], 3), dtype=np.uint8), mosaic], axis=0)

    return mosaic


def _plot_img_data(img_data_list, title):
    n_cols = 3
    n_rows = 3
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(12, 12), sharex=True, sharey=True)

    for col in range(n_cols):
        for row in range(n_rows):
            idx = row * n_cols + col
            if idx < len(img_data_list):
                img, title_info = img_data_list[idx]
                if 'original' in title_info:
                    title = "Original 28 x 28 (d=784) images."
                else:
                    title = "N Components: %s,                         \n VarianceExp: %s, MSE: %s" % (
                        title_info['p_comps'], title_info['var_pct'], title_info['mse'])

                ax[row, col].imshow(img, cmap='gray')
                ax[row, col].set_title(title, fontsize=12)
                ax[row, col].axis('off')

    plt.suptitle(title, fontsize=15)
    plt.tight_layout()


def _generate_hlpr(kwargs):
    return _generate(**kwargs)


def draw_pca_maps(images, labels, dataset):
    # Show originals where values==None.

    dim_grid = [None, 2,  4,
                8,   16,  32,
                64, 128, 256]

    var_grid = [None, 0.1,  0.25,
                0.33, 0.5,  0.75,
                0.90, 0.95, 0.99]
    # dim_grid,var_grid = [32],[.9]  # just draw one for testing
   
    sample_grid_shape = [10, 15]  # keep 10 rows for digits/numeric
    n_samples = np.prod(sample_grid_shape)
    n_labels = len(np.unique(labels))

    # Get an equal number of samples for each label:

    sample = [[] for _ in range(n_labels)]
    for i in range(n_labels):
        sample[i] = np.random.choice(np.where(labels == i)[0], n_samples // n_labels, replace=False)
    sample = np.array(sample).flatten()
    unused = np.setdiff1d(np.arange(len(labels)), sample)
    if len(sample) < n_samples:
        # fill the rest with random samples
        remaining = n_samples - len(sample)
        random_samples = np.random.choice(unused, remaining, replace=False)
        sample = np.concatenate([sample, random_samples])

    orient = 'vertical' if sample_grid_shape[0] > sample_grid_shape[1] else 'horizontal'

    img_data_by_dim_work = [dict(images=images, samples=sample, grid_shape=sample_grid_shape, d=d,
                                 var_frac=None, orient=orient) for d in dim_grid]
    img_data_by_var_work = [dict(images=images, samples=sample, grid_shape=sample_grid_shape, d=None,
                                 var_frac=v, orient=orient) for v in var_grid]
    img_data_by_dim = [_generate_hlpr(work) for work in img_data_by_dim_work]
    img_data_by_var = [_generate_hlpr(work) for work in img_data_by_var_work]
    # combined_work = img_data_by_dim_work + img_data_by_var_work
    # with Pool(processes=3) as pool:
    #     results = pool.map(_generate_hlpr, combined_work)
    # img_data_by_dim, img_data_by_var = results[:len(img_data_by_dim_work)], results[len(img_data_by_dim_work):]
    # Show by # of PCA dimensions:
    _plot_img_data(img_data_by_dim, "PCA Maps by Number of Components")
    # Show by fraction of explained variance:
    _plot_img_data(img_data_by_var, "PCA Maps by Fraction of Explained Variance")

    plt.show()

    img_by_var = _draw_img_data(img_data_by_var, "PCA Maps by Fraction of Explained Variance")
    img_by_dim = _draw_img_data(img_data_by_dim, "PCA Maps by Number of Components")

    cv2.imwrite(os.path.join(_OUTPUT_DIR, "PCA-Reconstruction_%s_by-n_comps.png" % (dataset,)), img_by_dim)
    cv2.imwrite(os.path.join(_OUTPUT_DIR, "PCA-Reconstruction_%s_by-var_exp.png" % (dataset,)), img_by_var)


def show_low_dim_results():
    # Show "reconstructions" evenly interpolated across 1 and 2-d PCA representations.
    pass


def _make_comp_img(components, grid_shape, magnification, max_z=3.75):
    """
    First arange all components in reverse order in the specified grid shape.
    Then normalize/scale and magnify.
    """
    # Arrange components in the specified grid shape
    float_img = np.zeros((grid_shape[0] * 28, grid_shape[1] * 28), dtype=np.float32)
    for i in range(components.shape[1]):
        row = i // grid_shape[1]
        col = i % grid_shape[1]
        x_left = col * 28
        y_top = row * 28
        comp_img = components[:, i].reshape(28, 28)

        float_img[y_top:y_top + 28, x_left:x_left + 28] = comp_img
    z_image = (float_img-np.mean(float_img)) / np.std(float_img)
    float_img = np.clip(z_image/max_z, -1, 1)/2 + 0.5
    float_img = (float_img*255).astype(np.uint8)
    new_size = (float_img.shape[1] * magnification, float_img.shape[0] * magnification)
    return cv2.resize(float_img, new_size, interpolation=cv2.INTER_CUBIC)


def show_components(images, labels, dataset):
    # Show a representation of the components.
    grid_shape = (26, 26)  # rows, cols of examples to show
    mag_factor = 5
    n_comps = grid_shape[0] * grid_shape[1]

    pca = PCA(dims=n_comps)

    pca.fit_transform(images)
    comps = pca.components
    image = _make_comp_img(comps, grid_shape, mag_factor)
    cv2.imwrite(os.path.join(_OUTPUT_DIR, "PCA-Components_%s.png" % dataset), image)
    cv2.imshow("PCA Components", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def make_figs(dataset, binary=False):

    if dataset == 'digits':
        (train_images, train_labels), (test_images, test_labels) = load_mnist()
    elif dataset == 'fashion':
        (train_images, train_labels), (test_images, test_labels) = load_fashion_mnist()
    elif dataset == 'numeric':
        (train_images, train_labels), (test_images, test_labels) = load_numeric()
    elif dataset == 'alphanumeric':
        (train_images, train_labels), (test_images, test_labels) = load_alphanumeric(
            subset=GOOD_CHAR_SET, numeric_labels=True)
    else:
        raise ValueError("Unknown dataset: %s" % dataset)

    images = np.concatenate((train_images, test_images), axis=0)
    labels = np.concatenate((train_labels, test_labels), axis=0)

    if binary:
        images = (images > 0.5).astype(np.float32)
        dataset = "%s-BIN" % dataset

    draw_pca_maps(images, labels, dataset)
    show_components(images, labels, dataset)


def get_args():
    parser = argparse.ArgumentParser(description="PCA Reconstruction & components visualization")
    parser.add_argument("--dataset", type=str, default="digits",
                        choices=["digits", "fashion", 'numeric', 'alphanumeric'], help="Dataset to use")
    parser.add_argument("--binary", action='store_true', help="Use binary images as input.")
    parsed = parser.parse_args()
    return {k: v for k, v in vars(parsed).items() if v is not None}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    make_figs(**get_args())
