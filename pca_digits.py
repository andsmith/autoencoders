from pca import PCA
from tests import load_mnist
import numpy as np
import matplotlib.pyplot as plt
from img_util import make_img
from colors import COLORS
import logging
import cv2


def _generate(images, labels, samples, grid_shape, d=None, var_frac=None, orient='vertical'):
    """
    :param images: The input images.
    :param labels: The labels corresponding to the images.
    :param samples: The indices of the samples to use.
    :param grid_shape: The shape of the grid for displaying images.
    :param d: The number of dimensions to reduce to (if using PCA).
    :param var_frac: The fraction of variance to retain (if using PCA).
    :param orient: Are images ordered across rows first (horizontal) or columns first (vertical)?
    """
    sample_labels = labels[samples]
    sample_order = np.argsort(sample_labels)
    if d is not None:
        pca = PCA(dims=d, whiten=False)
    elif var_frac is not None:
        pca = PCA(dims=var_frac, whiten=False)
    else:
        pca = None

    if pca is None:
        decoded_sample_images = [images[s].reshape(28, 28) for s in samples]
        title_parts = {'original': None}
    else:
        encoded_images = pca.fit_transform(images, use_cache=False)
        decoded_images = pca.decode(encoded_images)
        train_mse = np.mean((images - decoded_images) ** 2)
        decoded_sample_images = [decoded_images[s].reshape(28, 28) for s in samples]
        decoded_sample_images = [decoded_sample_images[i] for i in sample_order]

        var_exp, pca_dim = pca.variance_explained, pca.pca_dims
        title_parts = {"p_comps": "%i" % pca_dim, "var_pct": "%.2f %%" % (var_exp*100), "mse": "%.5f" % train_mse}

    out_img = np.zeros((grid_shape[0] * 28, grid_shape[1] * 28))
    for i, img in enumerate(decoded_sample_images):
        if orient=='horizontal':
            col = i % grid_shape[1]
            row = i // grid_shape[1]
        else:
            col = i // grid_shape[0]
            row = i % grid_shape[0]
        out_img[row*28:(row+1)*28, col*28:(col+1)*28] = np.clip(img, 0, 1)

    return out_img, title_parts


def show_low_dim_results():
    # Show "reconstructions" evenly interpolated across 1 and 2-d PCA representations.
    pass


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
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(12, 12))

    for col in range(n_cols):
        for row in range(n_rows):
            idx = col * n_rows + row
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


def draw_pca_maps():
    # Show originals where values==None.

    dim_grid = [None, 2, 4,
                8, 16, 32,
                64, 128, 256]

    var_grid = [None, 0.1, 0.25,
                0.33, 0.5, 0.75,
                0.90, 0.95, 0.99]

    _, (images, labels) = load_mnist()

    sample_grid_shape = [15, 10]
    n_samples = np.prod(sample_grid_shape)
    sample = np.array([np.random.choice(np.where(labels == i)[0], n_samples // 10, replace=False)
                      for i in range(10)]).flatten()
    
    orient = 'vertical' if sample_grid_shape[0] > sample_grid_shape[1] else 'horizontal'

    img_data_by_dim = [_generate(images, labels, sample, sample_grid_shape, d=d, var_frac=None, orient=orient) for d in dim_grid]
    img_data_by_var = [_generate(images, labels, sample, sample_grid_shape, d=None, var_frac=v, orient=orient) for v in var_grid]

    #Show by # of PCA dimensions:
    _plot_img_data(img_data_by_dim, "PCA Maps by Number of Components")
    # Show by fraction of explained variance:
    _plot_img_data(img_data_by_var, "PCA Maps by Fraction of Explained Variance")

    plt.show()

    img_by_var = _draw_img_data(img_data_by_var, "PCA Maps by Fraction of Explained Variance")
    img_by_dim = _draw_img_data(img_data_by_dim, "PCA Maps by Number of Components")

    cv2.imwrite("PCA-Digits_by-n_comps.png", img_by_dim)
    cv2.imwrite("PCA-Digits_by-var_exp.png", img_by_var)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    draw_pca_maps()
