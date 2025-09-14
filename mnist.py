import json
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
from tensorflow.keras.datasets import mnist, fashion_mnist
from load_typographyMNIST import load_numeric, load_alphanumeric, GOOD_CHAR_SET

from img_util import make_digit_mosaic, make_img
from color_blit import draw_color_tiles_cython


class MNISTData(object):
    _loader = mnist

    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self._loader.load_data()
        self.x_train = self.x_train.astype(np.float32) / 255.0
        self.x_test = self.x_test.astype(np.float32) / 255.0

        logging.info("MNIST Data loaded successfully:")
        logging.info("\tTraining samples: %d, %s", len(self.x_train), self.x_train.shape)
        logging.info("\tTesting samples: %d, %s", len(self.x_test), self.x_test.shape)


class FashionMNISTData(MNISTData):
    _loader = fashion_mnist


class NumericMNISTData(MNISTData):
    def __init__(self):
        (self.x_train, self.y_train, self.names_train), (
            self.x_test, self.y_test, self.names_test) = load_numeric(w_names=True)


class AlphaNumericMNISTData(MNISTData):
    def __init__(self, use_good_subset=True, test_train_split=0.15, font_file=None):
        """
        Load the Alphanumeric MNIST data set
        :param use_good_subset:  If True, only load a subset of the characters with good quality renderings
        :param test_train_split:  Fraction of data to use for testing (0.0-1.0)
        :param font_file: If not None, load data from the specified json font set file (output of cluster_font.py)


        """
        subset = GOOD_CHAR_SET if use_good_subset else None
        (self.x_train, self.labels_train, self.font_names_train), (
            self.x_test, self.labels_test, self.font_names_test) = load_alphanumeric(numeric_labels=False, w_names=True,
                                                                                     subset=subset, test_train_split=test_train_split,
                                                                                     font_file=font_file)
        label_classes = sorted(list(set(self.labels_train) | set(self.labels_test)))
        self.n_chars = len(label_classes)
        self.n_fonts = len(set(self.font_names_train) | set(self.font_names_test))
        self.class_name_to_index = {name: i for i, name in enumerate(label_classes)}
        self.index_to_class_name = {i: name for i, name in enumerate(label_classes)}
        self.y_train = np.array([self.class_name_to_index[name] for name in self.labels_train])
        self.y_test = np.array([self.class_name_to_index[name] for name in self.labels_test])
        logging.info("AlphaNumeric loaded %i samples from %i classes", len(self.x_train), len(label_classes))
        if font_file is not None:
            logging.info("\tFont file: %s had %i fonts", font_file, self.n_fonts)


datasets = {'digits': MNISTData,
            'fashion': FashionMNISTData,
            'numeric': NumericMNISTData,
            'alphanumeric': AlphaNumericMNISTData
            }


def _test_data(data_obj):
    print("Training data shape:", data_obj.x_train.shape)
    print("Testing data shape:", data_obj.x_test.shape)

    # Display a sample image
    grid_shape = (30, 50)
    n_samples = grid_shape[0] * grid_shape[1]
    sample_image = make_digit_mosaic([(x.reshape(28, 28) * 255).astype(np.uint8) for x in data_obj.x_train[:n_samples]])
    plt.figure(figsize=(10, 10))
    plt.imshow(sample_image, cmap='gray')
    plt.title("Dataset:  %s" % data_obj.__class__.__name__)


def test_mnist_data():
    # _test_data(MNISTData())
    # _test_data(FashionMNISTData())
    # _test_data(NumericMNISTData())
    _test_data(AlphaNumericMNISTData())
    plt.show()


def show_all_fonts(output_dir="T-MNIST"):
    """
    Make a sequences of images showing all typefaces
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    from colors import COLORS
    img_no = 0
    file_stem = f"alphanumeric_typefaces_page_%i.png"
    data = AlphaNumericMNISTData(good_subset=False)
    label_inds = np.concatenate((data.y_train, data.y_test), axis=0)
    data_order = np.argsort(label_inds)

    images = np.concatenate((data.x_train, data.x_test), axis=0)[data_order]
    labels = np.concatenate((data.labels_train, data.labels_test), axis=0)[data_order]
    fonts = np.concatenate((data.font_names_train, data.font_names_test), axis=0)[data_order]
    font_names = sorted(list(set(fonts)))
    label_names = sorted(set(labels.tolist()))
    index_col_w = 50

    print("Number of unique fonts:", len(font_names))
    print("Number of unique labels:", len(label_names))
    print("Total images:", len(images))
    n_cols, n_rows = len(label_names), 60
    img_size_wh = (n_cols*28+index_col_w, n_rows*28)

    blank = np.zeros((img_size_wh[1], img_size_wh[0], 3), dtype=np.uint8)
    blank[:] = COLORS['OFF_WHITE_RGB']

    color = COLORS['DARK_NAVY_RGB']
    # Header on all images
    img = blank.copy()

    def get_draw_row(y_pos, images, labels, font_num):
        """
        Write out the font, get list of tiles/positions to blit.
        """
        tiles, locs = [], []
        for i, label in enumerate(label_names):
            x_pos = i*28 + index_col_w
            s = np.where(labels == label)[0]
            if len(s) == 0:
                continue
            tiles.append(images[s[0]].reshape(28, 28))
            locs.append((x_pos, y_pos))
        new_y = y_pos + 28
        # on the left, write the font number
        cv2.putText(img, str(font_num), (5, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return new_y, tiles, locs

    tiles, locs = [], []
    y = 0
    font_table = {}
    for font_ind, font_name in enumerate(font_names):
        font_table[font_ind] = font_name
        samples = np.where(fonts == font_name)[0]
        image_samples = images[samples]
        label_samples = labels[samples]

        y, ntiles, nlocs = get_draw_row(y, image_samples, label_samples, font_ind)
        tiles += ntiles
        locs += nlocs
        if y + 28 >= img_size_wh[1]-1 or font_ind == len(font_names) - 1:
            # Save the current image and reset
            colors = np.array(color, dtype=np.uint8).reshape(1, 3)
            color_inds = np.zeros(len(tiles), dtype=np.int32)
            tiles = np.array(tiles, dtype=np.float32)
            locs = np.array(locs).astype(np.int32)
            draw_color_tiles_cython(img, locs, tiles, color_inds, colors)
            filename = os.path.join(output_dir, file_stem % img_no)
            cv2.imwrite(filename, img[:, :, ::-1])
            print("Wrote %i samples to image:  %s" % (len(tiles), filename))

            img_no += 1
            img = blank.copy()
            y = 0
            tiles, locs = [], []

    with open(os.path.join(output_dir, "font_table.txt"), 'w') as f:
        for font_ind in sorted(font_table.keys()):
            font_name = font_table[font_ind]
            f.write(f"{font_ind}: {font_name}\n")

    with open(os.path.join(output_dir, "font_table.json"), 'w') as f:
        json.dump(font_table, f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_mnist_data()
    show_all_fonts()
