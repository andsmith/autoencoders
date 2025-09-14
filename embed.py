"""
Load autoencoder weights, encode the MNIST dataset, create a 2d embedding of the latent representation.
i.e. on autoencoder.encode(x) for all images x.

Write the embedding back to the same folder, e.g., running:

      > python embed.py VAE-results/VAE-TORCH_trained_network.weights.h5

will create the following files:

        VAE-TORCH_trained_network.PCA(64)embedding.png
        VAE-TORCH_trained_network.TSNE(tsne-params)embedding.png
        VAE-TORCH_trained_network.UMAP(umap-params)embedding.png

and put them in the "VAE-results/" subdirectory.

"""
from latent_var_plots import LatentCodeSet
from fileinput import filename
import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
from embeddings import PCAEmbedding, TSNEEmbedding, UMAPEmbedding, RandomProjectionEmbedding
from util import PointSet2d
from colors import COLORS, MPL_CYCLE_COLORS
import argparse
from color_blit import draw_color_tiles_cython as draw_tiles
# from load_ae import load_autoencoder
from img_util import make_img as mnist_img
from load_ae import load_autoencoder, get_ae_dir, LOADERS
import os
import pickle,json
import re

from pca import PCA
# Run on each of these:
EMBEDDINGS = [PCAEmbedding, TSNEEmbedding, UMAPEmbedding]  # UMAPEmbedding, RandomProjectionEmbedding,

class LatentRepEmbedder(object):
    _WORKING_DIR = "embeddings"

    def __init__(self, embedder_class, weights_filename):
        """
        :param weights_filename: Path to the trained autoencoder weights file  (saved AutoencoderExperiment)
        :param map_size_wh: Size of the map to draw the embeddings on
        """
        self._weights_filename = weights_filename
        self._type = embedder_class
        self._autoencoder = load_autoencoder(self._weights_filename)
        self._images = self._autoencoder.x_train
        self._digits = self._autoencoder.y_train
        logging.info("Initializing Embedding of type: %s", self._type.__name__)
        if not self.load():

            logging.info("Didn't find encoding, creating new one...")
            self._codes = self._encode_data()
            self._embedder, self._embedded_train_data = self._calc_embedding()
            self.save()
        

        self._draw_maps()

    def get_filename(self):
        wts_base = os.path.basename(self._weights_filename)
        cls_name = self._type().get_name()
        return os.path.join(LatentRepEmbedder._WORKING_DIR, f"{wts_base}.embed-{cls_name}" + ".pkl")

    @staticmethod
    def from_filename(embed_filename):
        working_dir = get_ae_dir(embed_filename)
        embed_filename = os.path.basename(embed_filename)
        wts_file = re.search(r'^(.*?)[\.\_]embed-', embed_filename).group(1)
        cls_name = re.search(r'embed-(.+).pkl', embed_filename).group(1).upper().replace('-','')
        cls = next((c for c in EMBEDDINGS if c().get_name().upper().replace('-','') == cls_name), None)  # TODO: fix...
        if cls is None:
            raise ValueError(f"Unknown embedding class: {cls_name}")
        wts_file = os.path.join(working_dir, wts_file)
        return LatentRepEmbedder(cls, wts_file)

    def save(self):
        filename = self.get_filename()
        if not os.path.exists(LatentRepEmbedder._WORKING_DIR):
            os.makedirs(LatentRepEmbedder._WORKING_DIR)
        save_data = {'embedder': self._embedder,
                     'embedded_train_data': self._embedded_train_data,
                     'codes': self._codes}
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
        logging.info("Saved embedding to %s", filename)

    def load(self, filename=None):
        filename = filename if filename is not None else self.get_filename()
        if not os.path.exists(filename):
            return False
        logging.info("Loading embedding from %s ...", filename)
        with open(filename, 'rb') as f:
            load_data = pickle.load(f)
        self._embedder = load_data['embedder']
        self._embedded_train_data = load_data['embedded_train_data']
        self._codes = load_data['codes']
        logging.info("Loaded embedding from %s", filename)
        return True

    def _encode_data(self):
        logging.info("Encoding training set...")
        train_encoded = self._autoencoder.encode_samples(self._autoencoder.x_train_pca, raw=False)
        return train_encoded

    def _calc_embedding(self):
        # Run the specified embedding on the encoder
        logging.info("Calculating %s embedding of %i codes, dim %i...", self._type.__name__, self._codes.shape[0], self._codes.shape[1])
        embedder = self._type()
        train_2d = embedder.fit_embed(self._codes)
        logging.info("Finished calculating embedding, data in range:  x:(%.4f, %.4f), y:(%.4f, %.4f)", train_2d[:, 0].min(), train_2d[:, 0].max(), train_2d[:, 1].min(), train_2d[:, 1].max())
        return embedder, train_2d
    

    def _draw_maps(self, sample_size=0, bkg_color=None):

        sample = np.random.choice(self._codes.shape[0], sample_size,
                                  replace=False) if sample_size > 0 else np.arange(self._codes.shape[0])
        images_gray = [(self._images[i, :]).reshape(28, 28) for i in sample]
        labels = self._digits[sample]
        colors = np.array((MPL_CYCLE_COLORS),dtype=np.uint8)
        map_size_wh = (4096, 4096)
        images, bboxes = [], []
        image_locs = self._embedded_train_data[sample]

        map_image_name = "%s.map.png" % ( self.get_filename(),)
        image_locs = image_locs[sample]

        blank_map = np.zeros((map_size_wh[1], map_size_wh[0], 3), dtype=np.uint8)
        blank_map[:] = COLORS['OFF_WHITE_RGB'] if bkg_color is None else bkg_color

        map_img, mapped_bbox = _draw_embedding(
            blank_map, image_locs, images_gray, labels=labels, color_set=colors)
        images.append(map_img)
        bboxes.append(mapped_bbox)
        cv2.imwrite(map_image_name, map_img[:, :, ::-1])
        logging.info("Saved map image to %s", map_image_name)
        return images, bboxes


def _draw_embedding(image, locs_xy, tiles, labels, color_set):
    """
    Draw the embedding, filter by label subset if it is given.
    :param image: H x W x 3 numpy array.
    :param locs_xy: (N x 2) numpy array of x,y coordinates (pixel locations to draw the tiles)
    :param tiles: (N x Th x Tw x 3) numpy array of Tx x Tw tile images
    :param labels: (N,) numpy array of labels for each tile, int or string for alpha-numeric datasets.
    :param color_set: Set of colors to use for drawing the tiles (0..(num_labels-1))
    :param subset: Optional subset of labels to draw, same type as labels.
    """

    
    if labels.dtype not in [np.int32, np.uint8, int]:
        label_inds = np.zeros(labels.shape, dtype=np.int32)
        label_set = np.sort(np.unique(labels))
        for l_ind, l_str in enumerate(label_set):
            label_inds[labels == l_str] = l_ind
    else:
        label_inds = labels.astype(np.int32)    

    image_size_wh = np.array((image.shape[1], image.shape[0]), dtype=np.float32)
    pixel_locs = (locs_xy * (image_size_wh-28*2) + 28).astype(np.int32)
    tiles = np.array(tiles, dtype=np.float32)

    draw_tiles(image, pixel_locs, tiles, label_inds, np.array(color_set, dtype=np.uint8))
    return image, None


class CodeSetDist(LatentCodeSet):
    """
    Show a narrow boxplot for each code unit.
    Put on the same scale, etc.
    """

    def __init__(self, codes):
        test_codes, test_labels, digit_subset, colors = self._init_data(codes)
        super().__init__(test_codes, test_labels, digit_subset, colors)

    def _init_data(self, codes):
        codes_0 = codes
        labels_0 = np.zeros(codes.shape[0], dtype=np.int32)
        fake_labels_1_9 = np.array([[d]*5 for d in range(1, 10)]).flatten()
        fake_codes_1_9 = np.random.randn(fake_labels_1_9.shape[0], codes.shape[1])

        test_codes = np.concatenate((codes_0, fake_codes_1_9), axis=0)
        test_labels = np.concatenate((labels_0, fake_labels_1_9), axis=0)
        digit_subset = (0,)  # don't use the fake codes/labels
        colors = MPL_CYCLE_COLORS[:1]
        return test_codes, test_labels, digit_subset, colors


class ImageEmbedder(LatentRepEmbedder):
    """
    For each embedding method, preprocess the data, plot input statistics (box-plot of values for each dimension),
    create the embedding, and draw the map.
    """
    # Do not run PCAEmbedding on whitened data, since all components have equal variance.
    _PREPROC_METHODS = ['PCA', 'PCA-UW', 'random', None]

    def __init__(self, embedders, data, preproc_method, n_dims=None, map_size_wh=(4096, 4096), dataset='digits', disp_subset=None):
        self._preproc = preproc_method
        self._dataset = dataset
        self._n_dims = n_dims
        self._map_size_wh = map_size_wh
        self._disp_subset = disp_subset
        self._embedders = embedders
        print("\nEmbedders:")
        for embedder in self._embedders:
            print("\t", self._get_name(embedder))

        self._load_data(data)
        self._preprocess()
        self._embedded_train_data, _ = self._calc_embeddings(save=False)

        # Map filenames based on these:
        self._embed_files = [self._get_name(embedder) for embedder in self._embedders]
        # self._draw_stats()
        self._draw_maps(disp_subset=self._disp_subset)

    def _draw_stats(self):
        code_dist = CodeSetDist(self._embedded_train_data)
        img = code_dist.plot()

    def _get_name(self, embedder):
        pre = ("%s(d=%i)" % (self._preproc, self._n_dims)) if self._preproc is not None else "raw(d=768)"
        return "raw_%s_embedding=%s_PreProc=%s" % (self._dataset.lower(), embedder.get_name(), pre)

    def _load_data(self, data):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data
        self._images = self.x_train
        self._digits = self.y_train

    def _preprocess(self):
        if self._preproc is None:
            self._codes = self.x_train
            self._codes_test = self.x_test
        elif self._preproc.startswith('PCA'):
            whiten = not self._preproc.endswith('-UW')
            pca = PCA(dims=self._n_dims, whiten=whiten)
            self._codes = pca.fit_transform(self.x_train)
            self._codes_test = pca.encode(self.x_test)
        elif self._preproc == 'random':
            vecs = np.random.randn(self.x_train.shape[1], self._n_dims)
            vecs /= np.linalg.norm(vecs, axis=0)
            self._codes = self.x_train @ vecs
            self._codes_test = self.x_test @ vecs
        else:
            raise ValueError("Invalid method: must be one of %s, got %s" % (self._PREPROC_METHODS, self._preproc))


def embed_latent():
    """
    Find 2d embeddings of the latent representation of the dataset.
    Go through all embedding classes in EMBEDDINGS, draw a big map.
    """
    # TODO:  Add per-embedding args  --TSNE-perplexity --UMAP-n_neighbors
    #weights_filename = r'Dense-results\digits_Dense(digits-PCA(784,UW)_units=256-2048-256-8_dec-units=512_Drop(l=1,r=0.50)).weights.h5'

    description = """Compute 2d embeddings for latent representations:
    Syntax:  python embed.py <weights_filename>

    Will produce, for each embedder_class in EMBEDDINGS:
        * 2d embeddings of the latent representation, saved in embeddings/<weights_filename>.embed-<embedder_class>
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('weights_filename', type=str, help='Path to the trained autoencoder weights file  (saved AutoencoderExperiment)')
    args = parser.parse_args()


    for embed_cls in EMBEDDINGS:
        LatentRepEmbedder(embedder_class=embed_cls, weights_filename=args.weights_filename)


def embed_raw(dataset):
    """
    Find 2d embeddings of the raw data (with minimal pre-processing).
    Go through all embedding classes in EMBEDDINGS, draw a big map.
    """

    embedders = [embed_class() for embed_class in EMBEDDINGS]

    data = LOADERS[dataset.lower()]()
    for x in data:
        print("Train", x[0].shape, "Test", x[1].shape)


    # ImageEmbedder(embedders=embedders, data=data, preproc_method=None, dataset=dataset)

    # ImageEmbedder(embedders=embedders, data=data, preproc_method='PCA', n_dims=2, dataset=dataset)
    # ImageEmbedder(embedders=embedders, data=data, preproc_method='PCA-UW', n_dims=2, dataset=dataset)
    # ImageEmbedder(embedders=embedders, data=data, preproc_method='random', n_dims=2, dataset=dataset)

    # ImageEmbedder(embedders=embedders, data=data, preproc_method='PCA', n_dims=10, dataset=dataset)
    # ImageEmbedder(embedders=embedders, data=data, preproc_method='PCA-UW', n_dims=10, dataset=dataset)
    # ImageEmbedder(embedders=embedders, data=data, preproc_method='random', n_dims=10, dataset=dataset)

    # ImageEmbedder(embedders=embedders, data=data, preproc_method='PCA', n_dims=32, dataset=dataset)
    # ImageEmbedder(embedders=embedders, data=data, preproc_method='PCA-UW', n_dims=32, dataset=dataset)
    # ImageEmbedder(embedders=embedders, data=data, preproc_method='random', n_dims=32, dataset=dataset)
    img_char_subset = ['A', 'B', 'C', '1', '2', '3']
    ImageEmbedder(embedders=embedders, data=data, preproc_method='PCA',
                  n_dims=128, dataset=dataset, disp_subset=img_char_subset)
    # ImageEmbedder(embedders=embedders, data=data, preproc_method='PCA-UW', n_dims=64, dataset=dataset)
    # ImageEmbedder(embedders=embedders, data=data, preproc_method='random', n_dims=64, dataset=dataset)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    embed_latent()
    # embed_raw('alphanumeric')
