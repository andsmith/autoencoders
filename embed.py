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
from embedding_drawing import draw_embedding
from load_ae import load_autoencoder
from img_util import make_img as mnist_img
from tests import load_mnist, load_fashion_mnist

from pca import PCA
# Run on each of these:
EMBEDDINGS = [ TSNEEmbedding, UMAPEmbedding]#PCAEmbedding, RandomProjectionEmbedding,


class LatentRepEmbedder(object):
    def __init__(self, embedders, weights_filename, map_size_wh=(4096, 4096)):
        """
        :param weights_filename: Path to the trained autoencoder weights file  (saved AutoencoderExperiment)
        :param map_size_wh: Size of the map to draw the embeddings on
        """
        self._dataset = "Digits"
        self._map_size_wh = map_size_wh
        self._weights_filename = weights_filename
        self._autoencoder = self._load_autoencoder()
        self._embedders = embedders
        self._codes, self._digits, self._images = self._encode_data()
        self._embedded_train_data, self._embed_files = self._calc_embeddings()
        self._draw_maps()

    def _load_autoencoder(self):
        return load_autoencoder(self._weights_filename)

    def _encode_data(self):
        logging.info("Encoding training set...")
        train_encoded = self._autoencoder.encode_samples(self._autoencoder.x_train_pca, raw=False)
        train_digits = self._autoencoder.y_train
        train_images = self._autoencoder.x_train
        return train_encoded, train_digits, train_images

    def _calc_embeddings(self, save=True):
        # Run the specified embedding on the encoder
        train_2d = []
        save_files = []
        for embedder in self._embedders:
            logging.info("Calculating %s embedding for %i x %i matrix", embedder.get_name(),
                         self._codes.shape[0], self._codes.shape[1])
            train_2d.append(embedder.fit_embed(self._codes))
            if save:
                embedder_save_file = embedder.save(file_root=self._weights_filename)
                logging.info("\tsaved result to file:  %s" % (embedder_save_file, ))
                save_files.append(embedder_save_file)

        return train_2d, save_files

    def _draw_maps(self, sample_size=20000, bkg_color=None):
        sample = np.random.choice(self._codes.shape[0], sample_size, replace=False)
        images_gray = [(self._images[i, :]).reshape(28, 28) for i in sample]
        labels = self._digits[sample]
        colors = MPL_CYCLE_COLORS

        images, bboxes = [], []
        for i, (image_locs, save_file) in enumerate(zip(self._embedded_train_data, self._embed_files)):
            map_image_name = "%s-%s.map.png" % (self._dataset, save_file)
            image_locs = image_locs[sample]

            blank_map = np.zeros((self._map_size_wh[1], self._map_size_wh[0], 3), dtype=np.uint8)
            blank_map[:] = COLORS['OFF_WHITE_RGB'] if bkg_color is None else bkg_color

            map_img, mapped_bbox = draw_embedding(blank_map, image_locs, images_gray, labels=labels, colors=colors)
            images.append(map_img)
            bboxes.append(mapped_bbox)
            cv2.imwrite(map_image_name, map_img[:, :, ::-1])
            logging.info("Saved map image to %s", map_image_name)
        return images, bboxes


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

    def __init__(self, embedders, data, preproc_method, n_dims=None, map_size_wh=(4096, 4096), dataset='Digits'):
        self._preproc = preproc_method
        self._dataset = dataset
        self._n_dims = n_dims
        self._map_size_wh = map_size_wh
        self._embedders = embedders
        print("\nEmbedders:")
        for embedder in self._embedders:
            print("\t", self._get_name(embedder))

        self._load_data(data)
        self._preprocess()
        self._embedded_train_data, _ = self._calc_embeddings(save=False)

        # Map filenames based on these:
        self._embed_files = [self._get_name(embedder) for embedder in self._embedders]
        #self._draw_stats()
        self._draw_maps()

    def _draw_stats(self):
        code_dist = CodeSetDist(self._embedded_train_data)
        img = code_dist.plot()

    def _get_name(self, embedder):
        pre = ("%s(d=%i)" % (self._preproc, self._n_dims)) if self._preproc is not None else "raw(d=768)"
        return "raw_%s_embedding=%s_PreProc=%s" % (self._dataset.lower(),embedder.get_name(), pre)

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
    description = "Load a trained autoencoder and create embeddings for the MNIST dataset."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("weights_file", help="Path to the trained network weights file")
    args = parser.parse_args()
    embedders = [embed_class() for embed_class in EMBEDDINGS]
    LatentRepEmbedder(embedders=embedders, weights_filename=args.weights_file)


def embed_raw(dataset):
    """
    Find 2d embeddings of the raw data (with minimal pre-processing).
    Go through all embedding classes in EMBEDDINGS, draw a big map.
    """

    embedders = [embed_class() for embed_class in EMBEDDINGS]

    data = load_mnist() if dataset == 'Digits' else load_fashion_mnist()
    for x in data:
        print("Train", x[0].shape, "Test", x[1].shape)

    # import ipdb; ipdb.set_trace()
    ImageEmbedder(embedders=embedders, data=data, preproc_method=None, dataset=dataset)

    # ImageEmbedder(embedders=embedders, data=data, preproc_method='PCA', n_dims=2, dataset=dataset)
    # ImageEmbedder(embedders=embedders, data=data, preproc_method='PCA-UW', n_dims=2, dataset=dataset)
    # ImageEmbedder(embedders=embedders, data=data, preproc_method='random', n_dims=2, dataset=dataset)

    # ImageEmbedder(embedders=embedders, data=data, preproc_method='PCA', n_dims=10, dataset=dataset)
    # ImageEmbedder(embedders=embedders, data=data, preproc_method='PCA-UW', n_dims=10, dataset=dataset)
    # ImageEmbedder(embedders=embedders, data=data, preproc_method='random', n_dims=10, dataset=dataset)

    # ImageEmbedder(embedders=embedders, data=data, preproc_method='PCA', n_dims=32, dataset=dataset)
    # ImageEmbedder(embedders=embedders, data=data, preproc_method='PCA-UW', n_dims=32, dataset=dataset)
    # ImageEmbedder(embedders=embedders, data=data, preproc_method='random', n_dims=32, dataset=dataset)

    ImageEmbedder(embedders=embedders, data=data, preproc_method='PCA', n_dims=64, dataset=dataset)
    ImageEmbedder(embedders=embedders, data=data, preproc_method='PCA-UW', n_dims=64, dataset=dataset)
    ImageEmbedder(embedders=embedders, data=data, preproc_method='random', n_dims=64, dataset=dataset)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # embed_latent()
    embed_raw('Fashion')

