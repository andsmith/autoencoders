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
from tests import load_mnist
from pca import PCA
# Run on each of these:
EMBEDDINGS = [PCAEmbedding, RandomProjectionEmbedding, TSNEEmbedding, UMAPEmbedding]


class LatentRepEmbedder(object):
    def __init__(self, weights_filename, map_size_wh=(4096, 4096)):
        """
        :param weights_filename: Path to the trained autoencoder weights file  (saved AutoencoderExperiment)
        :param map_size_wh: Size of the map to draw the embeddings on
        """
        self._map_size_wh = map_size_wh
        self._weights_filename = weights_filename
        self._autoencoder = self._load_autoencoder()
        self._embedders = [EmbeddingClass() for EmbeddingClass in EMBEDDINGS]
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
            map_image_name = "%s.map.png" % save_file
            image_locs = image_locs[sample]

            blank_map = np.zeros((self._map_size_wh[1], self._map_size_wh[0], 3), dtype=np.uint8)
            blank_map[:] = COLORS['OFF_WHITE_RGB'] if bkg_color is None else bkg_color

            map_img, mapped_bbox = draw_embedding(blank_map, image_locs, images_gray, labels=labels, colors=colors)
            images.append(map_img)
            bboxes.append(mapped_bbox)
            cv2.imwrite(map_image_name, map_img[:, :, ::-1])
            logging.info("Saved map image to %s", map_image_name)
        return images, bboxes


class ImageEmbedder(LatentRepEmbedder):
    """
    Don't encode with a network, just find embeddings on the high-dim space, PCA optional.
    """
    _METHODS = ['PCA', 'PCA-UW', 'random', None]

    def __init__(self, method='PCA', n_dims=42, map_size_wh=(4096, 4096)):
        self._method = method
        self._n_dims = n_dims
        self._map_size_wh = map_size_wh
        self._embedders = [EmbeddingClass() for EmbeddingClass in EMBEDDINGS]
        print("Embedders:")
        for embedder in self._embedders:
            print("\t",self._get_name(embedder))

        self._load_data()
        self._preprocess()
        self._embedded_train_data, _ = self._calc_embeddings(save=False)
        # Map filenames based on these:
        self._embed_files = [self._get_name(embedder) for embedder in self._embedders]
        self._draw_maps()

    def _get_name(self, embedder):
        pre = ("%s(d=%i)" % (self._method, self._n_dims)) if self._method is not None else "raw(d=768)"
        return "raw_digits_Embedding=%s_PreProc=%s" % (embedder.get_name(), pre)
    
    def _load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_mnist()
        self._images =  self.x_train
        self._digits = self.y_train


    def _preprocess(self):
        if self._method is None:
            self._codes = self.x_train
            self._codes_test = self.x_test
        elif self._method.startswith('PCA'):
            whiten = not self._method.endswith('-UW')
            pca = PCA(dims=self._n_dims, whiten=whiten)
            self._codes = pca.fit_transform(self.x_train)
            self._codes_test = pca.encode(self.x_test)
        elif self._method == 'random':
            vecs = np.random.randn(self.x_train.shape[1], self._n_dims)
            vecs /= np.linalg.norm(vecs, axis=0)
            self._codes = self.x_train @ vecs
            self._codes_test = self.x_test @ vecs
        else:
            raise ValueError("Invalid method: must be one of %s, got %s" % (self._METHODS, self._method))


def embed():

    # TODO:  Add per-embedding args  --TSNE-perplexity --UMAP-n_neighbors
    description = "Load a trained autoencoder and create embeddings for the MNIST dataset."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("weights_file", help="Path to the trained network weights file")
    args = parser.parse_args()
    LatentRepEmbedder(args.weights_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # embed()
    ImageEmbedder(method='PCA', n_dims=16, map_size_wh=(4096//2, 4096//2))


#NOTE:  ADD NONWHITENING TO DENSE AND VAE!!!