"""
base class for autoencoder experiments
"""
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pca import PCA

from mnist import MNISTData
import logging
class AutoencoderExperiment(ABC):
    """
    Base class for autoencoder experiments.
    """

    def __init__(self, pca_dims, n_train_samples=0, bw_input=False):
        """
        Initialize the AutoencoderExperiment.
        :param pca_dims:
          if >=1, number of dimensions for PCA, 
              =0, no pca (just scale normalization), 
             < 1, fraction of variance
        :param n_train_samples: Number of training samples to use (0 for all 60k).
        """
        self.pca = PCA(dims=pca_dims)
        self.n_train_samples = n_train_samples
        self.pca_dims = self._load_data(binarize=bw_input)
        self._init_model()


    @abstractmethod
    def _init_model(self):
        """
        Initialize the model with the given encoder and decoder.
        :param encoder: Encoder model
        :param decoder: Decoder model
        """
        pass

    @staticmethod
    def print_model_architecture(encoder, decoder, model):
        """
        Print the architecture of the model, for each layer:
           - Input/output shape, number of units, activation function.

        :param model: Keras model
        """
        def _shape(layer_data):
            return layer_data.shape if not isinstance(layer_data, list) else "(na)"

        def _print_part(model, title):
            print(title)
            for layer in model.layers:
                print(f"{layer.name}: input_shape={_shape(layer.input)}, output_shape={_shape(layer.output)}")
        print("\n\n\nModel Architecture:")
        _print_part(encoder, "\nEncoder Architecture:")
        _print_part(decoder, "\nDecoder Architecture:")
        _print_part(model, "\nFull Model Architecture:")
        print("\n\n\n")

    def encode_samples(self, samples):
        pca_samples = self.pca.encode(samples)
        return self._encode_samples(pca_samples)

    @abstractmethod
    def _encode_samples(self, samples):
        """
        Encode samples into the latent space.
        :param samples: array of shape (n_samples, d_input)
        :return: array of shape (n_samples, d_latent)
        """
        pass

    @abstractmethod
    def decode_samples(self, codes):
        """
        Decode samples from the latent space.
        :param codes: array of shape (n_samples, d_latent)
        :return: array of shape (n_samples, d_input)
        """
        pass

    @abstractmethod
    def save_weights(self, filename):
        pass

    @abstractmethod
    def load_weights(self, filename):
        pass

    @staticmethod
    @abstractmethod
    def from_filename(cls, filename):
        """
        Parse the arcitecture of the autoencoder from the filename and
        create one, load its weights.
        :param filename: filename to parse
        :return: AutoencoderExperiment instance
        """
        pass

    @abstractmethod
    def run_staged_experiment(self, n_stages=10, n_epochs=25):
        """
        For each of the n_stages, train for n_epochs, plot intermediate results,
        save weights, and continue training.

        :param n_stages: number of training stages
        :param n_epochs: number of epochs to train each stage
        """
        pass

    @staticmethod
    def get_args(description=None, extra_args=()):
        """
        Syntax:  python dense.py --layers 512 128 64 --epochs 25 --stages 10 --no_plot
        this creates a dense autoencoder with encoding layers that have 512, 
        128, and 64 units (code size 64), trained for 10 rounds of 25 epochs each.
        The --no-plot option saves images instead of showing them.
        """
        description = "Run an autoencoder." if description is None else description
        parser = ArgumentParser(description=description)
        parser.add_argument('--pca_dims', type=float, default=25,
                            help="PCA-preprocessing:[=0, whitening, no pca] / [int>0, number of PCA dims] / [0<float<1, frac of variance to keep]")
        parser.add_argument('--layers', type=int, nargs='+', default=[64],
                            help='List of encoding layer sizes (default: [64])')
        parser.add_argument('--epochs', type=int, default=25,
                            help='Number of epochs to train each stage (default: 25)')
        parser.add_argument('--stages', type=int, default=5,
                            help='Number of training stages (default: 5)')
        parser.add_argument('--no_plot', action='store_true',
                            help='If set, saves images instead of showing them interactively')
        for arg in (extra_args):
            parser.add_argument(arg['name'], **{k: v for k, v in arg.items() if k != 'name'})

        parsed= parser.parse_args()

        if parsed.pca_dims==0.0:
            parsed.pca_dims = 0
        elif parsed.pca_dims > 1.0:
            parsed.pca_dims = int(parsed.pca_dims)

        return parsed

    def _load_data(self, binarize=False):
        """
        return dimensionality of training data
        """
        self.mnist_data = MNISTData()
        self.x_train = self.mnist_data.x_train.reshape(-1, 28*28)
        self.x_test = self.mnist_data.x_test.reshape(-1, 28*28)
        self.y_train = np.where(self.mnist_data.y_train)[1]
        self.y_test = np.where(self.mnist_data.y_test)[1]

        if self.n_train_samples > 0:
            inds = np.random.choice(len(self.x_train), self.n_train_samples, replace=False)
            self.x_train = self.x_train[inds]
            self.y_train = self.y_train[inds]

        if binarize:
            self.x_train = (self.x_train > 0.5).astype(np.float32)
            self.x_test = (self.x_test > 0.5).astype(np.float32)
            

        # Finally, train PCA layer
        self.x_train_pca = self.pca.fit_transform(self.x_train)
        self.x_test_pca = self.pca.encode(self.x_test)

        logging.info("Data loaded: %i training samples, %i test samples", self.x_train.shape[0], self.x_test.shape[0])
        return self.pca.pca_dims