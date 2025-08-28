"""
base class for autoencoder experiments
"""
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pca import MNISTPCA as PCA

from mnist import datasets
import logging


class AutoencoderExperiment(ABC):
    """
    Base class for autoencoder experiments.
    """

    def __init__(self, dataset, pca_dims, enc_layers, dec_layers=None, whiten_input=False, n_train_samples=0,
                 binary_input=True, d_latent=16, batch_size=512,  learning_rate=1e-3):
        """
        Initialize the AutoencoderExperiment.
        :param pca_dims:
          if >=1, number of dimensions for PCA, 
              =0, no pca (just scale normalization), 
             < 1, fraction of variance
        :param n_train_samples: Number of training samples to use (0 for all 60k).
        """
        if dataset not in datasets:
            raise ValueError("Dataset must be one of the following: %s" % list(datasets.keys()))
        self.dataset = dataset
        self.whiten_input = whiten_input
        self.learning_rate = learning_rate
        self.pca = PCA(dims=pca_dims, whiten=whiten_input, dataset=dataset)
        self.code_size = d_latent
        self.batch_size = batch_size
        self.n_train_samples = n_train_samples
        self.enc_layer_desc = enc_layers
        self.dec_layer_desc = dec_layers
        self.binary_input = binary_input
        self.pca_dims = self._load_data(binarize=binary_input)
        self._d_in = self.pca.d_out  # after loading data
        self._d_out = 784

        self._history_dict = {
            'loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        self._init_model()

    @abstractmethod
    def get_name(self):
        """
        Get the name of the experiment with all the model hyperparameters, layer sizes, etc.
        :return: string
        """
        pass

    @staticmethod
    @abstractmethod
    def parse_filename(self, filename):
        """
        Inverse of .get_name(), get model hyperparameters from the description in the filename
        :param filename: The filename to parse.
        :return: A dictionary with parsed parameters.
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

    def encode_samples(self, samples, raw=True):
        if raw:
            samples = self.pca.encode(samples)
        return self._encode_samples(samples)

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
    def load_weights(self,filename=None):
        pass

    @staticmethod
    @abstractmethod
    def from_filename(filename):
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

    def _accumulate_history(self, more_history):
        for key in more_history.keys():
            if key not in self._history_dict:
                self._history_dict[key] = []
            self._history_dict[key].extend(more_history[key])

    @staticmethod
    def get_args(description=None, extra_args=(), layers_default=[64]):
        """
        Syntax:  python dense.py --layers 512 128 64 --epochs 25 --stages 10 --no_plot
        this creates a dense autoencoder with encoding layers that have 512, 
        128, and 64 units (code size 64), trained for 10 rounds of 25 epochs each.
        The --no-plot option saves images instead of showing them.
        """
        description = "Run an autoencoder." if description is None else description
        parser = ArgumentParser(description=description)
        parser.add_argument('--dataset', type=str, default='digits',
                            help="Which dataset to use: "
                            "    'digits' (MNIST, handwritten digits),"
                            "    'fashion' (Fashion-MNIST),"
                            "    'numeric' (Typeface-MNIST),"
                            "    'alphanumeric' (94_character_TMNIST)")
        parser.add_argument('--pca_dims', type=float, default=25,
                            help="PCA-preprocessing:[=0, whitening, no pca] / [int>0, number of PCA dims] / [0<float<1, frac of variance to keep]")
        parser.add_argument('--whiten', action='store_true',
                            help='If set, each PCA feature is z-scored, else will have its original distribution.')
        parser.add_argument('--dropout_layer', type=int, default=None,
                            help='Which encoding layer uses dropout (index into --layers param, cannot be final/coding layer)')
        parser.add_argument('--batch_size', type=int, default=256,
                            help="Batch size for training (Default 256)"),
        parser.add_argument('--d_latent', type=int, default=16,
                            help='Dimensionality of the latent space (default: 16)')
        parser.add_argument('--dropout_rate', type=float, default=0.0,
                            help='Dropout rate to apply after each dense layer (default: 0.0)')
        parser.add_argument('--layers', type=int, nargs='+', default=layers_default,
                            help='List of encoder layer sizes (default, %i layers: %s)' % (len(layers_default), layers_default))
        parser.add_argument('--epochs', type=int, default=25,
                            help='Number of epochs to train each stage (default: 25)')
        parser.add_argument('--stages', type=int, default=1,
                            help='Number of times to train for the number of epochs, generate plots between each (default: 1)')
        parser.add_argument('--learn_rate', type=float, default=1e-3,
                            help='Learning rate for the optimizer (default: 1e-3)')
        parser.add_argument('--no_plot', action='store_true',
                            help='If set, saves images instead of showing them interactively.')
        parser.add_argument('--dec_layers', type=int, nargs='+', default=None,
                            help='List of decoder layer sizes (default: None, encoding layers reversed)')
        parser.add_argument('--binary_input', action='store_true', default=False,
                            help='If set, binarizes the input images (default: False)')
        for arg in (extra_args):
            parser.add_argument(arg['name'], **{k: v for k, v in arg.items() if k != 'name'})

        parsed = parser.parse_args()

        # Checks:
        if parsed.dropout_layer is not None:
            if parsed.dropout_rate == 0.0:
                parser.error("Dropout rate must be > 0.0 if dropout layer is specified.")
            if parsed.dropout_layer >= len(parsed.layers):
                parser.error("Dropout layer cannot be the final/coding layer")
            parsed.dropout = {'layer': parsed.dropout_layer,
                              'rate': parsed.dropout_rate}
        else:
            parsed.dropout = None

        if parsed.pca_dims == 0.0:
            parsed.pca_dims = 0
        elif parsed.pca_dims > 1.0:
            parsed.pca_dims = int(parsed.pca_dims)

        return parsed

    def _load_data(self, binarize=False):
        """
        return dimensionality of training data
        """

        self.mnist_data = datasets[self.dataset]()
        self.x_train = self.mnist_data.x_train.reshape(-1, 28*28)
        self.x_test = self.mnist_data.x_test.reshape(-1, 28*28)
        self.y_train, self.y_test = self.mnist_data.y_train, self.mnist_data.y_test

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

        logging.info("Data loaded: %i training samples, %i test samples.%s",
                     self.x_train.shape[0], self.x_test.shape[0], (" (BINARIZED: > 0.5)" if binarize else ""))
        return self.pca.d_out

    def _maybe_save_fig(self, fig, filename):
        if self._save_figs:
            fig.tight_layout()
            fig.savefig(filename, bbox_inches='tight')
            plt.close(fig)
            logging.info("Figure saved to %s", filename)
            return filename
