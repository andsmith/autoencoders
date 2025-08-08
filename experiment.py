"""
base class for autoencoder experiments
"""
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

class AutoencoderExperiment(ABC):
    """
    Base class for autoencoder experiments.
    """

    def __init__(self):
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

    @abstractmethod
    def encode_samples(self, samples):
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
        return parser.parse_args()