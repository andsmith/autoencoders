from fileinput import filename
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNISTData
import logging
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Reshape
from tensorflow.keras.models import Model
from img_util import diff_img, make_img, make_digit_mosaic
import os
from argparse import ArgumentParser
from matplotlib.gridspec import GridSpec
import json
from experiment import AutoencoderExperiment
import re


class DenseExperiment(AutoencoderExperiment):
    _DEFAULT_ACT_FNS = {'internal': 'relu',
                        'encoding': 'relu'}

    def __init__(self, enc_layers=(64,), act_fns=None, **kwargs):
        """
        Initialize the Dense Experiment with a specified number of encoding units.
        :param enc_layers: list of layer sizes, final value is the encoding layer size.
        :param n_epochs: Number of epochs to train the autoencoder.
        :param act_fns: Dictionary of activation functions for internal, encoding, and output layers.
                        If None, uses default activation functions.
        """

        self._history_dict = None
        if 'history_dict'  in kwargs:
            self._history_dict = kwargs['history_dict']
            del kwargs['history_dict']

        super().__init__(**kwargs)

        self.enc_layer_desc = enc_layers
        self.code_size = enc_layers[-1]
        self.act_fns = self._DEFAULT_ACT_FNS if act_fns is None else act_fns
        self._save_figs = None

        self._stage = 0
        self._epoch = 0

        if 'output' not in self.act_fns:
            # to resemble pixel values in [0, 1], probably don't change.
            self.act_fns['output'] = 'sigmoid'
        self._d_in = self.pca.pca_dims
        self._d_out = 784
        self._init_model()
        logging.info("Experiment initialized:  %s" % self.get_name())
        if isinstance(self.encoder, Model):
            self.print_model_architecture(self.encoder, self.decoder, self.autoencoder)

    def get_name(self, file_ext=None):
        desc_str = "_".join([str(n) for n in self.enc_layer_desc])
        pca_str = self.pca.get_name()
        fname = ("Dense(%s_units=%s_encode=%s_internal=%s)" %
                 (pca_str, desc_str, self.act_fns['encoding'], self.act_fns['internal']))
        if file_ext == 'weights':
            fname += ".weights.h5"
        elif file_ext == 'history':
            fname += "history.json"
        elif file_ext is not None:
            raise ValueError("Unknown file extension type: %s" % file_ext)
        return fname

    def _init_encoder_layers(self, inputs):

        encoder_layers = []
        for i, n_units in enumerate(self.enc_layer_desc):
            if i == 0:
                layer_input = inputs
            else:
                layer_input = encoder_layers[-1]

            act_fn = self.act_fns['internal'] if i < len(self.enc_layer_desc) - 1 else self.act_fns['encoding']
            layer = Dense(n_units, activation=act_fn, name=f'encoder_l{i}')(layer_input)
            encoder_layers.append(layer)

        return encoder_layers

    def _init_decoder_layers(self, encoding):
        decoder_layers = []
        decoder_layer_desc = list(reversed(self.enc_layer_desc[:-1]))  # skip the last encoding layer
        decoder_layer_desc.append(self._d_out)  # add the input layer size for decoding
        for i, n_units in enumerate(decoder_layer_desc):
            if i == 0:
                layer_input = encoding
            else:
                layer_input = decoder_layers[-1]
            act_fn = self.act_fns['internal'] if i < len(decoder_layer_desc) - 1 else self.act_fns['output']
            layer = Dense(n_units, activation=act_fn, name=f'decoder_l{i}')(layer_input)
            decoder_layers.append(layer)
        return decoder_layers

    def _init_model(self):
        """
        Initialize all the layers, keep references so internal states can be accessed later.
        """
        logging.info("Initializing auto encoder with encoding_layers=%s", self.enc_layer_desc)
        inputs = Input(shape=(self._d_in,))
        self.encoding_layers = self._init_encoder_layers(inputs)
        encoding = self.encoding_layers[-1]
        self.decoding_layers = self._init_decoder_layers(encoding)
        decoding = self.decoding_layers[-1]
        self.encoder = Model(inputs=inputs, outputs=encoding, name='encoder')
        self.autoencoder = Model(inputs=inputs, outputs=decoding, name='autoencoder')
        self.decoder = Model(inputs=encoding, outputs=decoding, name='decoder')
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        logging.info("Autoencoder model initialized and compiled")

    def is_trained(self):
        # check model
        if self.autoencoder.trainable_weights:
            return True
        return False

    def _encode_samples(self, samples):
        if not self.is_trained():
            raise RuntimeError("Model is not trained. Please train the model before encoding samples.")
        samples = samples.reshape(-1, self._d_in)
        encoded_samples = self.encoder.predict(samples)
        return encoded_samples

    def decode_samples(self, encoded_samples, reshape=False):
        if not self.is_trained():
            raise RuntimeError("Model is not trained. Please train the model before decoding samples.")
        encoded_samples = encoded_samples.reshape(-1, self.code_size)
        decoded_samples = self.decoder.predict(encoded_samples)
        if reshape:
            if encoded_samples.ndim == 1 or encoded_samples.shape[0] == 1:
                decoded_samples = decoded_samples.reshape(28, 28)
            else:
                decoded_samples = decoded_samples.reshape(-1, 28, 28)
        return decoded_samples

    def save_weights(self, path='.'):
        filename = self.get_name(file_ext='weights')
        self.autoencoder.save_weights(filename)
        logging.info("Model weights saved to %s", os.path.join(path, filename))
        # save history
        hist = self.get_name(file_ext='history')
        with open(os.path.join(path, hist), 'w') as f:
            json.dump(self._history_dict, f)
        logging.info("Training history saved to %s", os.path.join(path, hist))

    def load_weights(self, path='.'):
        filename = self.get_name(file_ext='weights')
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            self.autoencoder.load_weights(full_path)
            logging.info("Model weights loaded from %s", full_path)

            hist = self.get_name(file_ext='history')
            hist_path = os.path.join(path, hist)
            if os.path.exists(hist_path):
                with open(hist_path, 'r') as f:
                    self._history_dict = json.load(f)
                logging.info("Training history loaded from %s", hist_path)
            else:
                self._history_dict = None
                logging.info("No training history found at %s", hist_path)
        else:
            raise FileNotFoundError(f"Model weights file {full_path} not found.")

    @staticmethod
    def parse_filename(filename):
        """
        Parse the filename to extract encoding layer sizes and other parameters.
        :param filename: The filename to parse.
        :return: A dictionary with parsed parameters.
        """
        pattern = r'Dense\((.*?)_encode=(.*?)_internal=(.*?)\)\.weights\.h5'
        match = re.match(pattern, filename)
        if match:
            enc_layers = tuple(map(int, match.group(1).split(',')))
            encoding_act_fn = match.group(2)
            internal_act_fn = match.group(3)
            return {
                'enc_layers': enc_layers,
                'encoding_act_fn': encoding_act_fn,
                'internal_act_fn': internal_act_fn,
            }
        else:
            raise ValueError(f"Filename {filename} does not match expected format.")

    @staticmethod
    def from_filename(filename):
        params = DenseExperiment.parse_filename(filename)
        network = DenseExperiment(
            enc_layers=params['enc_layers'],
            act_fns={
                'encoding': params['encoding_act_fn'],
                'internal': params['internal_act_fn']
            }
        )
        network.load_weights(path=os.path.dirname(filename))
        return network

    def _attempt_resume(self):
        try:
            logging.info("Attempting to load pre-trained weights...")
            self.load_weights()
            return True
        except FileNotFoundError:
            logging.info("No pre-trained weights found, starting fresh training.")
        return False

    def train_more(self, n_epochs=10, save_wts=True):
        more_history = self.autoencoder.fit(self.x_train_pca, self.x_train,
                                            epochs=n_epochs, batch_size=512,
                                            validation_data=(self.x_test_pca, self.x_test))

        if self._history_dict is None:
            self._history_dict = more_history.history
        else:
            self._accumulate_history(more_history)

        if save_wts:
            self.save_weights()
        return self._eval()

    def _accumulate_history(self, more_history):
        if self._history_dict is None:
            self._history_dict = more_history.history
        else:
            for key in more_history.history.keys():
                if key not in self._history_dict:
                    self._history_dict[key] = []
                self._history_dict[key].extend(more_history.history[key])

    def _eval(self):

        def mse_err(imageA, imageB):
            err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
            err /= float(imageA.shape[0])
            return err

        self._reconstructed_test = self.autoencoder.predict(self.x_test_pca)

        self._mse_errors = np.array([mse_err(img_a, img_b)
                                    for img_a, img_b in zip(self.x_test, self._reconstructed_test)])
        self._order = np.argsort(self._mse_errors)
        logging.info("Evaluation completed on %i samples:", self._mse_errors.size)
        logging.info("\tMean squared error: %.4f (%.4f)", np.mean(self._mse_errors), np.std(self._mse_errors))

    def _plot_history(self):
        prefix = self.get_name()
        suffix = "stage_%i" % (self._stage+1)
        filename = "%s_training_history_%s.png" % (prefix, suffix)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self._history_dict['loss'])
        ax.plot(self._history_dict['val_loss'])
        ax.set_title('Training history: model loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend(['Train', 'Test'], loc='upper right')
        self._maybe_save_fig(fig, filename)
        return None

    def plot(self, n_samp=39, show_diffs=False):

        prefix = self.get_name()
        suffix = "stage_%i" % (self._stage+1)

        def show_mosaic(ax, inds, title, color):
            if not show_diffs:
                reconstructed_imgs = [make_img(self._reconstructed_test[i]) for i in inds]
                image = make_digit_mosaic(reconstructed_imgs, mosaic_aspect=4.0)
            else:
                diff_imgs = [diff_img(self.x_test[i], self._reconstructed_test[i]) for i in inds]
                image = make_digit_mosaic(diff_imgs, mosaic_aspect=4.0)
            ax.imshow(image)
            ax.set_title(title, fontsize=12, color=color)
            ax.axis('off')

        fig = plt.figure(constrained_layout=True, figsize=(10, 8))
        # show best, worst, and middle 4 quantiles above a histogram
        n_quantiles = 8
        n_q_rows = n_quantiles//2
        n_q_cols = n_quantiles // n_q_rows
        gs = GridSpec(n_q_rows+1, n_q_cols, figure=fig)
        q_axes = []
        for j in range(n_q_cols):
            for i in range(n_q_rows):
                ax = fig.add_subplot(gs[i, j])
                q_axes.append(ax)

        hist_axis = fig.add_subplot(gs[n_q_rows, :])

        best_inds = self._order[:n_samp]
        worst_inds = self._order[-n_samp:]
        quantile_sample_indices = np.linspace(0, len(self._order)-1, n_quantiles, dtype=int)[1:-1]
        self._quant_inds = [best_inds]
        for ind in quantile_sample_indices:
            extra = n_samp % 2
            self._quant_inds.append(self._order[ind-n_samp//2:ind+n_samp//2+extra])
        self._quant_inds.append(worst_inds)

        mid_labels = ["sample group %i - %s" % (i, suffix) for i in range(1, n_quantiles-1)]
        q_labels = ['Lowest Test MSE - %s' % suffix] + mid_labels + ['Highest Test MSE - %s' % suffix]
        n_colors = len(q_labels)
        cmap = plt.get_cmap('brg', n_colors)
        colors = [cmap(i) for i in range(n_colors)]
        for i, (inds, label) in enumerate(zip(self._quant_inds, q_labels)):
            show_mosaic(q_axes[i], inds, label, color=colors[i])

        model_name = self.get_name()
        title = "Autoencoder Model: %s (stage %i)" % (model_name, self._stage + 1) +\
            "\nData: n_train=%i, n_test = %i " % (self.x_train.shape[0], self.x_test.shape[0])
        if show_diffs:
            title += "          RED: decoded pixel >= 10% too high,"
        title += "\nResults: test MSE = %.4f (%.4f)" % (np.mean(self._mse_errors), np.std(self._mse_errors))
        if show_diffs:
            title += "             BLUE: decoded pixel >= 10% too low."

        plt.suptitle(title, fontsize=14)
        self._show_err_hist(hist_axis, q_labels, colors)
        filename = "%s_%s_%s.png" % (prefix, ("diffs" if show_diffs else "reconstructed"), suffix)
        self._maybe_save_fig(fig, filename)

    def _maybe_save_fig(self, fig, filename):
        if self._save_figs:
            fig.tight_layout()
            fig.savefig(filename, bbox_inches='tight')
            plt.close(fig)
            return filename
        # else:
        #    plt.show()
        #    return None

    def _show_err_hist(self, ax, labels, band_colors):
        ax.hist(self._mse_errors, bins=100, color='gray', alpha=0.8)
        ax.set_title('MSE distribution & sample group locations', fontsize=12)

        # Draw vertical bands for ranges of plot images

        def draw_band(ax, band_index, color, label):

            err_range = np.min(self._mse_errors[self._quant_inds[band_index]]), np.max(
                self._mse_errors[self._quant_inds[band_index]])
            ax.axvspan(err_range[0], err_range[1], color=color, alpha=0.6, label=label)

        for i, label in enumerate(labels):
            draw_band(ax, i, band_colors[i], label)
        # ax.legend(loc='upper center', fontsize=10)

    def run_staged_experiment(self, n_stages=10, n_epochs=25, save_figs=True):
        """
        Instead of training all at once, train for n_epochs, plot (save) the analysis,
        continue training, for n_stages total stages.
        :param n_stages: number of training stages
        :param n_epochs: number of epochs to train each stage
        :param save_figs: If True, saves plots to files instead of showing them interactively.
        """
        # first round, load weights if available
        self._attempt_resume()
        self._save_figs = save_figs
        for stage in range(n_stages):
            self._stage = stage
            logging.info("Running stage %i of %i", stage + 1, n_stages)
            self.train_more(n_epochs=n_epochs)
            self._plot_stage()

    def _plot_stage(self):
        self._plot_history()
        self.plot(show_diffs=False)
        self.plot(show_diffs=True)
        if not self._save_figs:
            plt.show()


def dense_demo():
    args = DenseExperiment.get_args("Train a dense autoencoder on MNIST data.")
    logging.info("Running Dense Autoencoder with args: %s", args)
    de = DenseExperiment(enc_layers=args.layers, pca_dims=args.pca_dims)
    de.run_staged_experiment(n_stages=args.stages, n_epochs=args.epochs, save_figs=args.no_plot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dense_demo()
