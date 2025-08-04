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


class DenseExperiment(object):
    _DEFAULT_ACT_FNS = {'internal': 'relu',
                        'encoding': 'relu'}

    def __init__(self, enc_layers=(64,), n_epochs=25, act_fns=None, save_figs=True):
        """
        Initialize the Dense Experiment with a specified number of encoding units.
        :param enc_layers: list of layer sizes, final value is the encoding layer size.
        :param n_epochs: Number of epochs to train the autoencoder.
        :param act_fns: Dictionary of activation functions for internal, encoding, and output layers.
                        If None, uses default activation functions.
        :param save_figs: If True, saves plots to files instead of showing them interactively.
        """
        self._n_epochs = n_epochs
        self.enc_layer_desc = enc_layers
        self.code_size = enc_layers[-1]
        self.act_fns = self._DEFAULT_ACT_FNS if act_fns is None else act_fns
        self._save_figs = save_figs

        self._stage = 0
        self._epoch = 0

        if 'output' not in self.act_fns:
            # to resemble pixel values in [0, 1], probably don't change.
            self.act_fns['output'] = 'sigmoid'

        self._d_in = 784  # number of pixels in MNIST images
        self._history = None
        self._load_data()
        self._init_model()
        logging.info("Experiment initialized:  %s" % self.get_name())

    def _get_loss_fn(self):
        return 'mean_squared_error'

    def _load_data(self, binarize=False):
        self.mnist_data = MNISTData()
        self.x_train = self.mnist_data.x_train.reshape(-1, self._d_in)
        self.x_test = self.mnist_data.x_test.reshape(-1, self._d_in)
        self.y_train = np.where(self.mnist_data.y_train)[1]
        self.y_test = np.where(self.mnist_data.y_test)[1]

        if binarize:
            self.x_train = (self.x_train > 0.5).astype(np.float32)
            self.x_test = (self.x_test > 0.5).astype(np.float32)

        logging.info("Data loaded: %i training samples, %i test samples", self.x_train.shape[0], self.x_test.shape[0])

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
        decoder_layer_desc.append(self._d_in)  # add the input layer size for decoding
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
        self.autoencoder.compile(optimizer='adam', loss=self._get_loss_fn())
        logging.info("Autoencoder model initialized and compiled")

    def is_trained(self):
        # check model
        if self.autoencoder.trainable_weights:
            return True
        return False

    def encode_samples(self, samples, binarize_thresh=None):
        if not self.is_trained():
            raise RuntimeError("Model is not trained. Please train the model before encoding samples.")
        samples = samples.reshape(-1, 784)
        encoded_samples = self.encoder.predict(samples)
        if binarize_thresh is not None:
            encoded_samples = (encoded_samples > binarize_thresh).astype(np.float32)
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

    def get_name(self, file_ext=False):
        desc_str = "_".join([str(n) for n in self.enc_layer_desc])
        fname = ("Dense(%s_encode=%s_internal=%s)_TrainEpochs=%i" %
                 (desc_str, self.act_fns['encoding'], self.act_fns['internal'], self._n_epochs))
        if file_ext:
            fname += ".weights.h5"
        return fname

    def _save_weights(self, path='.'):
        filename = self.get_name(file_ext=True)
        self.autoencoder.save_weights(filename)
        logging.info("Model weights saved to %s", os.path.join(path, filename))

    def _load_weights(self, path='.'):
        filename = self.get_name(file_ext=True)
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            self.autoencoder.load_weights(full_path)
            logging.info("Model weights loaded from %s", full_path)
        else:
            raise FileNotFoundError(f"Model weights file {full_path} not found.")

    def _attempt_resume(self):
        try:
            logging.info("Attempting to load pre-trained weights...")
            self._load_weights()
            return True
        except FileNotFoundError:
            logging.info("No pre-trained weights found, starting fresh training.")
        return False

    def train_more(self, n_epochs=None, save_wts=True):

        n_epochs = n_epochs if n_epochs is not None else self._n_epochs
        more_history = self.autoencoder.fit(self.x_train, self.x_train,
                                            epochs=n_epochs, batch_size=512,
                                            validation_data=(self.x_test, self.x_test))

        if self._history is None:
            self._history = more_history
        else:
            # merge histories
            self._history.history['loss'].extend(more_history.history['loss'])
            self._history.history['val_loss'].extend(more_history.history['val_loss'])
        if save_wts:
            self._save_weights()
        self._eval()

    def _eval(self):

        def mse_err(imageA, imageB):
            err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
            err /= float(imageA.shape[0])
            return err
        self._encoded_test = self.encode_samples(self.x_test)
        self._decoded_test = self.decode_samples(self._encoded_test)
        self._both_test = self.autoencoder.predict(self.x_test)

        self._mse_errors = np.array([mse_err(img_a, img_b) for img_a, img_b in zip(self.x_test, self._both_test)])
        self._order = np.argsort(self._mse_errors)
        logging.info("Evaluation completed on %i samples:", self._mse_errors.size)
        logging.info("\tMean squared error: %.4f (%.4f)", np.mean(self._mse_errors), np.std(self._mse_errors))

    def _plot_history(self):
        prefix = self.get_name(file_ext=False)
        suffix = "stage_%i" % (self._stage+1)
        filename = "%s_training_history_%s.png" % (prefix, suffix)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self._history.history['loss'])
        ax.plot(self._history.history['val_loss'])
        ax.set_title('Training history: model loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend(['Train', 'Test'], loc='upper right')
        self._maybe_save_fig(fig, filename)
        return None

    def plot(self, n_samp=39, show_diffs=False):

        prefix = self.get_name(file_ext=False)
        suffix = "stage_%i" % (self._stage+1)

        def show_mosaic(ax, inds, title, color):
            if not show_diffs:
                reconstructed_imgs = [make_img(self._decoded_test[i]) for i in inds]
                image = make_digit_mosaic(reconstructed_imgs, mosaic_aspect=4.0)
            else:
                diff_imgs = [diff_img(self.x_test[i], self._decoded_test[i]) for i in inds]
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

    def run_staged_experiment(self, n_stages=10):
        """
        Instead of training all at once, train for n_epochs, plot (save) the analysis,
        continue training, for n_stages total stages.
        """
        # first round, load weights if available
        self._attempt_resume()

        for stage in range(n_stages):
            self._stage = stage
            logging.info("Running stage %i of %i", stage + 1, n_stages)
            self.train_more()
            self._plot_history()
            self.plot(show_diffs=False)
            self.plot(show_diffs=True)
            if not self._save_figs:
                plt.show()


def _get_args():
    """
    Syntax:  python dense.py --layers 512 128 64 --epochs 25 --stages 10 --no_plot
      this creates a dense autoencoder with encoding layers that have 512, 
      128, and 64 units (code size 64), trained for 10 rounds of 25 epochs each.
      The --no-plot option saves images instead of showing them.
    """
    parser = ArgumentParser(description="Run a dense autoencoder experiment.")
    parser.add_argument('--layers', type=int, nargs='+', default=[64],
                        help='List of encoding layer sizes (default: [64])')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of epochs to train each stage (default: 25)')
    parser.add_argument('--stages', type=int, default=5,
                        help='Number of training stages (default: 5)')
    parser.add_argument('--no_plot', action='store_true',
                        help='If set, saves images instead of showing them interactively')
    return parser.parse_args()


def dense_demo():
    args = _get_args()
    logging.info("Running Dense Autoencoder with args: %s", args)
    de = DenseExperiment(enc_layers=args.layers, n_epochs=args.epochs, save_figs=args.no_plot)
    de.run_staged_experiment(n_stages=args.stages)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dense_demo()
