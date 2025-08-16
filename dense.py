from fileinput import filename
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNISTData
import logging
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
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

    WORKING_DIR = "Dense-results"

    def __init__(self,
                 enc_layers=(64,),
                 pca_dims=64,
                 whiten_input=False,
                 dropout_info=None,
                 learning_rate=1e-3,
                 dec_layers=None,
                 dataset='digits',
                 batch_size=512,
                 d_latent=16,
                 **kwargs):
        """
        Initialize the Dense Experiment with a specified number of encoding units.
        :param enc_layers: list of layer sizes, final value is the encoding layer size.
        :param n_epochs: Number of epochs to train the autoencoder.
        :param act_fns: Dictionary of activation functions for internal, encoding, and output layers.
                        If None, uses default activation functions.
        dec_layers: List of layer sizes for the decoder (if different from encoder).
           NOTE:  in the feed forward order, (... -> encoder{n-1} -> encoder{n}/code -> decoder{0} -> ...)
        """
        self.dropout = dropout_info
        self.enc_layer_desc = enc_layers
        self.dec_layer_desc = dec_layers
        self.act_fns = self._DEFAULT_ACT_FNS  # not worth changing?
        self._save_figs = None

        self._stage = 0
        self._epoch = 0

        # Should be sigmoid to resemble pixel values in [0, 1], probably don't change.
        if 'output' not in self.act_fns:
            self.act_fns['output'] = 'sigmoid'
        self._d_out = 784

        super().__init__(pca_dims=pca_dims, enc_layers=enc_layers, dec_layers=dec_layers, batch_size=batch_size, d_latent=d_latent,
                         whiten_input=whiten_input, learning_rate=learning_rate, dataset=dataset, **kwargs)

        logging.info("Experiment initialized:  %s" % self.get_name())
        if isinstance(self.encoder, Model):
            self.print_model_architecture(self.encoder, self.decoder, self.autoencoder)

    def get_name(self, file_ext=None, suffix=None):
        desc_str = "-".join([str(n) for n in (self.enc_layer_desc+[self.code_size])])
        dec_desc_str = "_dec-units="+"-".join([str(n) for n in self.dec_layer_desc]
                                              ) if self.dec_layer_desc is not None else ""
        pca_str = self.pca.get_short_name()
        drop_str = "" if self.dropout is None else "_Drop(l=%i,r=%.2f)" % (self.dropout['layer'], self.dropout['rate'])
        fname = ("%s_Dense(%s_units=%s%s%s)" %
                 (self.dataset, pca_str, desc_str, dec_desc_str, drop_str))

        if suffix is not None:
            fname = "%s_%s" % (fname, suffix)

        if file_ext == 'weights':
            fname += ".weights.h5"
        elif file_ext == 'image':
            fname += ".png"
        elif file_ext == 'history':
            fname += "history.json"
        elif file_ext is not None:
            raise ValueError("Unknown file extension type: %s" % file_ext)

        if file_ext is not None:
            return os.path.join(DenseExperiment.WORKING_DIR, fname)
        return fname

    @staticmethod
    def parse_filename(filename):
        """
        Parse the filename to extract experiment parameters.
        :returns: dict with network architecture parameters, can be passed as a set of arguments to DenseExperiment().
        """
        print("\n%s\n" % filename)
        file = os.path.split(os.path.abspath(filename))[1]
        kind_pattern = r'([a-z]+)[_\-]([a-zA-Z\-\_]+)'

        # internal structure is everything between outer parentheses
        int_start, int_end = file.find("("), len(file)-file[::-1].find(")")

        internal = file[int_start:int_end]  # keep parentheses
        prefix = file[:int_start]

        pca_vs_rest_pattern = r'PCA\(([^\)]+)\)_(.+)'  # dataset-pca(pca_params)
        # PCA format is "<dims>,<whitening>" where <dims> is an int, and <whitening> is "W" or "UW"
        pca_int_pattern = r'(\d+),(W|UW)'
        # ints separated by dash, final is code size (must have at least 1 number here)
        enc_pattern = r'units=(.+?)[^\d-]'
        # optional, same format, assumed reverse of encoder if not present, l=layer (of encoder unit, can't be final/code layer), r=rate.
        dec_pattern = r'dec_units=(.?)'
        dropout_pattern = r'Drop\(l=(\d+),r=(\d\.?\d*)\)'

        file_kind_match = re.search(kind_pattern, prefix)
        if not file_kind_match:
            raise ValueError("No dataset_model string found in filename: %s, in prefix: %s" % (filename, prefix))
        dataset, dense = file_kind_match.groups()
        if dense != 'Dense':
            raise ValueError("Expected 'Dense' in dataset_model string, found: %s" % dense)

        pca_match = re.search(pca_vs_rest_pattern, internal)
        if not pca_match:
            raise ValueError("No PCA match found in filename: %s, in params: %s" % (filename, internal))
        pca_desc, arch_desc = pca_match.group(1), pca_match.group(2)
        pca_int_match = re.search(pca_int_pattern, pca_desc)
        if not pca_int_match:
            raise ValueError("No PCA integer match found in filename: %s, in params: %s" % (filename, pca_desc))
        dims, whiten = int(pca_int_match.group(1)), pca_int_match.group(2) == 'W'

        encoder_match = re.search(enc_pattern, arch_desc)
        if encoder_match:
            enc_layers = tuple(map(int, encoder_match.group(1).split('-')))
        else:
            raise ValueError("No encoder description 'units=...' found in filename: %s, in params: %s" %
                             (filename, arch_desc))

        decoder_match = re.search(dec_pattern, arch_desc)
        if decoder_match:
            dec_layers = tuple(map(int, decoder_match.group(1).split('-')))
        else:
            dec_layers = None

        dropout_match = re.search(dropout_pattern, arch_desc)
        if dropout_match:
            dropout_info = dict(dropout_layer=int(dropout_match.group(1)),
                                dropout_rate=float(dropout_match.group(2)))
        else:
            dropout_info = None

        params = {
            'enc_layers': list(enc_layers[:-1]),
            'd_latent': enc_layers[-1],
            'pca_dims': 0 if dims == 784 else dims,
            'whiten_input': whiten,
            'dropout_info': dropout_info,
            'dec_layers': dec_layers,
            'dataset': dataset
        }

        return params

    def _init_encoder_layers(self, inputs):
        enc_layers_sizes = self.enc_layer_desc+[self.code_size]
        encoder_layers = []
        for i, n_units in enumerate(enc_layers_sizes):
            if i == 0:
                layer_input = inputs
            else:
                layer_input = encoder_layers[-1]

            act_fn = self.act_fns['internal'] if i < len(enc_layers_sizes) - 1 else self.act_fns['encoding']
            layer = Dense(n_units, activation=act_fn, name=f'encoder_l{i}')(layer_input)
            encoder_layers.append(layer)
            if self.dropout is not None and i == self.dropout['layer']:
                logging.info("Adding dropout layer at %i with rate %.2f", i, self.dropout['rate'])
                layer = Dropout(self.dropout['rate'], name=f'dropout_l{i}')(layer)

            encoder_layers.append(layer)

        return encoder_layers

    def _init_decoder_layers(self, encoding):
        decoder_layers = []
        decoder_layer_desc = self.dec_layer_desc[:
                                                 ] if self.dec_layer_desc is not None else self.enc_layer_desc[:-1][::-1]
        print("---------------------------> ", decoder_layer_desc, self.dec_layer_desc)
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
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.autoencoder.compile(optimizer=self.optimizer, loss='mean_squared_error')
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

    def load_weights(self):
        filename = self.get_name(file_ext='weights')
        if os.path.exists(filename):
            self.autoencoder.load_weights(filename)
            logging.info("Model weights loaded from %s", filename)

            hist = self.get_name(file_ext='history')
            hist_path = self.get_name(file_ext='history')
            if os.path.exists(hist_path):
                with open(hist_path, 'r') as f:
                    self._history_dict = json.load(f)
                logging.info("Training history loaded from %s", hist_path)
            else:
                self._history_dict = None
                logging.info("No training history found at %s", hist_path)
        else:
            raise FileNotFoundError(f"Model weights file {filename} not found.")

    @staticmethod
    def from_filename(filename):
        filename = os.path.abspath(filename)
        logging.info("Loading DenseExperiment from filename: %s", filename)
        params = DenseExperiment.parse_filename(filename)
        network = DenseExperiment(**params)
        network.load_weights()
        return network

    def _attempt_resume(self):
        try:
            logging.info("Attempting to load pre-trained weights...")
            self.load_weights()
            logging.info("Pre-trained weights loaded successfully, continuing training.")
            return True
        except FileNotFoundError:
            logging.info("No pre-trained weights found, training from scratch.")
        return False

    def train_more(self, n_epochs=10, save_wts=True):
        more_history = self.autoencoder.fit(self.x_train_pca, self.x_train,
                                            epochs=n_epochs, batch_size=512,
                                            validation_data=(self.x_test_pca, self.x_test))
        hist_update = more_history.history
        hist_update['learning_rate'] = [self.learning_rate] * n_epochs
        self._accumulate_history(hist_update)

        if save_wts:
            self.save_weights()
        return self._eval()

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

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})

        ax1.plot(self._history_dict['loss'])
        ax1.plot(self._history_dict['val_loss'])
        ax1.set_title('Loss')
        ax2.plot(self._history_dict['learning_rate'])
        ax2.set_title('Learning Rate')
        ax2.set_xlabel('Epoch')
        # set both axes logarithmic:
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        # add grid
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax1.legend(['Train', 'Test'], loc='upper right')
        filename = self.get_name(file_ext='image', suffix="_history")
        plt.suptitle("Training History for model \n%s" % self.get_name(), fontsize=14)
        self._maybe_save_fig(fig, filename)
        return None

    def plot(self, n_samp=39, show_diffs=False):

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
        suffix = "stage = %i" % (self._stage,)
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
        ds = "differences" if show_diffs else "reconstructions"
        filename = self.get_name(file_ext='image', suffix="stage=%i_ds=%s" % (self._stage, ds))
        self._maybe_save_fig(fig, filename)

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

        if not os.path.exists(DenseExperiment.WORKING_DIR):
            os.makedirs(DenseExperiment.WORKING_DIR)

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

    de = DenseExperiment(enc_layers=args.layers,
                         dec_layers=args.dec_layers,
                         pca_dims=args.pca_dims,
                         whiten_input=args.whiten,
                         learning_rate=args.learn_rate,
                         dropout_info=args.dropout,
                         dataset=args.dataset,
                         d_latent=args.d_latent,
                         batch_size=args.batch_size)

    de.run_staged_experiment(n_stages=args.stages,
                             n_epochs=args.epochs,
                             save_figs=args.no_plot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dense_demo()
