import numpy as np
import matplotlib.pyplot as plt
from mnist import MNISTData
import logging
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Reshape
from tensorflow.keras.models import Model
from img_util import diff_img, make_img, make_digit_mosaic
import os

from matplotlib.gridspec import GridSpec


class DenseExperiment(object):
    _DEFAULT_ACT_FNS = {'internal': 'relu',
                        'encoding': 'relu'}

    def __init__(self, enc_layers=(64,), n_epochs=25, act_fns=None):
        """
        Initialize the Dense Experiment with a specified number of encoding units.
        :param enc_layers: list of layer sizes, final value is the encoding layer size.
        :param n_epochs: Number of epochs to train the autoencoder.
        """
        self._n_epochs = n_epochs
        self.enc_layer_desc = enc_layers
        self.code_size = enc_layers[-1]
        self.act_fns = self._DEFAULT_ACT_FNS if act_fns is None else act_fns

        if 'output' not in self.act_fns:
            # to resemble pixel values in [0, 1], probably don't change.
            self.act_fns['output'] = 'sigmoid'

        self._d_in = 784  # number of pixels in MNIST images

        self._load_data()
        self._init_model()
        logging.info("Experiment initialized:  %s" % self.get_name())

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
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')
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

    def _train(self):

        try:
            logging.info("Attempting to load pre-trained weights...")
            self._load_weights()
            return True
        except FileNotFoundError:
            logging.info("No pre-trained weights found, starting fresh training.")
        self.train_more()
        return False

    def train_more(self, plot_hist=True):

        self._history = self.autoencoder.fit(self.x_train, self.x_train,
                                             epochs=self._n_epochs, batch_size=512,
                                             validation_data=(self.x_test, self.x_test))
        logging.info("Training completed")
        if plot_hist:
            self._plot_history()
        self._save_weights()

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

    def run_experiment(self):
        is_trained = self._train()
        #if is_trained:
        #    self.train_more()  # uncomment to train loaded weights more.
        self._eval()

    def _plot_history(self):
        plt.plot(self._history.history['loss'])
        plt.plot(self._history.history['val_loss'])
        plt.title('Training history: model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.show()

    def plot(self, n_samp=39, show_diffs=False):

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

        suffix = "reconstructed img" if not show_diffs else "difference image"
        mid_labels = ["sample group %i - %s" % (i, suffix) for i in range(1, n_quantiles-1)]
        q_labels = ['Lowest Test MSE - %s' % suffix] + mid_labels + ['Highest Test MSE - %s' % suffix]
        n_colors = len(q_labels)
        cmap = plt.get_cmap('brg', n_colors)
        colors = [cmap(i) for i in range(n_colors)]
        for i, (inds, label) in enumerate(zip(self._quant_inds, q_labels)):
            show_mosaic(q_axes[i], inds, label, color=colors[i])

        model_name = self.get_name()
        title = "Autoencoder Model: %s" % (model_name, ) +\
            "\nData: n_train=%i, n_test = %i " % (self.x_train.shape[0], self.x_test.shape[0])
        if show_diffs:
            title += "          RED: decoded pixel >= 10% too high,"
        title += "\nResults: test MSE = %.4f (%.4f)" % (np.mean(self._mse_errors), np.std(self._mse_errors))
        if show_diffs:
            title += "             BLUE: decoded pixel >= 10% too low."

        plt.suptitle(title, fontsize=14)

        self._show_err_hist(hist_axis, q_labels, colors)

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


def dense_demo():
    de = DenseExperiment(enc_layers=(256, 64,), n_epochs=40)
    de.run_experiment()
    de.plot(show_diffs=False)
    de.plot(show_diffs=True)
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dense_demo()
