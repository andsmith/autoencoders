import numpy as np
import matplotlib.pyplot as plt
from mnist import MNISTData
import logging
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from img_util import diff_img, make_img, make_digit_mosaic
import os

from matplotlib.gridspec import GridSpec


class DenseExperiment:

    def __init__(self, n_enc=64):
        """
        Initialize the Dense Experiment with a specified number of encoding units.
        :param n_enc: Number of encoding units in the autoencoder.
        """
        self.n_epoch = 25
        self.n_enc = n_enc
        self.encode_act_fn = 'relu'
        self.decode_act_fn = 'sigmoid'

        self.mnist_data = MNISTData()
        logging.info("Dense Experiment initialized with n_enc=%d", self.n_enc)

        self.d = 784  # number of pixels in MNIST images

        self._load_data()

        self._init_model()

    def is_trained(self):
        # check model
        if self.autoencoder.trainable_weights:
            return True
        return False

    def _init_model(self):
        logging.info("Initializing model with n_enc=%d", self.n_enc)

        inputs = Input(shape=(self.d,))
        encoding = Dense(self.n_enc, activation=self.encode_act_fn)(inputs)
        encoder = Model(inputs=inputs, outputs=encoding)

        code_inputs = Input(shape=(self.n_enc,))
        decoding = Dense(self.d, activation=self.decode_act_fn)(code_inputs)
        decoder = Model(inputs=code_inputs, outputs=decoding)

        self.autoencoder = Model(inputs=inputs, outputs=decoder(encoder(inputs)))
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        logging.info("Model initialized and compiled")

    def encode_samples(self, samples):
        if not self.is_trained():
            raise RuntimeError("Model is not trained. Please train the model before encoding samples.")
        samples = samples.reshape(-1, 784)
        encoded_samples = self.autoencoder.layers[1](samples)
        return encoded_samples.numpy()

    def decode_samples(self, encoded_samples):
        if not self.is_trained():
            raise RuntimeError("Model is not trained. Please train the model before decoding samples.")
        decoded_samples = self.autoencoder.layers[-1](encoded_samples)
        return decoded_samples.numpy()

    def get_name(self, file_ext=False):
        fname = ("Dense(%i-%s-%s)_TrainEpochs=%i" % (self.n_enc, self.encode_act_fn, self.decode_act_fn, self.n_epoch))
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

    def _load_data(self):
        self.x_train = self.mnist_data.x_train.reshape(-1, self.d)
        self.x_test = self.mnist_data.x_test.reshape(-1, self.d)
        self.y_train = self.mnist_data.y_train
        self.y_test = self.mnist_data.y_test

    def _train(self, plot_hist=False):

        try:
            logging.info("Attempting to load pre-trained weights...")
            self._load_weights()
            return
        except FileNotFoundError:
            logging.info("No pre-trained weights found, starting fresh training.")

        self._history = self.autoencoder.fit(self.x_train, self.x_train,
                                             epochs=self.n_epoch, batch_size=512,
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
        self._train()
        self._eval()

    def _plot_history(self):
        plt.plot(self._history.history['loss'])
        plt.plot(self._history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.show()

    def plot(self, n_samp=30):

        def show_diff_mosaic(ax, inds, title):
            diff_imgs = [diff_img(self.x_test[i], self._decoded_test[i]) for i in inds]
            diff_image = make_digit_mosaic(diff_imgs, mosaic_aspect=3.0)
            ax.imshow(diff_image)
            ax.set_title(title, fontsize=14)
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

        hist_axis = fig.add_subplot(gs[3, :])

        best_inds = self._order[:n_samp]
        worst_inds = self._order[-n_samp:]
        quantile_sample_indices = np.linspace(0, len(self._order)-1, n_quantiles, dtype=int)[1:-1]
        self._quant_inds = [best_inds]
        for ind in quantile_sample_indices:
            self._quant_inds.append(self._order[ind-n_samp//2:ind+n_samp//2])
        self._quant_inds.append(worst_inds)

        q_labels = ['Best'] + ["Q%i" % i for i in range(1, n_quantiles-1)] + ['Worst']
        for i, (inds, label) in enumerate(zip(self._quant_inds, q_labels)):
            show_diff_mosaic(q_axes[i], inds, label)

        model_name = self.get_name()
        title = "%s\nn_test = %i, MSE = %.4f (%.4f) " % (model_name, self._mse_errors.size,
                                                         np.mean(self._mse_errors), np.std(self._mse_errors)) + \
            "\nRED: decoded pixel >= 10% too high,\nBLUE: decoded pixel >= 10% too low"

        plt.suptitle(title, fontsize=14)

        self._show_err_hist(hist_axis, q_labels)

    def _show_err_hist(self, ax,labels):
        ax.hist(self._mse_errors, bins=100, color='blue', alpha=0.8)
        ax.set_title('MSE Error Histogram', fontsize=14)
        ax.set_xlabel('MSE')
        ax.set_ylabel('N')

        # Draw vertical bands for ranges of plot images

        def draw_band(ax, band_index, label):

            err_range = np.min(self._mse_errors[self._quant_inds[band_index]]), np.max(
                self._mse_errors[self._quant_inds[band_index]])
            ax.axvspan(err_range[0], err_range[1], alpha=0.6, label=label)
        for i, label in enumerate(labels):
            draw_band(ax, i, label)
        ax.legend(loc='upper center', fontsize=10)


def dense_demo():
    de = DenseExperiment(n_enc=64)
    de.run_experiment()
    de.plot()
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dense_demo()
