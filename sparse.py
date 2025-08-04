import os
from argparse import ArgumentParser
from dense import DenseExperiment
import numpy as np
import matplotlib.pyplot as plt
import logging
from mnist import MNISTData
from img_util import make_img, make_digit_mosaic
from matplotlib.gridspec import GridSpec
import seaborn as sns
import tensorflow as tf


@tf.custom_gradient
def binary_step_with_straight_through_estimator(x):
    y = tf.cast(x > 0.0, tf.float32)

    def grad(dy):
        return dy
    return y, grad


class BinaryActivation(tf.keras.layers.Layer):
    def call(self, inputs):
        return binary_step_with_straight_through_estimator(inputs)


class SparseExperiment(DenseExperiment):
    _DEFAULT_ACT_FNS = {
        'encoding': 'relu',
        'binarize_code': True,
        'internal': 'relu',
    }

    def __init__(self, enc_layers=(64,), n_epochs=25, act_fns=None, reg_lambda=0.5, save_figs=True):
        activation_functions = self._DEFAULT_ACT_FNS.copy() if act_fns is None else act_fns
        self.reg_lambda = reg_lambda
        super().__init__(enc_layers=enc_layers, n_epochs=n_epochs, act_fns=activation_functions, save_figs=save_figs)
        if self.act_fns['binarize_code'] and self.act_fns['encoding'] != 'relu':
            raise ValueError("Binarization requires 'relu' activation for encoding layer, but got '%s'" %
                             self.act_fns['encoding'])

    def _maybe_save_fig(self, fig, filename):
        if hasattr(self, '_save_figs') and self._save_figs:
            fig.tight_layout()
            fig.savefig(filename, bbox_inches='tight')
            logging.info("Saved figure to %s", filename)
            plt.close(fig)
            return filename
        # else:
        #     plt.show()
        #     return None

    def run_staged_experiment(self, n_stages=5):
        self._attempt_resume()
        for stage in range(n_stages):
            self._stage = stage
            logging.info("Running stage %i of %i", stage + 1, n_stages)
            self.train_more()
            self._plot_history()
            self.plot(show_diffs=False)
            self.plot(show_diffs=True)

            encoded_per_digit = self.plot_sparsity()
            self.plot_code_samples(encoded_per_digit)
            if not self._save_figs:
                plt.show()

            

    def _init_encoder_layers(self, inputs):
        encoding_layers = super()._init_encoder_layers(inputs)
        encoding = encoding_layers[-1]
        if self.act_fns['binarize_code']:
            encoding = BinaryActivation(name='binarize')(encoding)
        encoding_layers.append(encoding)
        return encoding_layers

    def get_name(self, file_ext=False):
        def func_to_str(func):
            if hasattr(func, '__name__'):
                return func.__name__
            return str(func)

        desc_str = "_".join([str(n) for n in self.enc_layer_desc])
        bin_str = "_REAL" if not self.act_fns['binarize_code'] else "_BINARY"
        fname = ("Sparse(%s%s_regL=%.1f)_TrainEpochs=%i" %
                 (desc_str, bin_str, self.reg_lambda, self._n_epochs))
        if file_ext:
            fname += ".weights.h5"
        return fname
    
    def _get_mse_term(self, x_true, x_pred):
        """
        Calculate the mean squared error between the true and predicted images.
        """
        mse_loss = tf.keras.losses.MeanSquaredError()(x_true, x_pred)
        return mse_loss
    
    def _get_sparse_term(self, x_true, x_pred, reg_method='l1'):
        """
        Calculate the sparsity term based on the encoding of the input images.
        """
        z = self.encoder(x_true)
        if reg_method == 'entropy':
            # Minimize entropy
            binary_reg_term = tf.reduce_mean(z * (1 - z))
        elif reg_method == 'l1':
            # Use the L1 norm of the encoded vector, normalized by the number of samples (mean active feature count)
            binary_reg_term = tf.reduce_mean(z)
        elif reg_method is None:
            return 0.0
        else:
            raise ValueError("Unknown method for sparse loss: %s" % reg_method)

        return binary_reg_term

    def sparse_loss(self, x_true, x_pred, reg_method='l1'):
        mse_loss = self._get_mse_term(x_true, x_pred)
        binary_reg_term = self._get_sparse_term(x_true, x_pred, reg_method)
        return mse_loss * (1 - self.reg_lambda) + binary_reg_term * self.reg_lambda

    def _eval(self):

        def mse_err(imageA, imageB):
            err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
            err /= float(imageA.shape[0])
            return err
        
        self._encoded_test = self.encode_samples(self.x_test)
        self._decoded_test = self.decode_samples(self._encoded_test)
        self._both_test = self.autoencoder.predict(self.x_test)

        self._mse_errors = np.array([mse_err(img_a, img_b) for img_a, img_b in zip(self.x_test, self._both_test)])
        self._mse_term= self._get_mse_term(self.x_test, self._both_test).numpy()
        self._sparse_term = self._get_sparse_term(self.x_test, self._both_test).numpy()
        self._losses = self.sparse_loss(self.x_test, self._both_test).numpy()

        # check that losses = (1-lambda) * mse + lambda * sparse_term
        expected_losses = (1 - self.reg_lambda) * self._mse_term + self.reg_lambda * self._sparse_term
        diffs = np.abs(self._losses - expected_losses)
        if np.any(diffs > 1e-6):
            logging.warning("Losses do not match expected values! Differences: %s", diffs[diffs > 1e-6])
        else:
            logging.info("Losses match expected values.")
                
        self._order = np.argsort(self._mse_errors)
        logging.info("Evaluation completed on %i samples:", self._mse_errors.size)
        logging.info("\tMean squared error: %.4f (%.4f)", np.mean(self._mse_errors), np.std(self._mse_errors))
        logging.info("\tMSE term: %.4f", self._mse_term)
        logging.info("\tSparsity term: %.4f", self._sparse_term)
        logging.info("\tMean loss (lambda=%.1f): %.4f (%.4f)", self.reg_lambda, np.mean(self._losses), np.std(self._losses))



    def plot_sparsity(self, n_disp_samples=33, n_stat_samples=1000):
        """
        Encode all statistics samples, calculate reconstruction MSE, and statistics of the encoded vectors


        On the left, two box plots (two axes), one above the other:
            - The reconstruction error (mse), one box per digit class.
            - The sparsity of the encoded vectors, one box per digit class.

        On the right, two mosaics, side-by-side:
            - The original images, one per digit class.
            - The reconstructed (encoded/thresholded/decoded) images, one per digit class.
        """
        x_disp = self.get_test_samples_by_class(n_disp_samples)
        x_stat = self.get_test_samples_by_class(n_stat_samples)

        fig = plt.figure(figsize=(12, 8))
        n_rows, n_cols = 4, 8
        gs = GridSpec(n_rows, n_cols, figure=fig)
        mse_box_ax = fig.add_subplot(gs[0:2, 0:4])
        sparsity_box_ax = fig.add_subplot(gs[2:4, 0:4])
        original_mosaic_ax = fig.add_subplot(gs[0:4, 4:6])
        reconstructed_mosaic_ax = fig.add_subplot(gs[0:4, 6:8])
        """
        # create slider
        ax_slider = fig.add_axes([0.15, 0.01, 0.7, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(ax_slider, 'Threshold', 0.0, 1.0, valinit=thresh, valstep=0.05)

        def update(val):
            thresh = slider.val
            # Update the mosaics and box plots based on the new threshold
            self._update_mosaics(original_mosaic_ax, reconstructed_mosaic_ax, x_test, thresh)
            self._update_box_plots(mse_box_ax, sparsity_box_ax, x_test, thresh)

        slider.on_changed(update)

        sample = self._plot_sparsity(sparsity_box_ax)


        FOR NOW don't use the slider, just plot with a fixed threshold
        """
        thresh = 0.5  # Fixed threshold for now

        z_per_digit = {digit: self.encode_samples(x_stat[digit], binarize_thresh=thresh) for digit in x_stat.keys()}
        decoded_per_digit = {digit: self.decode_samples(z) for digit, z in z_per_digit.items()}
        self._subplot_sparsity(sparsity_box_ax, z_per_digit, thresh)
        self._subplot_mse(mse_box_ax, x_stat, decoded_per_digit, thresh)
        self._subplot_reconstructions(original_mosaic_ax, reconstructed_mosaic_ax, x_disp, thresh)

        filename = "%s_sparsity_stage_%s.png" % (self.get_name(), self._stage)
        self._maybe_save_fig(fig, filename)

        return z_per_digit

    def _subplot_reconstructions(self, original_ax, reconstructed_ax, x_disp, thresh):
        """
        Plot the original and reconstructed digits side by side.
        Left plot is original images, 
        """
        original_imgs = []
        reconstructed_imgs = []
        # get dimensions of the subplot for shaping the mosaics aspect ratio
        bbox = original_ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())

        ax_h, ax_w = bbox.height, bbox.width
        tile_h, tile_w = ax_h/10, ax_w
        aspect = tile_w / tile_h

        decoded_per_digit = {digit: self.decode_samples(self.encode_samples(
            x_disp[digit], binarize_thresh=thresh)) for digit in x_disp.keys()}
        for digit in range(10):
            original_imgs.append(make_digit_mosaic([make_img(img) for img in x_disp[digit]], mosaic_aspect=aspect))
            reconstructed_imgs.append(make_digit_mosaic([make_img(img)
                                      for img in decoded_per_digit[digit]], mosaic_aspect=aspect))

        original_mosaic = np.concatenate(original_imgs, axis=0)
        reconstructed_mosaic = np.concatenate(reconstructed_imgs, axis=0)

        original_ax.imshow(original_mosaic)
        original_ax.set_title("Original Images")
        original_ax.axis('off')

        reconstructed_ax.imshow(reconstructed_mosaic)
        reconstructed_ax.set_title("Reconstructed Images")
        reconstructed_ax.axis('off')

    def _subplot_sparsity(self, ax, encoded_per_digit, thresh):
        """
        Plot the sparsity of the encoded vectors.
        """
        digit_stats = []

        for digit in range(10):
            digit_stats.append(np.sum(encoded_per_digit[digit], axis=1))  # Mean sparsity per sample
        ax.clear()
        ax.set_title("Sparsity: mean active units (of %i)." % (len(encoded_per_digit[0][0]), ))
        sns.boxplot(data=digit_stats, ax=ax)

        # turn off x axis
        # ax.xaxis.set_visible(False)
        # ax.set_xlabel("Sparsity")

    def _subplot_mse(self, ax, truth_per_digit, decoded_per_digit, thresh):
        """
        Plot the reconstruction error (MSE) for each digit class.
        """
        mse_per_digit = []
        for digit in range(10):
            mse = np.mean((truth_per_digit[digit] - decoded_per_digit[digit]) ** 2, axis=1)
            mse_per_digit.append(mse)

        ax.clear()
        ax.set_title("Reconstruction Error (MSE per digit)")
        sns.boxplot(data=mse_per_digit, ax=ax)
        # ax.set_xlabel("MSE")

    def get_test_samples_by_class(self, n_per_digit):
        """
        Get a fixed number of test samples for each digit class.
        """
        x_test = self.x_test
        y_test = self.y_test
        samples = {}
        for digit in range(10):
            digit_samples = x_test[y_test == digit]
            if len(digit_samples) > n_per_digit:
                sample_inds = np.random.choice(digit_samples.shape[0], n_per_digit, replace=False)
                digit_samples = digit_samples[sample_inds]
            samples[digit] = digit_samples
        return samples

    def run_experiment(self):
        pre_trained = self._train()
        if pre_trained:
            self.train_more()   # uncomment to train loaded weights more.
        self._eval()

    def plot_code_samples(self, encoded_digits, n_per_row=100):
        """
        Make an image showing the encoded vectors as rows of pixels.
        The image is as wide as the encoding layer.
        The height is spanned by the 10 digits, each with n_per_row samples.
        Sort the encoded bits by the number of active samples (over all digits), so
        the most frequently used code bits are on the left.
        """
        fig, ax = plt.subplots(figsize=(12, 9))

        # all samples for getting the order and stats/counts:
        code_arr = np.concatenate([encoded_digits[digit] for digit in range(10)], axis=0)
        n_unique_samples = np.unique(code_arr, axis=0).shape[0]
        n_always_on = np.sum(np.sum(code_arr, axis=0) == code_arr.shape[0])
        n_always_off = np.sum(np.sum(code_arr, axis=0) == 0)
        # Sort the encoded bits by the number of active samples (over all digits), so
        # the most frequently used code bits are on the left.
        code_counts = np.sum(code_arr, axis=0)
        code_order = np.argsort(code_counts)[::-1]
        code_size = code_arr.shape[1]
        n_samples = code_arr.shape[0]

        # Pare down to just the samples to display:
        code_arr = np.concatenate([encoded_digits[digit][:n_per_row, code_order] for digit in range(10)], axis=0)
        n_disp_samples = code_arr.shape[0]

        # Plot the image
        n_disp_offset = n_disp_samples//20
        ax.imshow(code_arr, aspect='auto', cmap='gray', interpolation='none',
                  extent = (0, code_size, -n_disp_offset, n_disp_samples-n_disp_offset))
        title = ("Binary Encoding of %i bits, evaluated on %i test samples\n" % (code_size, n_samples)) +\
            ("Units always On: %i, always Off: %i  (of  %i)\nUnique Samples: %i  (of %i)" % (n_always_on,
                                                                                             n_always_off,
                                                                                             code_size,
                                                                                             n_unique_samples,
                                                                                             n_samples))
        ax.set_title(title)
        ax.set_xlabel("Code Bits")
        ax.set_ylabel("Digit")

        # make the y-axis show the digit numbers
        # space evenly, in the middle of each digit's band of samples in the image
        tick_label_positions = np.linspace(0, n_disp_samples - n_disp_offset*2, 10)
        tick_labels = ["%i" % i for i in range(9, -1, -1)]
        
        ax.set_yticks(tick_label_positions)
        ax.set_yticklabels(tick_labels)        
        filename = "%s_codes_%s.png" % (self.get_name(), self._stage)
        self._maybe_save_fig(fig, filename)


def _get_args():
    parser = ArgumentParser(description="Run a sparse autoencoder experiment.")
    parser.add_argument('--layers', type=int, nargs='+', default=[128, 256],
                        help='List of encoding layer sizes (default: [128, 256])')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train each stage (default: 100)')
    parser.add_argument('--stages', type=int, default=5,
                        help='Number of training stages (default: 5)')
    parser.add_argument('--reg_lambda', type=float, default=0.1,
                        help='Regularization lambda for sparsity (default: 0.1)')
    parser.add_argument('--no_plot', action='store_true',
                        help='If set, saves images instead of showing them interactively')
    return parser.parse_args()


def sparse_demo():
    args = _get_args()
    logging.info("Running Sparse Autoencoder with args: %s", args)
    se = SparseExperiment(enc_layers=args.layers, n_epochs=args.epochs,
                          reg_lambda=args.reg_lambda, save_figs=args.no_plot)
    se.run_staged_experiment(n_stages=args.stages)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sparse_demo()
