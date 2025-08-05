import os
from argparse import ArgumentParser
from sys import prefix
from dense import DenseExperiment, diff_img, make_digit_mosaic
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


class SparseEvaluation(object):
    def __init__(self, experiment, x_test_per_digit):
        """
        Initialize the SparseEvaluation with the experiment and test samples.
        :param experiment: The SparseExperiment instance.
        :param x_test_per_digit: A dictionary of test samples per digit class, 
           key is the digit d, value is an N_d x D array of N_d test samples.
        :param n_display_samples: The number of samples to display for each digit.
        """
        self.experiment = experiment
        self.x_test_per_digit = x_test_per_digit
        self._eval_test_set()

    def _eval_test_set(self):

        self.code_thresh = self.experiment.code_thresh
        self.encoded_test_by_digit = {digit: self.experiment.encode_samples(self.x_test_per_digit[digit],
                                                                            binarize_thresh=self.code_thresh)
                                      for digit in self.x_test_per_digit.keys()}
        self.decoded_test_by_digit = {digit: self.experiment.decode_samples(self.encoded_test_by_digit[digit])
                                      for digit in self.encoded_test_by_digit.keys()}

        self.encoded_mat = np.concatenate([self.encoded_test_by_digit[d] for d in range(10)], axis=0)
        self.decoded_mat = np.concatenate([self.decoded_test_by_digit[d] for d in range(10)], axis=0)
        self.x_test_mat = np.concatenate([self.x_test_per_digit[d] for d in range(10)], axis=0)

        logging.info("Evaluating sparse autoencoder on %i test samples with threshold %s",
                     self.x_test_mat.shape[0], ("%.2f" % self.code_thresh) if self.code_thresh is not None else '(not binarized)')
        self.n_unique_samples = np.unique(self.encoded_mat, axis=0).shape[0]
        self.n_codebits_always_on = np.sum(np.sum(self.encoded_mat, axis=0) == self.encoded_mat.shape[0])
        self.n_codebits_always_off = np.sum(np.sum(self.encoded_mat, axis=0) == 0)
        self.n_codebits_used = self.encoded_mat.shape[1] - (self.n_codebits_always_on + self.n_codebits_always_off)

        self.mse_errors = np.array([self.experiment.get_mse_terms(self.x_test_mat, self.decoded_mat).numpy()]).reshape(-1)
        self.sparse_errors = np.array([self.experiment.get_sparse_terms(self.x_test_mat, self.decoded_mat).numpy()]).reshape(-1)
        self.sparse_term = np.mean(self.sparse_errors)
        self.mse_term = np.mean(self.mse_errors)
        self.loss_terms = (1-self.experiment.reg_lambda) * self.mse_errors + \
            self.experiment.reg_lambda * self.sparse_errors

        self.order = np.argsort(self.loss_terms).reshape(-1)

        logging.info("Evaluation completed on %i samples, BINARIZING the code layer:", self.mse_errors.size)
        logging.info("\tMean squared error: %.4f (%.4f)", np.mean(self.mse_errors), np.std(self.mse_errors))
        logging.info("\tMSE term: %.4f", self.mse_term)
        logging.info("\tSparsity term: %.4f", self.sparse_term)
        logging.info("\tCombined loss (w/lambda=%.4f): %.4f ", self.experiment.reg_lambda, np.mean(self.loss_terms))
        logging.info("\tUnique encoded samples (of %i): %i", self.n_unique_samples, self.encoded_mat.shape[0])
        logging.info("\tCodes unit utilization (%i total):\n\t\t\tused: %i",
                     self.encoded_mat.shape[1], self.n_codebits_used)
        logging.info("\t\talways On: %i,\n\t\t\talways Off: %i", self.n_codebits_always_on, self.n_codebits_always_off)


class SparseExperiment(DenseExperiment):

    _DEFAULT_ACT_FNS = {
        'encoding': 'sigmoid',
        'binarize_code_units': False,
        'internal': 'relu',
    }

    def __init__(self, enc_layers=(64,), n_epochs=25, act_fns=None, reg_lambda=0.5, save_figs=True,
                 reg_method='l1', binarize_code_units=None):
        activation_functions = self._DEFAULT_ACT_FNS.copy() if act_fns is None else act_fns
        self.reg_lambda = reg_lambda
        self.reg_method = reg_method
        if binarize_code_units is not None:
            activation_functions['binarize_code_units'] = binarize_code_units

        super().__init__(enc_layers=enc_layers, n_epochs=n_epochs, act_fns=activation_functions, save_figs=save_figs)

        self.code_thresh = {'sigmoid': 0.5,
                           'relu': 0.0}[self.act_fns['encoding']]

        if self.act_fns['binarize_code_units'] and self.act_fns['encoding'] != 'relu':
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
        if not self._attempt_resume():
            logging.info("Training 1 epoch to show loss function terms.")
            self.train_more(n_epochs=1, save_wts=False)
        logging.info("***************************")
        logging.info("Starting %i stages of training %i epochs each.", n_stages, self._n_epochs)
        for stage in range(n_stages):
            self._stage = stage
            logging.info("Running stage %i of %i", stage + 1, n_stages)
            result = self.train_more()
            self._plot_history()
            self.plot_distributions(result, show_diffs=False)
            self.plot_distributions(result, show_diffs=True)
            self.plot_sparsity(result)
            self.plot_code_samples(result)
            if not self._save_figs:
                plt.tight_layout()
                plt.show()

    def plot_distributions(self, result, show_diffs=False, n_samp=36):
        prefix = self.get_name(file_ext=False)
        suffix = "stage_%i" % (self._stage+1)
        fig = plt.figure(constrained_layout=True, figsize=(10, 8))
        # show best, worst, and middle 4 quantiles above a histogram
        n_quantiles = 8
        n_q_rows = n_quantiles//2
        n_q_cols = n_quantiles // n_q_rows
        gs = GridSpec(n_q_rows+1, n_q_cols, figure=fig)  # add a row for the loss histogram
        q_axes = []
        for j in range(n_q_cols):
            for i in range(n_q_rows):
                ax = fig.add_subplot(gs[i, j])
                q_axes.append(ax)
        loss_hist_axis = fig.add_subplot(gs[n_q_rows, :])
        order = result.order

        def show_mosaic(ax, inds, title, color, aspect=4.0):
            """
            Show reconstructed digits
            :param ax: The axis to plot on.
            :param inds: Indices of the samples to show.
            :param title: Title for the plot.
            :param color: Color for the title text.
            """
            if not show_diffs:
                reconstructed_imgs = [make_img(result.decoded_mat[i]) for i in inds]
                image = make_digit_mosaic(reconstructed_imgs, mosaic_aspect=aspect)
            else:
                diff_imgs = [diff_img(result.x_test_mat[i], result.decoded_mat[i]) for i in inds]
                image = make_digit_mosaic(diff_imgs, mosaic_aspect=aspect)

            ax.imshow(image)
            ax.set_title(title, fontsize=12, color=color)
            ax.axis('off')

        # show a mosaic of a few decoded/encoded (or diff) images around each quantile
        best_inds = order[:n_samp]
        worst_inds = order[-n_samp:]
        quantile_sample_indices = np.linspace(0, len(order)-1, n_quantiles, dtype=int)[1:-1]
        quant_inds = [best_inds]
        for ind in quantile_sample_indices:
            extra = n_samp % 2
            quant_inds.append(order[ind-n_samp//2:ind+n_samp//2+extra])
        quant_inds.append(worst_inds)
        mid_labels = ["sample group %i - %s" % (i, suffix) for i in range(1, n_quantiles-1)]
        q_labels = ['Lowest Test MSE - %s' % suffix] + mid_labels + ['Highest Test MSE - %s' % suffix]
        n_colors = len(q_labels)
        cmap = plt.get_cmap('brg', n_colors)
        colors = [cmap(i) for i in range(n_colors)]
        for i, (inds, label) in enumerate(zip(quant_inds, q_labels)):
            show_mosaic(q_axes[i], inds, label, color=colors[i])
        model_name = self.get_name()
        title = "Autoencoder Model: %s (stage %i)" % (model_name, self._stage + 1) +\
            "\nData: n_train=%i, n_test = %i " % (self.x_train.shape[0], self.x_test.shape[0])
        if show_diffs:
            title += "          RED: decoded pixel >= 10% too high,"
        title += "\nResults: test MSE = %.4f (%.4f)" % (np.mean(result.mse_errors), np.std(result.mse_errors))
        if show_diffs:
            title += "             BLUE: decoded pixel >= 10% too low."
        plt.suptitle(title, fontsize=14)

        # Now show the histograms for loss and the quantile locations within it.
        # then plot histograms for mse and regularization terms, showing the span of samples in each quantile (quant_inds)
        # i.e. find the x-positions of the lowest and highest mse for a sample group and put a colord band over that portion of the MSE histogram.
        #  THen do the same for the regularization term histogram.


        loss_title = "Test loss dist. (%i samp.), w/sample groups at %i quantiles" % (result.loss_terms.size, n_quantiles)

        self._show_err_hist(loss_hist_axis, result.loss_terms, band_colors=colors, quantile_band_inds=quant_inds, title=loss_title)
        filename = "%s_%s_%s.png" % (prefix, ("diffs" if show_diffs else "reconstructed"), suffix)
        self._maybe_save_fig(fig, filename)

    def _show_err_hist(self, ax, errors, band_colors, quantile_band_inds, title):
        """
        Show a histogram of the errors, with vertical bands for the quantile sample groups.
        :param ax: The axis to plot on.
        :param errors: The error values to plot.
        :param band_colors: List of colors for the quantile bands.
        :param quantile_band_inds: List of indices for the quantile sample groups.
        """
        ax.hist(errors, bins=100, color='gray', alpha=0.8)
        ax.set_title(title, fontsize=12)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        def draw_band(ax, band_index, color):
            sample_inds = quantile_band_inds[band_index]
            left, right = np.min(errors[sample_inds]), np.max(errors[sample_inds])
            ax.axvspan(left, right, color=color, alpha=0.5)
        if quantile_band_inds is not None:
            for i in range(len(band_colors)):
                draw_band(ax, i, band_colors[i])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def _init_encoder_layers(self, inputs):
        encoding_layers = super()._init_encoder_layers(inputs)
        encoding = encoding_layers[-1]
        if self.act_fns['binarize_code_units']:
            encoding = BinaryActivation(name='binarize')(encoding)
        encoding_layers.append(encoding)
        return encoding_layers

    def get_name(self, file_ext=False):
        def func_to_str(func):
            if hasattr(func, '__name__'):
                return func.__name__
            return str(func)

        desc_str = "_".join([str(n) for n in self.enc_layer_desc])
        bin_str = "_REAL" if not self.act_fns['binarize_code_units'] else "_BINARY"
        reg_str = "reg=%s" % self.reg_method if self.reg_method else "unregularized"
        fname = ("Sparse(%s%s_%s_regL=%.3f)" %
                 (desc_str, bin_str, reg_str, self.reg_lambda))
        if file_ext:
            fname += ".weights.h5"
        return fname

    def get_mse_terms(self, x_true, x_pred):
        """
        Calculate the mean squared error between the true and predicted images.
        """
        mse_loss = tf.keras.losses.MeanSquaredError(reduction='none')(x_true, x_pred)
        return mse_loss

    def get_sparse_terms(self, x_true, x_pred):
        """
        Calculate the sparsity term based on the encoding of the input images.
        """
        z = self.encoder(x_true)
        if self.reg_method.startswith('entropy'):
            # Minimize entropy
            binary_reg_term = tf.reduce_mean(z * (1 - z), axis=1)
            if self.reg_method == 'entropy_sq':
                # Use the square of the entropy term, for even stronger pressure to be 1 or 0.
                binary_reg_term = tf.square(binary_reg_term)
        elif self.reg_method == 'l1':
            # Use the L1 norm of the encoded vector, normalized by the number of samples (mean active feature count)
            binary_reg_term = tf.reduce_mean(z, axis=1)
        elif self.reg_method is None:
            return tf.zeros(shape=(x_true.shape[0],), dtype=tf.float32)
        else:
            raise ValueError("Unknown method for sparse loss: %s" % self.reg_method)

        return binary_reg_term

    def sparse_loss(self, x_true, x_pred):
        mse_loss = tf.reduce_mean(self.get_mse_terms(x_true, x_pred))
        binary_reg_term = tf.reduce_mean(self.get_sparse_terms(x_true, x_pred))
        return mse_loss * (1 - self.reg_lambda) + binary_reg_term * self.reg_lambda

    def _get_loss_fn(self):
        return self.sparse_loss

    def _eval(self):
        """
        update current stats, on a per-digit basis.
        This is called after trainingon CONTINUOUS (if the option is used), to evaluate the
          model on the test set with BINARIZED codes.
        """
        test_samples_per_digit = self.get_test_samples_by_class(1000)
        self._stage_result = SparseEvaluation(self, test_samples_per_digit)
        self._order = self._stage_result.order
        return self._stage_result

    def plot_sparsity(self, result):
        """
        Encode all statistics samples, calculate reconstruction MSE, and statistics of the encoded vectors


        On the left, two box plots (two axes), one above the other:
            - The reconstruction error (mse), one box per digit class.
            - The sparsity of the encoded vectors, one box per digit class.

        On the right, two mosaics, side-by-side:
            - The original images, one per digit class.
            - The reconstructed (encoded/thresholded/decoded) images, one per digit class.
        """
        fig = plt.figure(figsize=(12, 8))
        n_rows, n_cols = 4, 8

        gs = GridSpec(n_rows, n_cols, figure=fig)
        mse_box_ax = fig.add_subplot(gs[0:2, 0:4])
        sparsity_box_ax = fig.add_subplot(gs[2:4, 0:4])
        original_mosaic_ax = fig.add_subplot(gs[0:4, 4:6])
        reconstructed_mosaic_ax = fig.add_subplot(gs[0:4, 6:8])
        self._subplot_sparsity(sparsity_box_ax, result)
        self._subplot_mse(mse_box_ax, result)
        self._subplot_reconstructions(original_mosaic_ax, reconstructed_mosaic_ax, result)

        filename = "%s_sparsity_stage_%s.png" % (self.get_name(), self._stage)
        self._maybe_save_fig(fig, filename)
        fig.tight_layout()

    def _subplot_reconstructions(self, original_ax, reconstructed_ax, result, n_examples_per_digit=33):
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

        # sample the result decoded images
        disp_inds_pd = {digit: np.random.choice(result.encoded_test_by_digit[digit].shape[0],
                                                n_examples_per_digit, replace=False) for digit in range(10)}
        x_disp_pd = {digit: result.x_test_per_digit[digit][disp_inds_pd[digit]] for digit in range(10)}
        decoded_pd = {digit: result.decoded_test_by_digit[digit][disp_inds_pd[digit]] for digit in range(10)}

        for digit in range(10):
            original_imgs.append(make_digit_mosaic([make_img(img) for img in x_disp_pd[digit]], mosaic_aspect=aspect))
            reconstructed_imgs.append(make_digit_mosaic([make_img(img)
                                      for img in decoded_pd[digit]], mosaic_aspect=aspect))

        original_mosaic = np.concatenate(original_imgs, axis=0)
        reconstructed_mosaic = np.concatenate(reconstructed_imgs, axis=0)

        original_ax.imshow(original_mosaic)
        original_ax.set_title("Original Images")
        original_ax.axis('off')

        reconstructed_ax.imshow(reconstructed_mosaic)
        reconstructed_ax.set_title("Reconstructed Images")
        reconstructed_ax.axis('off')

    def _subplot_sparsity(self, ax, result):
        """
        Plot the sparsity of the encoded vectors.
        """ 
        digit_stats = []

        for digit in range(10):
            digit_stats.append(np.sum(result.encoded_test_by_digit[digit], axis=1) -
                               result.n_codebits_always_on)  # Mean sparsity per sample
        ax.clear()

        ax.set_title("Sparsity (code_units=%i, non-constant=%i), bits to encode each digit:" %
                     (len(result.encoded_test_by_digit[0][0]), result.n_codebits_used), fontsize=11)
        sns.boxplot(data=digit_stats, ax=ax)

        # turn off x axis
        # ax.xaxis.set_visible(False)
        # ax.set_xlabel("Sparsity")

    def _subplot_mse(self, ax, result):
        """
        Plot the reconstruction error (MSE) for each digit class.
        """
        truth_per_digit = result.x_test_per_digit
        decoded_per_digit = result.decoded_test_by_digit
        mse_per_digit = []
        n_test_samples = []
        for digit in range(10):
            mse = np.mean((truth_per_digit[digit] - decoded_per_digit[digit]) ** 2, axis=1)
            mse_per_digit.append(mse)
            n_test_samples.append(mse.size)


        ax.clear()
        ax.set_title("Reconstruction Error (MSE per digit), N = %i" % int(np.mean(n_test_samples)))
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

    def plot_code_samples(self, result, n_per_row=100):
        """
        Make an image showing the encoded vectors as rows of pixels.
        The image is as wide as the encoding layer.
        The height is spanned by the 10 digits, each with n_per_row samples.
        Sort the encoded bits by the number of active samples (over all digits), so
        the most frequently used code bits are on the left.
        """
        fig, ax = plt.subplots(figsize=(12, 9))

        # all samples for getting the order and stats/counts:
        code_arr = np.concatenate([result.encoded_test_by_digit[digit] for digit in range(10)], axis=0)
        # Sort the encoded bits by the number of active samples (over all digits), so
        # the most frequently used code bits are on the left.
        code_counts = np.sum(code_arr, axis=0)
        code_order = np.argsort(code_counts)[::-1]
        code_size = code_arr.shape[1]
        n_samples = code_arr.shape[0]

        # Pare down to just the samples to display:
        code_arr = np.concatenate([result.encoded_test_by_digit[digit][:n_per_row, code_order] for digit in range(10)], axis=0)
        n_disp_samples = code_arr.shape[0]

        # Plot the image
        n_disp_offset = n_disp_samples//20
        ax.imshow(code_arr, aspect='auto', cmap='gray', interpolation='none',
                  extent=(0, code_size, -n_disp_offset, n_disp_samples-n_disp_offset))
        title = ("Binary Encoding of %i bits, evaluated on %i test samples\n" % (code_size, n_samples)) +\
            ("Units used: %i, always On: %i, always Off: %i  (of  %i)\nUnique Samples: %i  (of %i)" % (result.n_codebits_used,
                                                                                                       result.n_codebits_always_on,
                                                                                                       result.n_codebits_always_off,
                                                                                                       code_size,
                                                                                                       result.n_unique_samples,
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
    reg_methods = ['entropy', 'entropy_sq', 'l1', 'None']
    parser.add_argument('--reg_method', type=str, choices=reg_methods, default='entropy',
                        help='Method for calculating the sparsity (regularization) term (default: entropy)' +
                        " valid options: %s" % ', '.join(reg_methods))
    parser.add_argument('--real_code_activations', action='store_true',
                        help='If set, the encoding layer will use real-valued activations instead of thresholding (Heavisidew/pseudoderivative) units.  (default: False)')
    parsed = parser.parse_args()

    if not (0 <= parsed.reg_lambda <= 1):
        parser.error("Invalid value for --reg_lambda: %f (must be between 0 and 1)" % parsed.reg_lambda)
    return parsed


def sparse_demo():
    args = _get_args()
    logging.info("Running Sparse Autoencoder with args: %s", args)
    reg_method = args.reg_method if args.reg_method != 'None' else None
    se = SparseExperiment(enc_layers=args.layers, n_epochs=args.epochs,
                          reg_lambda=args.reg_lambda, save_figs=args.no_plot,
                          reg_method=reg_method, binarize_code_units=not args.real_code_activations)
    se.run_staged_experiment(n_stages=args.stages)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sparse_demo()
