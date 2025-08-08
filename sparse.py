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
    y = tf.cast(x > 0.5, tf.float32)

    def grad(dy):
        return dy
    return y, grad


class BinaryActivation(tf.keras.layers.Layer):
    def call(self, inputs):
        return binary_step_with_straight_through_estimator(inputs)


class SparseEvaluation(object):
    def __init__(self, experiment):
        """
        Initialize the SparseEvaluation with the experiment and test samples.
        :param experiment: The SparseExperiment instance.
                   key is the digit d, value is an N_d x D array of N_d test samples.
        :param n_display_samples: The number of samples to display for each digit.
        """
        self.experiment = experiment
        self._eval_test_set()

    def _eval_test_set(self):
        x_test, y_test = self.experiment.x_test, self.experiment.y_test
        x_train, y_train = self.experiment.x_train, self.experiment.y_train

        self.code_thresh = self.experiment.code_thresh

        thresh_str = ("%.2f" % self.code_thresh) if self.code_thresh is not None else '(not binarized)'

        logging.info("Encoding/decoding test set (%i samples) with threshold %s", x_test.shape[0], thresh_str)
        self.encoded_test_mat = self.experiment.encode_samples(x_test, binarize=True)
        self.decoded_test_mat = self.experiment.decode_samples(self.encoded_test_mat)

        logging.info("Encoding/decoding training set (%i samples) with threshold %s", x_train.shape[0], thresh_str)
        self.encoded_train_mat = self.experiment.encode_samples(x_train, binarize=True)
        self.decoded_train_mat = self.experiment.decode_samples(self.encoded_train_mat)

        # sorting test data by digit is helpful for plotting
        self.encoded_test_by_digit = {d: self.encoded_test_mat[y_test == d]
                                      for d in range(10)}
        self.decoded_test_by_digit = {d: self.decoded_test_mat[y_test == d]
                                      for d in range(10)}

        self.x_test_by_digit = {d: x_test[y_test == d]
                                for d in range(10)}

        def _get_terms_stats_and_losses(true_samples, encoded_samples, decoded_samples):
            logging.info("Evaluating sparse autoencoder on %i samples with threshold %s",
                         true_samples.shape[0], thresh_str)

            mse_loss_terms = np.array([self.experiment.get_mse_terms(
                true_samples, decoded_samples).numpy()]).reshape(-1)
            reg_loss_terms = np.array([self.experiment.get_regularization_terms(
                true_samples, decoded_samples).numpy()]).reshape(-1) if self.experiment.reg_method != 'none' else np.zeros_like(mse_loss_terms)
            loss_terms = (1-self.experiment.reg_lambda) * mse_loss_terms + \
                self.experiment.reg_lambda * reg_loss_terms
            n_unique_samples = np.unique(encoded_samples, axis=0).shape[0]
            n_codebits_always_on = np.sum(np.sum(encoded_samples, axis=0) == encoded_samples.shape[0])
            n_codebits_always_off = np.sum(np.sum(encoded_samples, axis=0) == 0)
            n_codebits_used = encoded_samples.shape[1] - (n_codebits_always_on + n_codebits_always_off)

            return {'n_samples': true_samples.shape[0],
                    'mse_errors': mse_loss_terms,
                    'reg_errors': reg_loss_terms,
                    'losses': loss_terms,
                    'reg_term': np.mean(reg_loss_terms),
                    'mse_term': np.mean(mse_loss_terms),
                    'loss': np.mean(loss_terms),
                    'n_unique_samples': n_unique_samples,
                    'n_codebits_always_on': n_codebits_always_on,
                    'n_codebits_always_off': n_codebits_always_off,
                    'n_codebits_used': n_codebits_used}

        self.test_errs = _get_terms_stats_and_losses(x_test, self.encoded_test_mat, self.decoded_test_mat)
        self.train_errs = _get_terms_stats_and_losses(x_train, self.encoded_train_mat, self.decoded_train_mat)

        self.order = np.argsort(self.test_errs['losses']).reshape(-1)

        def _log_eval(errs, kind):
            code_size = self.encoded_test_mat.shape[1]
            # TODO: Add regularization breakdown

            logging.info("\n\n\n%s evaluation completed on %i samples, BINARIZING the code layer:",
                         kind, errs['n_samples'])
            logging.info("\tMSE term: %.4f", errs['mse_term'], )
            logging.info("\tSparsity term: %.4f", errs['reg_term'])
            logging.info("\tCombined loss (w/lambda=%.4f): %.4f ",
                         self.experiment.reg_lambda, errs['loss'])
            logging.info("\tUnique encoded samples (of %i): %i", errs['n_samples'], errs['n_unique_samples'])
            logging.info("\tCodes unit utilization (%i total):\n\t\t\tused: %i",
                         code_size, self.test_errs['n_codebits_used'])
            logging.info("\t\talways On: %i,\n\t\t\talways Off: %i",
                         self.test_errs['n_codebits_always_on'], self.test_errs['n_codebits_always_off'])

        _log_eval(self.test_errs, "Test set")
        _log_eval(self.train_errs, "Training set")


class SparseExperiment(DenseExperiment):

    _DEFAULT_ACT_FNS = {
        # 'encoding' set in __init__
        'binarize_code_units': False,
        'internal': 'relu',
    }

    def __init__(self, enc_layers=(64,), act_fns=None, reg_lambda=0.5, 
                 reg_method='none', binarize_code_units=None):
        activation_functions = self._DEFAULT_ACT_FNS.copy() if act_fns is None else act_fns
        self.reg_lambda = reg_lambda
        self.reg_method = reg_method

        if binarize_code_units is not None:
            activation_functions['binarize_code_units'] = binarize_code_units

        activation_functions['encoding'] = 'sigmoid'
        self.code_thresh = {'sigmoid': 0.5,
                            'tanh': 0.0}[activation_functions['encoding']]
        super().__init__(enc_layers=enc_layers, act_fns=activation_functions)


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

    def run_staged_experiment(self, n_stages=5, n_epochs=25, save_figs=False):
        self._save_figs = save_figs

        if not self._attempt_resume():
            logging.info("Training 1 epoch to show loss function terms.")
            self.train_more(n_epochs=1, save_wts=False)
        logging.info("***************************")
        logging.info("Starting %i stages of training %i epochs each.", n_stages, n_epochs)
        for stage in range(n_stages):
            self._stage = stage
            logging.info("Running stage %i of %i", stage + 1, n_stages)
            result = self.train_more(n_epochs=n_epochs, save_wts=True)
            self._plot_history()
            self.plot_distributions(result, show_diffs=False)
            self.plot_distributions(result, show_diffs=True)
            self.plot_sparsity(result)
            self.plot_code_samples(result)
            if not self._save_figs:
                plt.tight_layout()
                plt.show()

    def plot_distributions(self, result, show_diffs=False, n_samp=36):
        prefix = self.get_name()
        suffix = "stage_%i" % (self._stage,)
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
                reconstructed_imgs = [make_img(result.decoded_test_mat[i]) for i in inds]
                image = make_digit_mosaic(reconstructed_imgs, mosaic_aspect=aspect)
            else:
                diff_imgs = [diff_img(self.x_test[i], result.decoded_test_mat[i]) for i in inds]
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
        title += "\nResults: test MSE = %.4f (%.4f)" % (
            np.mean(result.test_errs['mse_errors']), np.std(result.test_errs['mse_errors']))
        if show_diffs:
            title += "             BLUE: decoded pixel >= 10% too low."
        plt.suptitle(title, fontsize=14)

        # Now show the histograms for loss and the quantile locations within it.
        # then plot histograms for mse and regularization terms, showing the span of samples in each quantile (quant_inds)
        # i.e. find the x-positions of the lowest and highest mse for a sample group and put a colord band over that portion of the MSE histogram.
        #  THen do the same for the regularization term histogram.

        loss_title = "Test loss dist. (%i samp.), w/sample groups at %i quantiles" % (
            result.test_errs['losses'].size, n_quantiles)

        self._show_err_hist(loss_hist_axis, result.test_errs['losses'], band_colors=colors,
                            quantile_band_inds=quant_inds, title=loss_title)
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

    def get_name(self, file_ext=None):
        def func_to_str(func):
            if hasattr(func, '__name__'):
                return func.__name__
            return str(func)

        desc_str = "_".join([str(n) for n in self.enc_layer_desc])
        bin_str = "_REAL" if not self.act_fns['binarize_code_units'] else "_BINARY"
        reg_str = "reg=%s" % self.reg_method
        fname = ("Sparse(%s%s_%s_regL=%.3f)" %
                 (desc_str, bin_str, reg_str, self.reg_lambda))
        if file_ext=='weights':
            fname += ".weights.h5"
        elif file_ext=='history':
            fname += ".history.json"
        elif file_ext is not None:
            raise ValueError("Unknown file extension: %s" % file_ext)
        return fname

    def get_mse_terms(self, x_true, x_pred):
        """
        Calculate the mean squared error between the true and predicted images.
        """
        mse_loss = tf.keras.losses.MeanSquaredError(reduction='none')(x_true, x_pred)
        return mse_loss

    def get_regularization_terms(self, x_true, x_pred):

        # Sparsity/binary component:
        if self.reg_method == 'entropy':
            # Minimize entropy
            z = self.pre_binarized_encoder(x_true) if self.act_fns['binarize_code_units'] else self.encoder(x_true)
            binary_reg_terms = tf.reduce_mean(z * (1 - z), axis=1)

        elif self.reg_method == 'L1':
            # Use the L1 norm of the encoded vector, normalized by the number of samples (mean active feature count)
            z = self.encoder(x_true)
            binary_reg_terms = tf.reduce_mean(z, axis=1)

        else:
            raise ValueError("Unknown method for sparse loss: %s" % self.reg_method)

        return binary_reg_terms

    def sparse_loss(self, x_true, x_pred):
        mse_loss = tf.reduce_mean(self.get_mse_terms(x_true, x_pred))
        if self.reg_method != 'none':
            binary_reg_term = tf.reduce_mean(self.get_regularization_terms(x_true, x_pred))
        else:
            binary_reg_term = 0.0

        return mse_loss * (1 - self.reg_lambda) + binary_reg_term * self.reg_lambda

    def _get_loss_fn(self):
        return self.sparse_loss
    
    def encode_samples(self, samples, binarize=True):
        """
        Encode samples into the latent space, optionally binarizing the output.
        :param samples: array of shape (n_samples, d_input)
        :param binarize: threshold for binarization, if None, no binarization is applied.
        :return: array of shape (n_samples, d_latent)
        """
        encoded = self.encoder(samples).numpy()
        if binarize is not None:
            encoded = (encoded > self.code_thresh)
        return encoded
    
    def decode_samples(self, codes):
        """
        Decode samples from the latent space.
        :param codes: array of shape (n_samples, d_latent)
        :return: array of shape (n_samples, d_input)
        """
        decoded = self.decoder(codes).numpy()
        return decoded

    def _eval(self):
        """
        update current stats, on a per-digit basis.
        This is called after trainingon CONTINUOUS (if the option is used), to evaluate the
          model on the test set with BINARIZED codes.
        """
        self._stage_result = SparseEvaluation(self)
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
        x_disp_pd = {digit: result.x_test_by_digit[digit][disp_inds_pd[digit]] for digit in range(10)}
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
                               result.test_errs['n_codebits_always_on'])  # Mean sparsity per sample
        ax.clear()
        code_size = result.encoded_test_by_digit[0].shape[1]
        ax.set_title("Sparsity (code_units=%i, non-constant=%i), bits to encode each digit:" %
                     (code_size, result.test_errs['n_codebits_used']), fontsize=11)
        sns.boxplot(data=digit_stats, ax=ax)
        ax.set_ylim(0, result.test_errs['n_codebits_used'] + 1)

        # turn off x axis
        # ax.xaxis.set_visible(False)
        # ax.set_xlabel("Sparsity")

    def _subplot_mse(self, ax, result):
        """
        Plot the reconstruction error (MSE) for each digit class.
        """
        truth_per_digit = result.x_test_by_digit
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

    def get_test_samples_by_class(self, n_per_digit=None):
        """
        Get a fixed number of test samples for each digit class.
        """
        x_test = self.x_test
        y_test = self.y_test
        samples = {}
        for digit in range(10):
            digit_samples = x_test[y_test == digit]
            if n_per_digit is not None and len(digit_samples) > n_per_digit:
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

        # Pare down to just the samples to display:
        code_arr = np.concatenate([result.encoded_test_by_digit[digit][:n_per_row, code_order]
                                  for digit in range(10)], axis=0)
        n_display_samples = n_per_row * 10
        n_samples = code_arr.shape[0]
        # Plot the image
        n_disp_offset = n_display_samples//20
        ax.imshow(code_arr, aspect='auto', cmap='gray', interpolation='none',
                  extent=(0, code_size, -n_disp_offset, n_display_samples-n_disp_offset))
        title = ("Binary Encoding of %i bits, evaluated on %i test samples\n" % (code_size, n_samples)) +\
            ("Units used: %i, always On: %i, always Off: %i  (of  %i)\nUnique Samples: %i  (of %i)" % (result.test_errs['n_codebits_used'],
                                                                                                       result.test_errs['n_codebits_always_on'],
                                                                                                       result.test_errs['n_codebits_always_off'],
                                                                                                       code_size,
                                                                                                       result.test_errs['n_unique_samples'],
                                                                                                       n_samples))
        ax.set_title(title)
        ax.set_xlabel("Code Bits")
        ax.set_ylabel("Digit")

        # make the y-axis show the digit numbers
        # space evenly, in the middle of each digit's band of samples in the image
        tick_label_positions = np.linspace(0, n_display_samples - n_disp_offset*2, 10)
        tick_labels = ["%i" % i for i in range(9, -1, -1)]

        ax.set_yticks(tick_label_positions)
        ax.set_yticklabels(tick_labels)
        filename = "%s_codes_%s.png" % (self.get_name(), self._stage)
        self._maybe_save_fig(fig, filename)

    @staticmethod
    def get_args():
        description = "Run a sparse autoencoder on MNIST digits"
        reg_methods = ['none', 'entropy', 'L1']
        extra_args = [dict(name='--reg_lambda', type=float, default=0.1,
                           help='Regularization lambda for sparsity (default: 0.1)'),
                      dict(name='--reg_method', type=str, choices=reg_methods, default='none',
                           help='Method for calculating the binary/sparsity (regularization) term (default: none)' +
                           " valid options: %s," % ', '.join(reg_methods) +
                           " NOTE: Using the ENTROPY method calculates the regularization term on the pre-binarized layer of the encoder (layer[-2])" +
                           " if using binary codes, or encoder outputs directly if using real-valued codes (the final layer, layer[-1])"),
                      dict(name='--real_code_activations', action='store_true',
                           help='Codes (output of encoder layer) are real-valued vectors instead of binary.' +
                           'NOTE: this removes the Heaviside pass-through units from the end of the encoder ')]
        parsed = super(SparseExperiment, SparseExperiment).get_args(description=description, extra_args=extra_args)

        def check_lambda(reg_lambda, which):
            if not (0 <= reg_lambda <= 1):
                raise ValueError("Invalid value for --%s: %f (must be between 0 and 1)" % (which, reg_lambda))

        parsed.reg_method = parsed.reg_method.lower()

        if parsed.reg_method != 'none':
            check_lambda(parsed.reg_lambda, "reg_lambda")
        else:
            parsed.reg_lambda = 0.0

        return parsed
    



def sparse_demo():
    args = SparseExperiment.get_args()
    logging.info("Running Sparse Autoencoder with args: %s", args)
    se = SparseExperiment(enc_layers=args.layers,
                           reg_lambda=args.reg_lambda,
                             reg_method=args.reg_method,
                               binarize_code_units=not args.real_code_activations)
    se.run_staged_experiment(n_stages=args.stages, save_figs=args.no_plot, n_epochs=args.epochs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sparse_demo()
