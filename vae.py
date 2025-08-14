import pickle
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import re
from matplotlib.gridspec import GridSpec
from colors import COLORS
from img_util import make_digit_mosaic, make_img, diff_img
from latent_var_plots import LatentDigitDist
import time
from experiment import AutoencoderExperiment
WORKING_DIR = "VAE-results"

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout_layer=None, dropout_rate=0.5):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.hidden_layers = nn.ModuleList()
        self.dropouts = nn.ModuleDict()  # store optional dropout layers
        last_dim = input_dim

        for i, layer_size in enumerate(hidden_dims):
            self.hidden_layers.append(nn.Linear(last_dim, layer_size))
            if dropout_layer is not None and i == dropout_layer:
                self.dropouts[str(i)] = nn.Dropout(p=dropout_rate)
            last_dim = layer_size

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers):
            x = self.relu(layer(x))
            if str(i) in self.dropouts:
                x = self.dropouts[str(i)](x)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var



class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.hidden_layers = nn.ModuleList()
        last_dim = latent_dim
        for layer_size in hidden_dims:
            self.hidden_layers.append(nn.Linear(last_dim, layer_size))
            last_dim = layer_size
        self.fc3 = nn.Linear(last_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        for layer in self.hidden_layers:
            z = self.relu(layer(z))
        recon_x = self.sigmoid(self.fc3(z))
        return recon_x


class VAE(nn.Module):
    def __init__(self, input_dim, enc_hidden_dims, dec_hidden_dims, latent_dim, output_dim, device, dropout_info=None, lambda_reg=0.001):
        super().__init__()
        d_l, d_r = (dropout_info['layer'], dropout_info['rate']) if dropout_info is not None else (None, None)
        self.encoder = Encoder(input_dim, enc_hidden_dims, latent_dim,dropout_layer=d_l, dropout_rate=d_r)
        self.decoder = Decoder(latent_dim, dec_hidden_dims, output_dim)
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.lambda_reg = lambda_reg
        self.device = device

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z)
        return recon_x, mu, log_var

    def loss_function(self, recon_y, y_batch, mu, log_var, return_terms=False):
        MSE = F.mse_loss(recon_y, y_batch, reduction='mean')
        KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = (1-self.lambda_reg) * MSE + (self.lambda_reg) * KLD
        if return_terms:
            return total_loss, MSE, KLD
        return total_loss

    def predict(self, x):
        self.eval()
        x_tensor = torch.from_numpy(x).to(self.device).float()
        x_flat = x_tensor.view(-1, self.input_dim)
        with torch.no_grad():
            recon_x, _, _ = self(x_flat)
        return recon_x.cpu().numpy()

    def encode(self, x):
        self.eval()
        x_tensor = torch.from_numpy(x).to(self.device).float()
        x_flat = x_tensor.view(-1, self.input_dim)
        with torch.no_grad():
            mu, log_var = self.encoder(x_flat)
            z = self.reparameterize(mu, log_var)
        return z.cpu().numpy()

    def decode(self, x):
        self.eval()
        z_tensor = torch.from_numpy(x).to(self.device).float()
        with torch.no_grad():
            recon_x = self.decoder(z_tensor)
        return recon_x.cpu().numpy()


class VAEExperiment(AutoencoderExperiment):
    def __init__(self, pca_dims, enc_layers, d_latent, dec_layers=None, reg_lambda=0.001, batch_size=256, dropout_info=None, **kwargs):
        self.device = torch.device(kwargs.get("device", "cpu"))
        self.enc_layer_desc = enc_layers
        self.dec_layer_desc = dec_layers
        self.dropout_info = dropout_info
        self.batch_size = batch_size
        self.code_size = d_latent
        self._stage = 0
        self._epoch = 0
        self.reg_lambda = reg_lambda

        self._save_figs = None
        self._order = None
        self._mse_errors = None
        self._reconstructed_test = None
        self._encoded_test = None

        self._d_in = pca_dims
        self._d_out = 28*28
        super().__init__(pca_dims=pca_dims, **kwargs)
        self._history_dict = {
            'loss': [],
            'mse': [],
            'kld': [],
            'val-loss': [],
            'val-kld': [],
            'val-mse': [],
            'lambda': []
        }
        logging.info("Initialized VAEExperiment:  %s" % (self.get_name(),))

    def get_name(self, file_kind=None, suffix=None):
        drop_str = "" if self.dropout_info is None else "_Drop(l=%i,r=%.2f)" % (self.dropout_info['layer'], self.dropout_info['rate'])
        pca_str = self.pca.get_short_name()
        decoder_str = "_Decoder-%s" % "-".join(map(str, self.dec_layer_desc)) if self.dec_layer_desc is not None else ""
        encoder_str = "_Encoder-%s" % "-".join(map(str, self.enc_layer_desc))

        root = ("VAE-TORCH(%s%s%s%s_Dlatent=%i_RegLambda=%.3f)" %  (pca_str, encoder_str, drop_str, decoder_str, self.code_size, self.reg_lambda))
        if suffix is not None:
            root = "%s_%s" % (root, suffix)

        if file_kind == 'weights':
            return os.path.join(WORKING_DIR, root + ".weights")
        elif file_kind == 'image':
            return os.path.join(WORKING_DIR, root + ".image.png")
        elif file_kind == 'history':
            return os.path.join(WORKING_DIR, root + ".history.pkl")

        return root

    @staticmethod
    def from_filename(filename):
        filename = os.path.basename(filename)
        match = re.match(r"VAE-TORCH\(pca=(\d+)_hidden=([\d\-]+)_d-latent=(\d+)_reg-lambda=(\d+\.\d+)\)", filename)
        if not match:
            raise ValueError(f"Filename {filename} is not in the expected format.")

        pca_dims = int(match.group(1))
        enc_layers = list(map(int, match.group(2).split('-')))
        d_latent = int(match.group(3))
        reg_lambda = float(match.group(4))
        return VAEExperiment(pca_dims=pca_dims, enc_layers=enc_layers, d_latent=d_latent,
                             reg_lambda=reg_lambda)

    def _init_model(self):
        dec_layers = self.enc_layer_desc[::-1] if self.dec_layer_desc is None else self.dec_layer_desc
        self.model = VAE(input_dim=self._d_in,
                         enc_hidden_dims=self.enc_layer_desc,
                         dec_hidden_dims=dec_layers,
                         latent_dim=self.code_size,
                         output_dim=self._d_out,
                         device=self.device,
                         dropout_info=self.dropout_info,
                         lambda_reg=self.reg_lambda).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.print_model_architecture(self.model.encoder, self.model.decoder, self.model)

    def print_model_architecture(self, encoder, decoder, model):
        logging.info("Model architecture:")
        logging.info("Encoder:")
        for layer in encoder.children():
            logging.info("  %s" % (layer,))
        logging.info("Decoder:")
        for layer in decoder.children():
            logging.info("  %s" % (layer,))
        logging.info("VAE:")
        logging.info("  %s" % (model,))

    def train_more(self, epochs=25):
        x_train_tensor = torch.from_numpy(self.x_train_pca).float()
        y_train_tensor = torch.from_numpy(self.x_train).float()
        x_test_tensor = torch.from_numpy(self.x_test_pca).float()
        y_test_tensor = torch.from_numpy(self.x_test).float()
        x_train_flat = x_train_tensor.view(-1, self._d_in)
        y_train_flat = y_train_tensor.view(-1, self._d_out)
        x_test_flat = x_test_tensor.view(-1, self._d_in)
        y_test_flat = y_test_tensor.view(-1, self._d_out)
        train_dataset = TensorDataset(x_train_flat, y_train_flat)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataset = TensorDataset(x_test_flat, y_test_flat)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(epochs):
            t_start = time.perf_counter()
            self.model.train()
            train_losses = []
            train_loss_terms = {'kld': [], 'mse': []}
            for i, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                recon_y, mu, log_var = self.model(x_batch)
                loss, MSE, KLD = self.model.loss_function(recon_y, y_batch, mu, log_var, return_terms=True)
                train_loss_terms['kld'].append(KLD.item())
                train_loss_terms['mse'].append(MSE.item())
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
            avg_train_loss = np.mean(train_losses)
            self._history_dict['loss'].append(avg_train_loss)
            self._history_dict['mse'].append(np.mean(train_loss_terms['mse']))
            self._history_dict['kld'].append(np.mean(train_loss_terms['kld']))
            self.model.eval()
            test_losses = []
            test_loss_terms = {'kld': [], 'mse': []}
            duration = time.perf_counter() - t_start
            with torch.no_grad():
                for i, (x_batch, y_batch) in enumerate(test_loader):
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    recon_y, mu, log_var = self.model(x_batch)
                    loss, MSE, KLD = self.model.loss_function(recon_y, y_batch, mu, log_var, return_terms=True)
                    test_loss_terms['kld'].append(KLD.item())
                    test_loss_terms['mse'].append(MSE.item())
                    test_losses.append(loss.item())
            avg_test_loss = np.mean(test_losses)
            self._history_dict['val-loss'].append(avg_test_loss)
            self._history_dict['val-mse'].append(np.mean(test_loss_terms['mse']))
            self._history_dict['val-kld'].append(np.mean(test_loss_terms['kld']))
            self._history_dict['lambda'].append(self.model.lambda_reg)
            print(f"Epoch {epoch+1}/{epochs} ({duration:.4f}s), " +
                  f"Training Loss: {avg_train_loss:.4f}," +
                  f"(MSE: {self._history_dict['mse'][-1]:.4f}, " +
                  f"KLD: {self._history_dict['kld'][-1]:.4f}), " +
                  f"Test Loss: {avg_test_loss:.4f}  " +
                  f"(MSE: {self._history_dict['val-mse'][-1]:.4f}, " +
                  f"KLD: {self._history_dict['val-kld'][-1]:.4f})")

        self.save_weights()
        
    def _attempt_resume(self):
        try:
            logging.info("Attempting to load pre-trained weights...")
            self.load_weights()
            return True
        except FileNotFoundError:
            logging.info("No pre-trained weights found, starting fresh training.")
        return False

    def _encode_samples(self, x):
        return self.model.encode(x=x)

    def decode_samples(self, z):
        return self.model.decode(x=z)

    def save_weights(self):
        filename = self.get_name(file_kind='weights')
        torch.save(self.model.state_dict(), filename)
        logging.info("Saved model weights to %s", filename)

    def load_weights(self):
        filename = self.get_name(file_kind='weights')
        self.model.load_state_dict(torch.load(filename, map_location=self.device, weights_only=True))
        logging.info("Loaded model weights from %s", filename)

    def _plot_code_samples(self, n_samp=39):
        """
        TODO: Sort the code units by how likely they are to be useful in the encoding.

        For every code unit show a distribution of its activations given the digit.
        These will be like narrow-box plots with a thin line spanning +/- 3 standard deviations,
        drawn over a thick line spanning the interquartile range (IQR), and a dot representing the median.
        outliers are drawn as single pixels.
        """
        image_size_wh = 300, 1200
        dist_width = 250
        blank = np.zeros((image_size_wh[1], image_size_wh[0], 3), dtype=np.uint8)
        blank[:] = np.array(COLORS['OFF_WHITE_RGB'], dtype=np.uint8)
        codes = self._encoded_test
        n_code_units = codes.shape[1]
        digit_labels = self.y_test

        digit_subset = [1, 3, 8]
        colors = [COLORS['MPL_BLUE_RGB'],
                  COLORS['MPL_ORANGE_RGB'],
                  COLORS['MPL_GREEN_RGB']]

        colors = [np.array(c) for c in colors]
        height, t, pad_y = 12, 3, 9  # calc_scale(None, n_code_units)
        print(height, t, pad_y)

        unit_dists = [LatentDigitDist(codes[:, code_unit], digit_labels, colors=colors)
                      for code_unit in range(n_code_units)]
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)

        x = 25
        y = 10
        for unit, unit_dists in enumerate(unit_dists):
            bbox = {'x': (x, x + dist_width), 'y': (y, y + height)}
            bottom = bbox['y'][1]
            try:

                d_bbox = unit_dists.render(blank, bbox, orient='horizontal',
                                           centered=True, show_axis=False,
                                           thicknesses=[
                                               t, t, t], alphas=[.2, .5, .5, 1],
                                           digit_subset=digit_subset)[1]

                bottom = d_bbox['y'][1]

                # draw_bbox(blank, d_bbox, thickness=1, inside=True, color=(256 - bkg_color))
            except Exception as e:
                # raise e
                break

            y = bottom + pad_y
        fig, ax = plt.subplots(figsize=(5, 8))
        ax.imshow(blank)
        ax.axis('off')
        plt.suptitle("Latent Variable Distributions for %i Code Units" % n_code_units, fontsize=14)
        plt.tight_layout()

    def _eval(self):
        self._encoded_test = self.encode_samples(self.x_test)
        self._reconstructed_test = self.decode_samples(self._encoded_test)

        def mse_err(imageA, imageB):
            err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
            err /= float(imageA.shape[0])
            return err
        self._mse_errors = np.array([mse_err(img_a, img_b)
                                    for img_a, img_b in zip(self.x_test, self._reconstructed_test)])
        self._order = np.argsort(self._mse_errors)
        logging.info("Evaluation completed on %i TEST samples:", self._mse_errors.size)
        logging.info("\tMean squared error: %.4f (%.4f)", np.mean(self._mse_errors), np.std(self._mse_errors))

    def run_staged_experiment(self, n_stages=5, n_epochs=25, save_figs=True):
        if not os.path.exists(WORKING_DIR):
            os.makedirs(WORKING_DIR)

        self._attempt_resume()
        self._save_figs = save_figs
        for stage in range(n_stages):
            self._stage = stage
            logging.info("Running stage %i of %i", stage + 1, n_stages)
            self.train_more(epochs=n_epochs)

            self._eval()
            self._plot_model()

    def _plot_model(self):
        self._plot_encoding_errors(show_diffs=False)
        self._plot_encoding_errors(show_diffs=True)
        self._plot_code_samples()
        self._plot_history()
        if not self._save_figs:
            plt.show()

    def _plot_history(self):
        height_ratios = [3, 3, 3, 1]
        fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(8, 10),sharex=True,
                               gridspec_kw={'height_ratios': height_ratios})
        ax[0].plot(self._history_dict['loss'], label='Train Loss')
        ax[0].plot(self._history_dict['val-loss'], label='Validation Loss')
        ax[0].set_title('Loss history', fontsize=12)
        ax[0].legend()

        # MSE:
        ax[1].plot(self._history_dict['mse'], label='Train MSE')
        ax[1].plot(self._history_dict['val-mse'], label='Validation MSE')
        ax[1].set_title('Loss, MSE-term history', fontsize=12)
        ax[1].legend()

        # KLD:
        ax[2].plot(self._history_dict['kld'], label='Train KLD')
        ax[2].plot(self._history_dict['val-kld'], label='Validation KLD')
        ax[2].set_title('Loss, KLD-term history', fontsize=12)
        ax[2].legend()

        # Lambda:
        ax[3].plot(self._history_dict['lambda'], label='Lambda')
        ax[3].set_title('lambda history', fontsize=12)
        ax[3].set_xlabel('Epoch', fontsize=10)
        ax[3].set_ylabel('lambda', fontsize=10)
        # grid

        # turn off x-axis for all but bottom plots:
        for i in range(len(ax)-1):
            ax[i].set_xticklabels([])
            ax[i].grid(True)
            
            ax[i].set_xscale('log')

        ax[3].grid(True)
        ax[3].set_xscale('log')


        plt.tight_layout()
        if self._save_figs:
            filename = self.get_name(file_kind='image', suffix='History')
            self._maybe_save_fig(fig, filename)
        else:
            plt.show()

    def _plot_code_samples(self, n_samp=39):
        """
        TODO: Sort the code units by how likely they are to be useful in the encoding.

        For every code unit show a distribution of its activations given the digit.
        These will be like narrow-box plots with a thin line spanning +/- 3 standard deviations,
        drawn over a thick line spanning the interquartile range (IQR), and a dot representing the median.
        outliers are drawn as single pixels.
        """
        digit_subset = [1, 3, 8]

        image_size_wh = 300, 1200
        dist_width = 250
        blank = np.zeros((image_size_wh[1], image_size_wh[0], 3), dtype=np.uint8)
        blank[:] = np.array(COLORS['OFF_WHITE_RGB'], dtype=np.uint8)
        codes = self._encoded_test
        n_code_units = codes.shape[1]
        digit_labels = self.y_test

        colors = [COLORS['MPL_BLUE_RGB'],
                  COLORS['MPL_ORANGE_RGB'],
                  COLORS['MPL_GREEN_RGB']]

        colors = [np.array(c) for c in colors]
        height, t, pad_y = 12, 3, 9  # calc_scale(None, n_code_units)
        print(height, t, pad_y)

        unit_dists = [LatentDigitDist(codes[:, code_unit], digit_labels, colors=colors)
                      for code_unit in range(n_code_units)]
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)

        x = 25
        y = 10
        for unit, unit_dists in enumerate(unit_dists):
            bbox = {'x': (x, x + dist_width), 'y': (y, y + height)}
            bottom = bbox['y'][1]
            try:

                d_bbox = unit_dists.render(blank, bbox, orient='horizontal',
                                           centered=True, show_axis=False,
                                           thicknesses=[
                                               t, t, t], alphas=[.2, .5, .5, 1],
                                           digit_subset=digit_subset)[1]

                bottom = d_bbox['y'][1]

                # draw_bbox(blank, d_bbox, thickness=1, inside=True, color=(256 - bkg_color))
            except Exception as e:
                # raise e
                break

            y = bottom + pad_y
        fig, ax = plt.subplots(figsize=(5, 8))
        ax.imshow(blank)
        ax.axis('off')
        plt.suptitle("Code unit distributions (black shows all units, color shows digits: %s)\n%s" % (
            ", ".join(str(d) for d in digit_subset), self.get_name()), fontsize=14)

        suffix = "LatentDist_stage=%i" % (self._stage+1)
        filename = self.get_name(file_kind='image', suffix=suffix)
        # self._maybe_save_fig(fig, filename)
        if self._save_figs:
            cv2.imwrite(filename, blank)

    def _plot_encoding_errors(self, n_samp=39, show_diffs=False):

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
        suffix = ""
        mid_labels = ["sample group %i - %s" % (i, suffix) for i in range(1, n_quantiles-1)]
        q_labels = ['Lowest Test MSE - %s' % suffix] + mid_labels + ['Highest Test MSE - %s' % suffix]
        n_colors = len(q_labels)
        cmap = plt.get_cmap('brg', n_colors)
        colors = [cmap(i) for i in range(n_colors)]
        for i, (inds, label) in enumerate(zip(self._quant_inds, q_labels)):
            show_mosaic(q_axes[i], inds, label, color=colors[i])

        model_name = self.get_name()
        title = "Autoencoder Model: %s " % (model_name, ) +\
            "\nData: n_train=%i, n_test = %i " % (self.x_train.shape[0], self.x_test.shape[0])
        if show_diffs:
            title += "          RED: decoded pixel >= 10% too high,"
        title += "\nResults: test MSE = %.4f (%.4f)" % (np.mean(self._mse_errors), np.std(self._mse_errors))
        if show_diffs:
            title += "             BLUE: decoded pixel >= 10% too low."

        plt.suptitle(title, fontsize=14)
        self._show_err_hist(hist_axis, q_labels, colors)

        suffix = "decoded-diffs_stage=%i" % (self._stage,) if show_diffs else "decoded-images_stage=%i" % (self._stage,)
        filename = self.get_name(file_kind='image', suffix=suffix)
        self._maybe_save_fig(fig, filename)

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

    # Add plotting methods as needed to match DenseExperiment


def vae_demo():
    args = VAEExperiment.get_args("Train a variational autoencoder on MNIST data.",
                                  extra_args=[
                                      dict(name='--batch_size', type=int, default=256,
                                           help="Batch size for training (Default 256)"),
                                      dict(name='--reg_lambda', type=float, default=0.01,
                                           help='Regularization parameter for VAE (default: 0.01)'),
                                      dict(name='--d_latent', type=int, default=16,
                                           help='Dimensionality of the latent space (default: 16)')
                                  ])
    logging.info("Running VAE with args: %s", args)
    ve = VAEExperiment(
        batch_size=args.batch_size,
        enc_layers=args.layers,
        d_latent=args.d_latent,
        reg_lambda=args.reg_lambda,
        pca_dims=args.pca_dims,
        dropout_info=args.dropout,
    )
    ve.run_staged_experiment(n_stages=args.stages, n_epochs=args.epochs, save_figs=args.no_plot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    vae_demo()
