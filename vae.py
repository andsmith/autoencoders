import pickle
import cv2
from flask import json
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
from colors import COLORS, MPL_CYCLE_COLORS
from img_util import make_digit_mosaic, make_img, diff_img
from latent_var_plots import LatentDigitDist
import time
import json
from experiment import AutoencoderExperiment,  font_set_from_filename
from anneal import make_annealing_schedule


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
        self.encoder = Encoder(input_dim, enc_hidden_dims, latent_dim, dropout_layer=d_l, dropout_rate=d_r)
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

    def loss_function(self, recon_y, y_batch, mu, log_var, return_terms=False, beta=0.0):
        MSE = F.mse_loss(recon_y, y_batch, reduction='mean')
        # get negative log-likelihood loss to compare:
        # MSE = F.binary_cross_entropy(recon_y, y_batch, reduction='mean')
        KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = MSE + (beta) * KLD
        if return_terms:
            return total_loss, MSE, KLD
        return total_loss
    

    def collapse_metric(self, mu, log_var, threshold=1e-2):
        """
        Measure of latent space collapse, from "Understanding the Difficulty of Training
        Variational Autoencoders" by Lucas Theis and Matthias Bethge, 2017"
        """
        # collect mu for many x (e.g., whole validation set): mus shape [N, D]
        var_mu = mu.var(dim=0)   # variance across dataset per-dim
        active_mask = var_mu > threshold   # threshold e.g. 1e-2 or 1e-3 (tune)
        fraction_active = active_mask.float().mean()
        return  fraction_active

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
    WORKING_DIR = "VAE-results"

    def __init__(self, dataset, pca_dims, enc_layers, d_latent, dec_layers=None, reg_lambda=0.001, binary_input=False,
                 batch_size=512, whiten_input=False, learn_rate=1e-3, dropout_info=None, anneal=None, **kwargs):
        self.device = torch.device(kwargs.get("device", "cpu"))

        self.dropout_info = dropout_info
        self._stage = 0
        self._epoch = 0
        self.reg_lambda = reg_lambda
        self.anneal = anneal
        self._save_figs = None
        self._order = None
        self._mse_errors = None
        self._reconstructed_test = None
        self._encoded_test = None

        self._d_in = pca_dims
        self._d_out = 28*28
        super().__init__(dataset=dataset, pca_dims=pca_dims, enc_layers=enc_layers, dec_layers=dec_layers, d_latent=d_latent, batch_size=batch_size,
                         whiten_input=whiten_input, learning_rate=learn_rate, binary_input=binary_input, **kwargs)
        self._history_dict = {
            'loss': [],
            'mse': [],
            'kld': [],
            'collapse': [],
            'val-loss': [],
            'val-kld': [],
            'val-mse': [],
            'val-collapse': [],
            'lambda': [],
            'learn_rate': []
        }
        logging.info("Initialized VAEExperiment:  %s" % (self.get_name(),))

    def get_name(self, file_kind=None, suffix=None):
        drop_str = "" if self.dropout_info is None else "_Drop(l=%i,r=%.2f)" % (
            self.dropout_info['layer'], self.dropout_info['rate'])
        pca_str = self.pca.get_short_name()
        decoder_str = "_Decoder-%s" % "-".join(map(str, self.dec_layer_desc)) if self.dec_layer_desc is not None else ""
        encoder_str = "_Encoder-%s" % "-".join(map(str, self.enc_layer_desc))
        dataset_str = "%s%s" % (self.get_dataset_name(), "-BIN" if self.binary_input else "")
        root = ("%s_VAE-TORCH(%s%s%s%s_Dlatent=%i)" %
                (dataset_str, pca_str, encoder_str, drop_str, decoder_str, self.code_size))
        if suffix is not None:
            root = "%s_%s" % (root, suffix)

        if file_kind == 'weights':
            return os.path.join(VAEExperiment.WORKING_DIR, root + ".weights")
        elif file_kind == 'fontset':
            return os.path.join(VAEExperiment.WORKING_DIR, root + "_fontset.json")
        elif file_kind == 'image':
            return os.path.join(VAEExperiment.WORKING_DIR, root + ".image.png")
        elif file_kind == 'history':
            return os.path.join(VAEExperiment.WORKING_DIR, root + ".history.pkl")

        return root

    @staticmethod
    def from_filename(filename):

        import ipdb
        ipdb.set_trace()

        filename = os.path.abspath(filename)
        logging.info("Loading VAEExperiment from filename: %s", filename)
        params = VAEExperiment.parse_filename(filename)

        # disambiguate cmd-line fontset and fontset associated with weights file
        weights_font_filename = font_set_from_filename(filename)
        param_font_filename = params['dataset'] if params['dataset'].endswith('.json') else None
        if weights_font_filename is not None and param_font_filename is not None:
            if weights_font_filename != param_font_filename:
                logging.warning("Font set file in weights filename is different from that in parameters, using weights file's:\n\t%s\n\t%s",
                                weights_font_filename, param_font_filename)

        if weights_font_filename is not None:
            logging.info("Using font set file from weights filename: %s", weights_font_filename)
            params['dataset'] = weights_font_filename

        params = VAEExperiment.parse_filename(os.path.basename(filename))
        params['lambda_reg'] = 0.0  # not in the filename, set to default
        network = VAEExperiment(**params)
        network.load_weights(filename)
        return network

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

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
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

    def _calc_anneal_schedule(self, n_epochs, n_batches):
        """
        From paper: "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing."


           ------   ------   ------   ------  <- beta_max
          /     |  /     |  /     |  /     
         /      | /      | /      | /       
        /       |/       |/       |/         <- 0.0

        ramp_frac = fraction of each cycle spent ramping up to beta_max.
        m_cycles = 4.

        returns: list of lists of beta_max, indexing like [epoch_no][minibatch_no].

        """
        n_steps = n_epochs * n_batches

        
        if self.anneal is None:
            cycles = np.ones(n_steps) * self.reg_lambda
        else:
            period = n_steps // self.anneal['m_cycles']
            n_ramp_steps = int(period * self.anneal['ramp_frac'])
            n_flat_steps = period - n_ramp_steps
            cycle = np.concatenate((
                np.linspace(0.0, self.anneal['beta_max'], n_ramp_steps, endpoint=False),
                np.ones(n_flat_steps) * self.anneal['beta_max']
            ))

            cycles = np.tile(cycle, self.anneal['m_cycles'])
            cycles = np.concatenate((cycles, np.ones(n_steps - cycles.size) * self.anneal['beta_max']))

        plt.plot(cycles)
        plt.title("Beta annealing schedule")
        plt.xlabel("Minibatch number")
        plt.ylabel("Beta value")
        plt.grid(True)
        plt.show()
        # break up into list of lists:
        beta_schedule = []
        for epoch in range(n_epochs):
            start = epoch * n_batches
            end = start + n_batches
            beta_schedule.append(cycles[start:end].tolist())
        return beta_schedule



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

        n_batches_per_epoch = len(train_loader)
        self._beta_schedule = self._calc_anneal_schedule(n_epochs=epochs, n_batches=n_batches_per_epoch)

        for epoch in range(epochs):
            t_start = time.perf_counter()
            self.model.train()
            train_losses = []
            train_loss_terms = {'kld': [], 'mse': [], 'collapse': []}
            for i, (x_batch, y_batch) in enumerate(train_loader):
                beta = self._beta_schedule[self._epoch][i]
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                recon_y, mu, log_var = self.model(x_batch)
                loss, MSE, KLD = self.model.loss_function(recon_y, y_batch, mu, log_var, return_terms=True, beta=beta)
                train_loss_terms['kld'].append(KLD.item())
                train_loss_terms['mse'].append(MSE.item())
                train_loss_terms['collapse'].append(self.model.collapse_metric(mu, log_var).item())
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
            avg_train_loss = np.mean(train_losses)

            self._history_dict['loss'].append(avg_train_loss)
            self._history_dict['mse'].append(np.mean(train_loss_terms['mse']))
            self._history_dict['kld'].append(np.mean(train_loss_terms['kld']))
            self._history_dict['collapse'].extend(train_loss_terms['collapse'])
            self.model.eval()
            test_losses = []
            test_loss_terms = {'kld': [], 'mse': [], 'collapse': []}
            duration = time.perf_counter() - t_start
            with torch.no_grad():
                for i, (x_batch, y_batch) in enumerate(test_loader):
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    recon_y, mu, log_var = self.model(x_batch)
                    loss, MSE, KLD = self.model.loss_function(recon_y, y_batch, mu, log_var, return_terms=True)
                    test_loss_terms['kld'].append(KLD.item())
                    test_loss_terms['mse'].append(MSE.item())
                    test_loss_terms['collapse'].append(self.model.collapse_metric(mu, log_var).item())
                    test_losses.append(loss.item())
            avg_test_loss = np.mean(test_losses)
            self._history_dict['val-loss'].append(avg_test_loss)
            self._history_dict['val-mse'].append(np.mean(test_loss_terms['mse']))
            self._history_dict['val-kld'].append(np.mean(test_loss_terms['kld']))
            self._history_dict['val-collapse'].append(np.mean(test_loss_terms['collapse']))
            self._history_dict['lambda'].append(self.model.lambda_reg)
            self._history_dict['learn_rate'].append(self.learning_rate)
            print(f"Epoch {epoch+1}/{epochs} ({duration:.4f}s), " +
                  f"Training Loss: {avg_train_loss:.4f}," +
                  f"(MSE: {self._history_dict['mse'][-1]:.6f}, " +
                  f"KLD: {self._history_dict['kld'][-1]:.6f}, " +
                  f"frac active: {self._history_dict['collapse'][-1]:.4f}), " +
                  f"Test Loss: {avg_test_loss:.4f}  " +
                  f"(MSE: {self._history_dict['val-mse'][-1]:.6f}, " +
                  f"KLD: {self._history_dict['val-kld'][-1]:.6f}, " +
                  f"frac active: {self._history_dict['val-collapse'][-1]:.4f})")

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

    def save_weights(self, filename=None):

        filename = self.get_name(file_kind='weights') if filename is None else filename
        torch.save(self.model.state_dict(), filename)
        logging.info("Saved model weights to %s", filename)
        hist_filename = self.get_name(file_kind='history')
        with open(hist_filename, 'w') as f:
            json.dump(self._history_dict, f)
        logging.info("Saved model history to %s", hist_filename)
        self._save_font_set_info(filename)

    def load_weights(self, filename=None):

        filename = self.get_name(file_kind='weights') if filename is None else filename
        self.model.load_state_dict(torch.load(filename, map_location=self.device, weights_only=True))
        logging.info("Loaded model weights from %s", filename)
        hist_filename = self.get_name(file_kind='history') if filename is None else "%s.history.pkl" % (
            os.path.splitext(filename)[0])
        with open(hist_filename, 'r') as f:
            self._history_dict.update(json.load(f))
        logging.info("Loaded model history from %s", hist_filename)

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
        if not os.path.exists(VAEExperiment.WORKING_DIR):
            os.makedirs(VAEExperiment.WORKING_DIR)

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
        height_ratios = [2, 2, 2, 2, 1, 1]
        fig, ax = plt.subplots(nrows=len(height_ratios), ncols=1, figsize=(8, 10), sharex=True,
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

        # Collapse:
        x = np.linspace(0, len(self._history_dict['learn_rate'])-1, len(self._history_dict['collapse']))
        ax[3].plot(x,self._history_dict['collapse'], label='Train Collapse')
        ax[3].plot(self._history_dict['val-collapse'], label='Validation Collapse')
        ax[3].set_title('Loss, Collapse-term history', fontsize=12)
        ax[3].legend()

        # beta
        beta_schedule_flat = [b for epoch in self._beta_schedule for b in epoch]
        x = np.linspace(0, len(self._history_dict['learn_rate'])-1, len(beta_schedule_flat))
        ax[4].plot(x, beta_schedule_flat, label='Beta')
        ax[4].set_title('Beta history', fontsize=12)
        ax[4].set_xlabel('minibatch', fontsize=10)
        ax[4].set_ylabel('Beta', fontsize=10)

        # learning rate:
        ax[5].plot(self._history_dict['learn_rate'], label='Learning Rate')
        ax[5].set_title('Learning Rate history', fontsize=12)
        ax[5].set_xlabel('Epoch', fontsize=10)
        ax[5].set_ylabel('Learning Rate', fontsize=10)

        # turn off x-axis for all but bottom plots:
        for i in range(len(ax)-1):
            ax[i].set_xticklabels([])
            ax[i].grid(True)

            ax[i].set_yscale('log')

        ax[3].grid(True)
        ax[3].set_yscale('log')

        filename = self.get_name(file_kind='image', suffix='History')
        self._maybe_save_fig(fig, filename)

    def _plot_code_samples(self, n_samp=39):
        """
        TODO: Sort the code units by how likely they are to be useful in the encoding.

        For every code unit show a distribution of its activations given the digit.
        These will be like narrow-box plots with a thin line spanning +/- 3 standard deviations,
        drawn over a thick line spanning the interquartile range (IQR), and a dot representing the median.
        outliers are drawn as single pixels.
        """
        digit_subset = [0, 1, 3, 5, 8]

        image_size_wh = 300, 690
        dist_width = 250
        blank = np.zeros((image_size_wh[1], image_size_wh[0], 3), dtype=np.uint8)
        blank[:] = np.array(COLORS['OFF_WHITE_RGB'], dtype=np.uint8)
        codes = self._encoded_test
        value_span = np.min(codes), np.max(codes)
        margin = 0.02*(value_span[1] - value_span[0])
        value_span = (value_span[0] - margin, value_span[1] + margin)
        n_code_units = codes.shape[1]
        digit_labels = self.y_test

        colors = MPL_CYCLE_COLORS

        colors = [np.array(c) for c in colors]
        height, t, pad_y = 12, 3, 9  # calc_scale(None, n_code_units)
        print(height, t, pad_y)

        unit_dists = [LatentDigitDist(codes[:, code_unit], digit_labels)
                      for code_unit in range(n_code_units)]
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)

        x = 25
        y = 10
        for unit, unit_dists in enumerate(unit_dists):
            bbox = {'x': (x, x + dist_width), 'y': (y, y + height)}
            bottom = bbox['y'][1]

            loc = (bbox['x'][0], bbox['y'][0])
            scale = bbox['x'][1] - bbox['x'][0]
            print(loc, scale)
            try:
                d_bbox = unit_dists.render(blank, loc_xy=loc, scale=scale, orient='horizontal',
                                           centered=False, show_axis=True, separation_px=1,
                                           val_span=value_span,
                                           thicknesses_px=[2, 2, 2, 2], alphas=[.2, .5, .5, 1],
                                           digit_subset=digit_subset, colors=colors)
            except:
                logging.info("OUT OF SPACE!")
                break

            bottom = d_bbox['y'][1]

            # draw_bbox(blank, d_bbox, thickness=1, inside=True, color=(256 - bkg_color))

            y = bottom + pad_y
        fig, ax = plt.subplots(figsize=(5, 8))
        ax.imshow(blank)
        ax.axis('off')
        plt.suptitle("Code unit distributions (black shows all units, color shows digits: %s)\n%s" % (
            ", ".join(str(d) for d in digit_subset), self.get_name()), fontsize=14)

        suffix = "LatentDist_stage=%i" % (self._stage+1)
        filename = self.get_name(file_kind='image', suffix=suffix)
        self._maybe_save_fig(fig, filename)

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

    @staticmethod
    def parse_filename(filename):
        """
        Parse the filename to extract experiment parameters.
        :returns: dict with network architecture parameters, can be passed as a set of arguments to VAEExperiment().
        """
        print("\n%s\n" % filename)
        file = os.path.split(os.path.abspath(filename))[1]
        kind_pattern = r'([^_]+)[_\-]([a-zA-Z\-\_]+)'

        # internal structure is everything between outer parentheses
        int_start, int_end = file.find("("), len(file)-file[::-1].find(")")

        internal = file[int_start:int_end]  # keep parentheses
        prefix = file[:int_start]

        pca_vs_rest_pattern = r'(digits-)?PCA\(([^\)]+)\)_(.+)'  # dataset-pca(pca_params)
        # PCA format is "<dims>,<whitening>" where <dims> is an int, and <whitening> is "W" or "UW"
        pca_int_pattern = r'(\d+),(W|UW)'
        # ints separated by dash, final is code size (must have at least 1 number here)
        enc_pattern = r'Encoder-(.+?)[^\d-]'
        # optional, same format, assumed reverse of encoder if not present, l=layer (of encoder unit, can't be final/code layer), r=rate.
        dec_pattern = r'Decoder-([0-9\-]+)'
        dropout_pattern = r'Drop\(l=(\d+),r=(\d\.?\d*)\)'
        d_latent_pattern = r'Dlatent=(\d+)'

        file_kind_match = re.search(kind_pattern, prefix)
        if not file_kind_match:
            raise ValueError("No dataset_model string found in filename: %s, in prefix: %s" % (filename, prefix))
        dataset, vae = file_kind_match.groups()
        if vae != 'VAE-TORCH':
            raise ValueError("Expected 'VAE-TORCH' in dataset_model string, found: %s" % vae)

        dataset_fontset_pattern = r"(digits|fashion|mnist|alphanumeric)-([\d]+f-[\d]+c)"
        dataset_bare_pattern = r"(digits|fashion|mnist|alphanumeric)"
        bare_match = re.search(dataset_bare_pattern, dataset.lower())
        fontset_match = re.search(dataset_fontset_pattern, dataset.lower())
        if fontset_match:
            # Check for parallel fontset file.
            fontset_filename = "%s_fontset.json" % (os.path.splitext(os.path.splitext((filename))[0])[0])
            dataset = fontset_filename
            if not os.path.exists(dataset):
                raise FileNotFoundError("Fontset file referenced in weights filename ('%s'), but no fontset file found at that location:\n\t%s" % (
                    fontset_filename, dataset))
        elif bare_match:
            dataset = bare_match.group(1)
        else:
            raise ValueError("No dataset match found in filename: %s, in prefix: %s" % (filename, prefix))

        pca_match = re.search(pca_vs_rest_pattern, internal)
        if not pca_match:
            raise ValueError("No PCA match found in filename: %s, in params: %s" % (filename, internal))
        pca_desc, arch_desc = pca_match.group(2), pca_match.group(3)
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
            logging.info("##################Found explicit decoder layer: %s", dec_layers)
        else:
            dec_layers = enc_layers[::-1]  # reverse of encoder, minus code layer
            logging.info("No decoder layer found, using reverse of encoder: %s", dec_layers)

        d_latent_match = re.search(d_latent_pattern, arch_desc)
        if d_latent_match:
            d_latent = int(d_latent_match.group(1))
        else:
            d_latent = None

        dropout_match = re.search(dropout_pattern, arch_desc)
        if dropout_match:
            dropout_info = dict(layer=int(dropout_match.group(1)),
                                rate=float(dropout_match.group(2)))
        else:
            dropout_info = None

        params = {
            'enc_layers': list(enc_layers),
            'd_latent': d_latent,
            'pca_dims': 0 if dims == 784 else dims,
            'whiten_input': whiten,
            'dropout_info': dropout_info,
            'dec_layers': list(dec_layers),
            'dataset': dataset
        }

        return params


def vae_demo():
    """
    Add two parameters, reg_lambda (single float, regularization weight for KLD term),
       and anneal (m_cycles, beta_max, ramp_frac) for cyclical annealing of KLD weight.
       """
    args = VAEExperiment.get_args("Train a variational autoencoder on MNIST data.",
                                  extra_args=[
                                      dict(name='--reg_lambda', type=float, default=0.01,
                                           help='Regularization parameter for VAE (default: 0.01)'),
                                      dict(name='--anneal', type=float, nargs=3, default=None,
                                           help='Annealing parameters (m_cycles, beta_max, ramp_frac) for KLD weight.')
                                  ])
    logging.info("Running VAE with args: %s", args)
    if args.anneal is not None:
        args.anneal = {'m_cycles': int(args.anneal[0]), 'beta_max': args.anneal[1], 'ramp_frac': args.anneal[2]}
    ve = VAEExperiment(
        batch_size=args.batch_size,
        enc_layers=args.layers,
        dec_layers=args.dec_layers,
        d_latent=args.d_latent,
        reg_lambda=args.reg_lambda,
        pca_dims=args.pca_dims,
        learn_rate=args.learn_rate,
        binary_input=args.binary_input,
        dataset=args.dataset,
        anneal=args.anneal,
        dropout_info=args.dropout,
    )
    # If no font set is specified on the cmd line but one is associated with the weights file, it will have the wrong
    # data loaded so the VAEExperiment should be reloaded with the font set filename as the dataset.

    filename = ve.get_name(file_kind='weights')
    fontset_filename = ve.get_name(file_kind='fontset')
    if not args.dataset.endswith('.json') and os.path.exists(fontset_filename):
        logging.info("Reloading VAEExperiment with font set file associated with weights: %s", fontset_filename)
        ve = VAEExperiment(
            batch_size=args.batch_size,
            enc_layers=args.layers,
            dec_layers=args.dec_layers,
            d_latent=args.d_latent,
            reg_lambda=args.reg_lambda,
            pca_dims=args.pca_dims,
            learn_rate=args.learn_rate,
            binary_input=args.binary_input,
            dataset=fontset_filename,
            dropout_info=args.dropout)

    ve.run_staged_experiment(n_stages=args.stages, n_epochs=args.epochs, save_figs=args.no_plot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    vae_demo()
