import numpy as np  # Import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import logging
from img_util import make_digit_mosaic, make_img, diff_img
import numpy as np
from tests import load_mnist
from pca import PCA
import os
from latent_var_plots import LatentDigitDist,calc_scale
from colors import COLORS

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        # Input layer to hidden layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Hidden layer 1 to hidden layer 2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Output layers for mean and log-variance
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()  # Activation function

    def forward(self, x):
        # Flatten the input if necessary (e.g., for image data)
        # x = x.view(-1, self.input_dim)

        # Pass through the linear layers and activation function
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))

        # Get the mean and log-variance of the latent distribution
        mu = self.fc_mu(h2)
        log_var = self.fc_logvar(h2)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        # Input layer from latent space
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        # Hidden layer 1 to hidden layer 2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Output layer (reconstruction)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()  # Activation function
        # For image data, consider using sigmoid to output probabilities
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # Pass through the linear layers and activation function
        h1 = self.relu(self.fc1(z))
        h2 = self.relu(self.fc2(h1))

        # Output the reconstructed data
        recon_x = self.sigmoid(self.fc3(h2))  # For image data
        return recon_x


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim, lambda_reg=0.001):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.lambda_reg = lambda_reg

    def get_name(self, file_kind=None):
        root= ("VAE-TORCH(pca=%i_hidden=%i_d-latent=%i_reg-lambda=%.5f)" % (self.input_dim, self.hidden_dim, self.latent_dim, self.lambda_reg))
        if file_kind=='weights':
            return root + ".weights"
        return root

    def save_weights(self, filename):
        torch.save(self.state_dict(), filename)
        logging.info("Saved model weights to %s", filename)

    def load_weights(self, filename):
        self.load_state_dict(torch.load(filename))
        logging.info("Loaded model weights from %s", filename)

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
    
    def encode(self, x):
        """
        Encode the input data into the latent space.
        """
        x = torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return z
    
    def decode(self, z):
        z = torch.from_numpy(z).float() if isinstance(z, np.ndarray) else z
        recon_x = self.decoder(z)
        return recon_x

    def loss_function(self, recon_y, y_batch, mu, log_var):
        # BCE = F.binary_cross_entropy(recon_y, y_batch, reduction='sum')
        MSE = F.mse_loss(recon_y, y_batch, reduction='sum')

        # If y_batch came from a source that wasn't flat,
        # you might need to flatten it here if it wasn't already in the DataLoader:
        # BCE = F.binary_cross_entropy(recon_y, y_batch.view(-1, decoder_output_dim), reduction='sum')

        # KL Divergence Loss
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) if self.lambda_reg != 0 else 0

        # L2 Regularization term for decoder weights
        # decoder_l2_reg = 0
        # for name, param in self.decoder.named_parameters():
        #    if 'weight' in name:
        #        decoder_l2_reg += torch.sum(param.pow(2))

        # Total Loss with regularization
        total_loss = (1-self.lambda_reg) * MSE + (self.lambda_reg) * KLD
        return total_loss


# (Encoder, Decoder, and VAE classes as defined in previous responses)

# (Encoder, Decoder, and VAE classes as defined in previous responses,
# ensuring the Decoder's output_dim matches the dimensionality of y_train/y_test)

# (Encoder, Decoder, and VAE classes as defined in previous responses)


def train_vae_with_separate_targets(x_train, y_train, x_test, y_test,
                                    n_hidden, n_latent, lambda_reg, epochs,
                                    learning_rate=5e-3, batch_size=128, device='cpu'):

    # 1. Data Preparation
    # Convert NumPy arrays to PyTorch Tensors and specify data type (e.g., float32)
    # Ensure y_train/y_test are correctly shaped for the decoder's output_dim

    # Calculate input_dim for the encoder (based on x_train)
    # Assuming x_train has a shape that can be flattened
    encoder_input_dim = np.prod(x_train.shape[1:])  # Calculate flattened size, ignoring batch dim
    decoder_output_dim = np.prod(y_train.shape[1:])  # Calculate flattened size, ignoring batch dim

    x_train_tensor = torch.from_numpy(x_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    x_test_tensor = torch.from_numpy(x_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()

    # Flatten the tensors
    x_train_flat = x_train_tensor.view(-1, encoder_input_dim)
    y_train_flat = y_train_tensor.view(-1, decoder_output_dim)
    x_test_flat = x_test_tensor.view(-1, encoder_input_dim)
    y_test_flat = y_test_tensor.view(-1, decoder_output_dim)
    # Create PyTorch Datasets and DataLoaders
    train_dataset = TensorDataset(x_train_flat, y_train_flat)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(x_test_flat, y_test_flat)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 2. VAE Model Initialization
    vae_model = VAE(encoder_input_dim, n_hidden, n_latent, decoder_output_dim, lambda_reg).to(device)
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=learning_rate)

    weights_file = vae_model.get_name(file_kind='weights')
    logging.info("Checking for existing weights file:  %s", weights_file)
    if os.path.exists(weights_file):
        vae_model.load_weights(weights_file)
        logging.info("---> Loaded existing model weights from %s", weights_file)
    else:
        logging.info("---> No existing model weights found, starting training from scratch.")


    # 3. Training Loop (remains the same)
    for epoch in range(epochs):
        vae_model.train()
        train_loss = 0
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            recon_y, mu, log_var = vae_model(x_batch)

            loss = vae_model.loss_function(recon_y, y_batch, mu, log_var)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss / len(train_loader.dataset):.4f}")

        vae_model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                recon_y, mu, log_var = vae_model(x_batch)
                loss = vae_model.loss_function(recon_y, y_batch, mu, log_var)
                test_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Test Loss: {test_loss / len(test_loader.dataset):.4f}")
    vae_model.save_weights(weights_file)
    logging.info("--> Saved model weights to %s", weights_file)
    return vae_model


class MNISTTest(object):
    def __init__(self):
        self._load_data()
        self._fit_model()
        self._eval_model()
        self._plot_model()

    def _load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_mnist()
        pca = PCA(dims=64)
        self.x_train_pca = pca.fit_transform(self.x_train.reshape(self.x_train.shape[0], -1))
        self.x_test_pca = pca.encode(self.x_test.reshape(self.x_test.shape[0], -1))

    def _fit_model(self):
        # Assuming vae_model, optimizer, and data_loader are set up
        print("Cuda available:", torch.cuda.is_available())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = train_vae_with_separate_targets(self.x_train_pca, self.x_train, self.x_test_pca, self.x_test,
                                                     n_hidden=512, n_latent=16, lambda_reg=0.05,
                                                     epochs=50, learning_rate=1e-3, batch_size=2048,
                                                     device=device)
        filename = "%s.weights" % self.model.get_name()

        self.model.save_weights(filename)

    def _eval_model(self):

        def mse_err(imageA, imageB):
            err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
            err /= float(imageA.shape[0])
            return err

        encoded_test_tensor = self.model.encode(self.x_test_pca)
        reconstructed_test_tensor = self.model.decode(encoded_test_tensor)
        self._encoded_test = encoded_test_tensor.detach().cpu().numpy()
        self._reconstructed_test = reconstructed_test_tensor.detach().cpu().numpy()
        self._mse_errors = np.array([mse_err(img_a, img_b)
                                    for img_a, img_b in zip(self.x_test, self._reconstructed_test)])
        self._order = np.argsort(self._mse_errors)
        logging.info("Evaluation completed on %i TEST samples:", self._mse_errors.size)
        logging.info("\tMean squared error: %.4f (%.4f)", np.mean(self._mse_errors), np.std(self._mse_errors))



    def _plot_model(self):
        self._plot_encoding_errors(show_diffs=False)
        self._plot_encoding_errors(show_diffs=True)
        self._plot_code_samples()
        plt.show()

    def _plot_code_samples(self, n_samp=39):
        """
        TODO: Sort the code units by how likely they are to be useful in the encoding.

        For every code unit show a distribution of its activations given the digit.
        These will be like narrow-box plots with a thin line spanning +/- 3 standard deviations,
        drawn over a thick line spanning the interquartile range (IQR), and a dot representing the median.
        outliers are drawn as single pixels.
        """
        image_size_wh=300,800
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
        height, t, pad_y = 14, 3, 9 # calc_scale(None, n_code_units)
        print(height, t, pad_y)

        unit_dists = [LatentDigitDist(codes[:, code_unit], digit_labels,colors=colors) for code_unit in range(n_code_units)]
        img = np.zeros((1000,1000,3),dtype=np.uint8)
            
    
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
                
                #draw_bbox(blank, d_bbox, thickness=1, inside=True, color=(256 - bkg_color))
            except Exception as e:
                # raise e
                break

            y = bottom + pad_y
        fig, ax = plt.subplots(figsize=(5,8))
        ax.imshow(blank)
        ax.axis('off')
        plt.suptitle("Latent Variable Distributions for %i Code Units" % n_code_units, fontsize=14)
        plt.tight_layout()

    def _plot_encoding_errors(self, n_samp=39, show_diffs=False):

        prefix = self.model.get_name()

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

        model_name = self.model.get_name()
        title = "Autoencoder Model: %s " % (model_name, ) +\
            "\nData: n_train=%i, n_test = %i " % (self.x_train.shape[0], self.x_test.shape[0])
        if show_diffs:
            title += "          RED: decoded pixel >= 10% too high,"
        title += "\nResults: test MSE = %.4f (%.4f)" % (np.mean(self._mse_errors), np.std(self._mse_errors))
        if show_diffs:
            title += "             BLUE: decoded pixel >= 10% too low."

        plt.suptitle(title, fontsize=14)
        self._show_err_hist(hist_axis, q_labels, colors)
        filename = "%s_%s_%s.png" % (prefix, ("diffs" if show_diffs else "reconstructed"), suffix)


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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    MNISTTest()
