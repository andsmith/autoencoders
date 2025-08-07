# Adapted from response of google AI to prompt "simple 1-layer variational autoencoder in keras"
# Variational Autoencoder (VAE) implementation using Keras
import keras
from keras import layers
from keras import backend as K
import numpy as np
from sympy import comp
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
# import colormaps
import seaborn as sns
# Set the style for seaborn
import cv2
import logging
import os
import json


class VAEModel(keras.Model):

    def __init__(self, encoder, decoder, reg_lambda, **kwargs):
        self.reg_lambda = reg_lambda
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        # Define the optimizer here if you don't want to pass it to compile
        self.optimizer_vae = keras.optimizers.Adam()  # <--- Optimizer defined here

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):  # <--- Added call method
        _, _, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    def test_step(self, data):
        total_loss, reconstruction_loss, kl_loss = self.get_losses(data[0])
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def get_losses(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(data, reconstruction)
        )
        if self.reg_lambda == 0:
            # If no regularization, return only reconstruction loss
            return reconstruction_loss, reconstruction_loss, 0.0
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss * (1-self.reg_lambda) + kl_loss * self.reg_lambda
        return total_loss, reconstruction_loss, kl_loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, kl_loss = self.get_losses(data)
        grads = tape.gradient(total_loss, self.trainable_weights)
        # <--- Use the internally defined optimizer
        self.optimizer_vae.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

class VAE(object):

    def __init__(self, d_input, ds_hidden, d_latent, reg_lambda=0.01):
        self.d_input = d_input
        self.ds_hidden = ds_hidden
        self.d_latent = d_latent
        self.reg_lambda = reg_lambda
        self.history = {
            "loss": [],
            "reconstruction_loss": [],
            "kl_loss": [],
            "val_loss": [],
            "val_reconstruction_loss": [],
            "val_kl_loss": [],
        }

        self._model = self._init_model()
        # compile the model
        # <--- Pass the internally defined optimizer
        self._model.compile(optimizer=self._model.optimizer_vae, loss=tf.keras.losses.MeanSquaredError())

    def get_name(self):
        hidden_layer_st = "_".join(str(n) for n in self.ds_hidden)
        return "VAE(d_input=%i, hidden_layers=%s, d_latent=%i)" % (self.d_input, hidden_layer_st, self.d_latent)

    def _init_model(self):
        # Encoder
        self.inputs = keras.Input(shape=(self.d_input,))
        self.enc_hidden = [self.inputs]
        for enc_layer, n_hidden in enumerate(self.ds_hidden):
            activation = 'relu' if enc_layer < len(self.ds_hidden) - 1 else 'sigmoid'
            hidden_layer = layers.Dense(n_hidden, activation=activation)(self.enc_hidden[-1])
            self.enc_hidden.append(hidden_layer)
        self.z_mean = layers.Dense(self.d_latent)(self.enc_hidden[-1])
        self.z_log_var = layers.Dense(self.d_latent)(self.enc_hidden[-1])
        self.z = layers.Lambda(self.sampling, output_shape=(self.d_latent,), name='z')([self.z_mean, self.z_log_var])
        self.encoder = keras.Model(self.inputs, [self.z_mean, self.z_log_var, self.z], name="encoder")

        # Instantiate the encoder and decoder models
        self.decoder_input = keras.Input(shape=(self.d_latent,))
        self.dec_hidden = [self.decoder_input]
        for n_hidden in reversed(self.ds_hidden):
            hidden_layer = layers.Dense(n_hidden, activation='relu')(self.dec_hidden[-1])
            self.dec_hidden.append(hidden_layer)
        # Decoder output

        self.decoder_output = layers.Dense(self.d_input, activation='sigmoid')(self.dec_hidden[-1])
        self.decoder = keras.Model(self.decoder_input, self.decoder_output, name="decoder")

        return VAEModel(self.encoder, self.decoder, reg_lambda=self.reg_lambda)

    @tf.function
    def sampling(self, args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def save_history(self, filename):
        """
        Save the training history to a JSON file.
        """
        with open(filename, 'w') as f:
            json.dump(self.history, f)

    def load_history(self, filename):
        """
        Load the training history from a JSON file.
        """
        with open(filename, 'r') as f:
            self.history = json.load(f)

    def fit(self, x, epochs=50, batch_size=256, validation_data=None):
        """
        Fit the VAE model to the data.
        """
        history = self._model.fit(
            x,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data
        ).history
        
        self.history['loss'].extend(history['loss'])
        self.history['reconstruction_loss'].extend(history['reconstruction_loss'])   
        self.history['kl_loss'].extend(history['kl_loss'])
        self.history['val_loss'].extend(history['val_loss'])
        self.history['val_reconstruction_loss'].extend(history['val_reconstruction_loss'])
        self.history['val_kl_loss'].extend(history['val_kl_loss'])

    def predict(self, x):
        return self._model(x)
    def save_weights(self, filename):
        self._model.save_weights(filename)
    def load_weights(self, filename):
        self._model.load_weights(filename)

    def plot_history(self,ax=None):
        """
        Plot the training history.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.history['loss'], label='Loss')
        ax.plot(self.history['reconstruction_loss'], label='Reconstruction Loss')
        ax.plot(self.history['kl_loss'], label='KL Loss')
        if 'val_loss' in self.history:
            ax.plot(self.history['val_loss'], label='Validation Loss')
            ax.plot(self.history['val_reconstruction_loss'], label='Validation Reconstruction Loss')
            ax.plot(self.history['val_kl_loss'], label='Validation KL Loss')
        ax.set_title('VAE Training History')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend() 

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((len(x_train), -1)).astype('float32') / 255.0
    x_test = x_test.reshape((len(x_test), -1)).astype('float32') / 255.0
    return (x_train, y_train), (x_test, y_test)


DEFAULT_PARAMS = {
    'd_hidden': (256,),  # Hidden layer size
    'd_latent': 32,   # Latent space size
    'n_epochs': 50,   # Number of epochs to train
    'batch_size': 256,  # Batch size for training
    'reg_lambda': 0.05  # Regularization parameter
}


def wts_name_to_hist_name(wts_name):
    return wts_name.replace('.weights.h5', '.history.json')


def test_vae(params=None):
    param = params if params is not None else DEFAULT_PARAMS
    d_input = 784
    ds_hidden = params['d_hidden']
    d_latent = params['d_latent']
    n_epochs = params['n_epochs']
    batch_size = params['batch_size']
    reg_lambda = params['reg_lambda']

    vae = VAE(d_input, ds_hidden, d_latent, reg_lambda=reg_lambda)

    # Create dummy data
    (x_train, y_train), (x_test, y_test) = load_mnist()
    logging.info(f"Training VAE with input shape: {x_train.shape}")
    model_name = vae.get_name()
    filename = "%s.weights.h5" % model_name
    if os.path.exists(filename):
        logging.info(f"Loading existing model weights from {filename}")
        _ = vae.predict(x_train[:30, :])
        vae.load_weights(filename)
        vae.load_history(wts_name_to_hist_name(filename))
    else:
        logging.info(f"Training new model: {model_name}")

    vae.fit(x_train, epochs=n_epochs, batch_size=batch_size, validation_data=(x_test, x_test))
    # build and save the model after training
   
    _ = vae.predict(x_train[:30, :])
    vae.save_weights(filename)
    vae.save_history(wts_name_to_hist_name(filename))
    logging.info(f"Model weights saved to {filename}")
    vae.plot_history()
    plt.show()
    def _encode_decode_samples(x):
        """
        Encode and decode n sample images
        """
        z_mean, z_log_var, z = vae.encoder.predict(x)
        recon = vae.decoder.predict(z)
        return (recon * 255.0).astype(np.uint8), z

    def make_digit_comparison_image(shape, digit):
        """
        digit comparison:  Make an image with 2 side-by-side grids of digits, each of shape shape[0]xshape[1],
            left are original, right are reconstructed.
        Sort images by reconstruction loss.
        """
        n_test = shape[0]*shape[1]
        h = shape[0]*28
        w = shape[1]*28
        orig_img = np.zeros((h, w), dtype=np.uint8)
        recon_img = np.zeros((h, w), dtype=np.uint8)
        valid = np.where(y_test == digit)[0]
        sample_inds = np.random.choice(valid, size=n_test, replace=False)

        # Fill the original image grid
        reconstructed, codes = _encode_decode_samples(x_test[sample_inds])
        errors = np.mean(np.abs(x_test[sample_inds] - reconstructed), axis=1)
        sorted_inds = np.argsort(errors)
        sample_inds = sample_inds[sorted_inds]
        ind = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                orig_img[i*28:(i+1)*28, j*28:(j+1)*28] = (x_test[sample_inds[ind]]
                                                          * 255.0).astype(np.uint8).reshape(28, 28)
                ind += 1

        # Generate the reconstructed image grid
        recon_img = np.zeros((h, w), dtype=np.uint8)
        ind = 0
        # import ipdb; ipdb.set_trace()
        for i in range(shape[0]):
            for j in range(shape[1]):
                recon = reconstructed[sorted_inds[ind]]
                recon_img[i*28:(i+1)*28, j*28:(j+1)*28] = recon.reshape(28, 28)
                ind += 1
        return np.concatenate((orig_img, recon_img), axis=1), codes

    def make_dimension_dist_image(codes, sample_dims, x_spread=10, h=200):
        """
        For each sample dimension, plot the distribution of values of the encoding for each digit.
        Show each digit's distribution as a vertical band (x-dimension random) in 10 different columns & colors.

             sample dim 1          (for the n sample dimensions)

          . . o . . . . . . O .   
          o o O . o . . . o O .   
          o O o . o O . o O o .   
          O o . . o . . o o . o   
          o . . o O . O O . . .   
          . . . O O . . . o . o   

        """
        colors = sns.color_palette("husl", 10)  # Use seaborn's color palette for distinct colors
        all_codes = np.concatenate(codes, axis=0)

        header = 25

        def get_dim_strip(dimension, dot=2):
            high, low = np.max(all_codes[:, dimension], axis=0), np.min(all_codes[:, dimension], axis=0)
            strip_img = np.zeros((h+header, (x_spread + (dot-1))*10, 3), dtype=np.uint8)

            def get_digit_strip(digit):
                color = (np.array(colors[digit]) * 255).astype(np.uint8)
                digit_strip_img = np.zeros((h, x_spread + (dot-1), 3), dtype=np.uint8)  # + 255
                values = codes[digit][:, dimension]
                digit_values = (high-values) / (high-low) * (h-(dot-1))
                digit_values = digit_values.astype(int)
                digit_x_offsets = np.random.randint(0, x_spread, size=len(digit_values))

                for (dx, dy) in zip(digit_x_offsets, digit_values):
                    digit_strip_img[dy:dy+dot, dx:dx+dot, :] = color.reshape(1, 1, 3)

                return digit_strip_img

            strips = [get_digit_strip(digit) for digit in range(10)]
            strip_img[header:, :, :] = np.concatenate(strips, axis=1)

            # Write dimension in header space
            header_txt = "dim %i" % dimension

            cv2.putText(strip_img, header_txt, (25, header-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            return strip_img
        strip_images = [get_dim_strip(dim) for dim in sample_dims]
        spacer = np.zeros((h+header, 10, 3), dtype=np.uint8)  # Spacer between dimensions

        strip_images = [np.concatenate((img, spacer), axis=1) for img in strip_images]
        strip_image = np.concatenate(strip_images, axis=1)
        # Add a title for each dimension
        return strip_image

    comparisons = [make_digit_comparison_image((20, 3), digit) for digit in range(10)]

    columns = [comp[0] for comp in comparisons]
    columns = np.concatenate(columns, axis=1)
    n_sample_dims = 6
    sample_dimensions = np.random.choice(range(d_latent), size=n_sample_dims, replace=False)
    codes = [comp[1] for comp in comparisons]
    code_image = make_dimension_dist_image(codes, sample_dimensions)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 13))
    ax[0].imshow(columns, cmap='gray')
    ax[0].set_title("Original (left) and Reconstructed (right) Images\n" +
                    vae.get_name() + "\n" +
                    "Trained for %i epochs, final loss %.6f, (%.6f * (1-l) + %.6f * (l))" % (n_epochs,
                                                                                             vae.history['loss'][-1],
                                                                                             vae.history['reconstruction_loss'][-1],
                                                                                             vae.history['kl_loss'][-1]))
    ax[0].axis('off')

    ax[1].imshow(code_image)
    ax[1].set_title("Distribution of Encoded Values in Latent Space\n" +
                    "for Sample Dimensions %s" % str(sample_dimensions))

    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_vae(params=dict(
        d_hidden=(512,),  # Hidden layer(s)' size(s)
        d_latent=16,   # Latent space size
        n_epochs=100,   # Number of epochs to train
        batch_size=8192,  # Batch size for training
        reg_lambda=0.0,  # Regularization parameter
    ))

    logging.info("VAE training completed.")
