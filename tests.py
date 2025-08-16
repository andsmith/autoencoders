from dense import DenseExperiment
from vae import VAEExperiment
from img_util import make_digit_mosaic
import numpy as np
import matplotlib.pyplot as plt
import logging
from keras.datasets import fashion_mnist, mnist


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], -1)) / 255.0
    x_test = x_test.reshape((x_test.shape[0], -1)) / 255.0
    return (x_train, y_train), (x_test, y_test)


def load_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], -1)) / 255.0
    x_test = x_test.reshape((x_test.shape[0], -1)) / 255.0
    return (x_train, y_train), (x_test, y_test)

def test_vae_from_filename():
    filename="VAE(d_input=784, hidden_layers=128_512, d_latent=2).weights.h5"

    net= VAEExperiment.from_filename(filename)
    logging.info("Loaded VAEExperiment from filename: %s", filename)
    _, (x_test, y_test) = load_mnist()
    digits = x_test[:100,:]
    coded_digits = net.encode_samples(digits)
    decoded_digits = net.decode_samples(coded_digits)
    orig_images = [(digits[i,:].reshape(28,28)*255.0).astype(np.uint8) for i in range(100)]
    decoded_images = [(decoded_digits[i,:].reshape(28,28)*255.0).astype(np.uint8) for i in range(100)]
    orig_mosaic = make_digit_mosaic(orig_images, mosaic_aspect=1.0)
    decoded_mosaic = make_digit_mosaic(decoded_images, mosaic_aspect=1.0)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(orig_mosaic, cmap='gray')
    ax[0].set_title("Original Images")
    ax[1].imshow(decoded_mosaic, cmap='gray')
    ax[1].set_title("Decoded Images")

    digit_order = np.argsort(y_test[:100])
    code_img = coded_digits[digit_order]
    n_digits = [np.sum(y_test[:100] == i) for i in range(10)]

    ax[2].imshow(code_img, cmap='viridis')
    # colorbar
    plt.colorbar(ax[2].imshow(coded_digits, cmap='viridis'), ax=ax[2])
    ax[2].set_title("Encoded Samples")
    plt.show()


def test_dense_from_filename():
    file_small = r".\Dense-results\digits_Dense(PCA(784,UW)_units=2-16)history.json"
    net = DenseExperiment.from_filename(file_small)
    logging.info("Loaded DenseExperiment from filename: %s", file_small)
    x_test, y_test = net.x_test, net.y_test
    digits = x_test[:100,:]
    coded_digits = net.encode_samples(digits)
    decoded_digits = net.decode_samples(coded_digits)
    orig_images = [(digits[i,:].reshape(28,28)*255.0).astype(np.uint8) for i in range(100)]
    decoded_images = [(decoded_digits[i,:].reshape(28,28)*255.0).astype(np.uint8) for i in range(100)]
    orig_mosaic = make_digit_mosaic(orig_images, mosaic_aspect=1.0)
    decoded_mosaic = make_digit_mosaic(decoded_images, mosaic_aspect=1.0)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(orig_mosaic, cmap='gray')
    ax[0].set_title("Original Images")
    ax[1].imshow(decoded_mosaic, cmap='gray')
    ax[1].set_title("Decoded Images")

    digit_order = np.argsort(y_test[:100])
    code_img = coded_digits[digit_order]
    n_digits = [np.sum(y_test[:100] == i) for i in range(10)]

    ax[2].imshow(code_img, cmap='viridis')
    # colorbar
    plt.colorbar(ax[2].imshow(coded_digits, cmap='viridis'), ax=ax[2])
    ax[2].set_title("Encoded Samples")
    plt.suptitle("Dense Autoencoder sample digits & codes\nfrom file: " + file_small)
    plt.tight_layout()
    plt.show()


def test_fashion_mnist():
    data = load_fashion_mnist()
    logging.info("Loaded Fashion MNIST dataset")
    logging.info("Training data shape: %s", data[0][0].shape)
    logging.info("Test data shape: %s", data[1][0].shape)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_dense_from_filename()
    #test_fashion_mnist()