import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
from tensorflow.keras.datasets import mnist, fashion_mnist


class MNISTData(object):
    _loader = mnist

    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self._loader.load_data()
        self.x_train = self.x_train.astype(np.float32) / 255.0
        self.x_test = self.x_test.astype(np.float32) / 255.0
        self.y_train = tf.keras.utils.to_categorical(self.y_train, 10)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, 10)
        logging.info("MNIST Data loaded successfully:")
        logging.info("\tTraining samples: %d, %s", len(self.x_train), self.x_train.shape)
        logging.info("\tTesting samples: %d, %s", len(self.x_test), self.x_test.shape)


class FashionMNISTData(MNISTData):
    _loader = fashion_mnist


def test_mnist_data():
    mnist_data = MNISTData()
    print("Training data shape:", mnist_data.x_train.shape)
    print("Testing data shape:", mnist_data.x_test.shape)

    # Display a sample image
    plt.imshow(mnist_data.x_train[0], cmap='gray')
    plt.title(f"Label: {np.argmax(mnist_data.y_train[0])}")
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_mnist_data()
