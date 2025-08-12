from dense import DenseExperiment, WORKING_DIR as DENSE_DIR
from vae import VAEExperiment, WORKING_DIR as VAE_DIR
import os

AUTOENCODERS = {DENSE_DIR: DenseExperiment,
                VAE_DIR: VAEExperiment}

def get_outer_path(filename):
    full = os.path.abspath(filename)
    return os.path.split(os.path.dirname(full))[1]

def load_autoencoder(weights_filename):
    """
    Load an autoencoder experiment by name.
    """
    path = get_outer_path(weights_filename)
    experiment_class = AUTOENCODERS[path]
    return experiment_class.from_filename(weights_filename)