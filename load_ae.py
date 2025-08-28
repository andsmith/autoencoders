from dense import DenseExperiment
from vae import VAEExperiment
import os
import re
import pprint
import logging

AUTOENCODERS = {DenseExperiment.WORKING_DIR: DenseExperiment,
                VAEExperiment.WORKING_DIR: VAEExperiment}

def get_outer_path(filename):
    full = os.path.abspath(filename)
    return os.path.split(os.path.dirname(full))[1]

def load_autoencoder(weights_filename):
    """
    Load an autoencoder experiment by name.
    """
    path = get_outer_path(weights_filename)
    experiment_class = AUTOENCODERS[path]
    info = experiment_class.parse_filename(weights_filename)
    logging.info("Loaded autoencoder info: %s", pprint.pformat(info))
    return experiment_class.from_filename(weights_filename)


def get_ae_dir(weights_filename):
    # if the filename has no directory, use the part after the first underscore and before the first open-paren
    base_name = os.path.basename(weights_filename)
    match = re.match(r"^[^\-_]+[\-_](.+?)\(", base_name)
    class_name = match.groups(0)[0] if match else None
    class_name = class_name.replace('-TORCH', '')
    # Matches start of experiments:
    for exp_dir in AUTOENCODERS.keys():
        if exp_dir.startswith(class_name):
            return exp_dir
    return class_name
