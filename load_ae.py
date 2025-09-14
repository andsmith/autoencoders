from dense import DenseExperiment
from vae import VAEExperiment
import os
import re
import pprint
import logging
from tests import load_mnist, load_fashion_mnist
from load_typographyMNIST import load_alphanumeric, load_numeric, GOOD_CHAR_SET
import numpy as np
import json

AUTOENCODERS = {DenseExperiment.WORKING_DIR: DenseExperiment,
                VAEExperiment.WORKING_DIR: VAEExperiment}

LOADERS = {'digits': load_mnist,
           'numeric': load_numeric,
           'alphanumeric': lambda **kwargs: load_alphanumeric(**kwargs, numeric_labels=False, subset=GOOD_CHAR_SET),
           'fashion': load_fashion_mnist}


def get_outer_path(filename):
    full = os.path.abspath(filename)
    return os.path.split(os.path.dirname(full))[1]

def load_autoencoder(weights_filename):
    """
    Load an autoencoder experiment by name.
    return (autoencoder_experiment, dataset)
    """
    path = get_outer_path(weights_filename)
    experiment_class = AUTOENCODERS[path]
    info = experiment_class.parse_filename(weights_filename)
    logging.info("Loaded autoencoder info: %s", pprint.pformat(info))
    return experiment_class.from_filename(weights_filename)


def get_ae_dir(weights_filename):
    # if the filename has no directory, use the part after the first underscore and before the first open-paren
    base_name = os.path.basename(weights_filename)
    prefix = base_name[:base_name.find('(')]
    
    # The class name is the non dash/underscore part before the first open paren
    match = re.search(r"_([^_]+?)\(", base_name)
    class_name = match.groups(0)[0] if match else None
    class_name = class_name.replace('-TORCH', '')
    # Matches start of experiments:
    for exp_dir in AUTOENCODERS.keys():
        if exp_dir.lower().startswith(class_name.lower()):
            return exp_dir
    return class_name
