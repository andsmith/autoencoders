"""
Look for all history files, plot them together, 
mouse-over individual curves to display the responsible network.
"""
from operator import contains
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import logging
import sys
import re
from dense import DenseExperiment
from vae import VAEExperiment
import argparse
from matplotlib.widgets import CheckButtons

EXP_CLASSES = {
    "dense": DenseExperiment,
    "vae": VAEExperiment
}

TEST_WORKING_DIR = None  # "Dense-results_test"  # SET TO NONE FOR PRODUCTION


class MultiHistory(object):
    """
    Manage data/files for many experiments.

    filter on hyperparameters:
        Latent dimensionality / code size
        Encoder layer description, Decoder layer description
        Preprocessing (PCA dim)
        Dropout,
        regularization_lambda

    All of these are inferred from filenames using the experiment classes parse_filename() function.

    """

    def __init__(self, dataset, exp_class):
        self.dataset = dataset
        self.exp_class = exp_class
        self.hist_files, self.creation_times = self._scan_history_files()
        self.history = self._load_data()
        self.keys_by_file = {file: self._get_plot_info(hist) for file, hist in self.history.items()}
        self.files_by_param_val = self._analyze_params()

    def _analyze_params(self):
        """
        Get constructor parameters for each experement by parsing the filename.
        Figure out which parameters are present that we can filter on,
        for each, get a list of values it can take, for each value,
        get a list of files that have that parameter set to that value.
        """
        # param -> param value -> [file1(param=value), file2(param=value), ...]
        par_values = {'d_latent': {},
                      'enc_layers': {},
                      'dec_layers': {},
                      'pca_dims': {},
                      'pca_whiten': {},
                      'dropout_layer': {},
                      'reg_lambda': {}}

        model_params = [(f, self.exp_class.parse_filename(f)) for f in self.hist_files]

        for file, param_set in model_params:

            def accum_param(par):
                if par == 'dropout_layer':
                    if 'dropout_info' in param_set and param_set['dropout_info'] is not None:
                        value = param_set['dropout_info']['layer']
                    else:
                        value = None
                elif par not in param_set:
                    return
                else:
                    value = param_set[par]

                if par in ['enc_layers', 'dec_layers']:
                    value = tuple(value) if value is not None else None

                if value not in par_values[par]:
                    par_values[par][value] = []

                par_values[par][value].append(file)

            for par in par_values.keys():
                accum_param(par)

        return par_values

    def _scan_history_files(self):
        
        
        hist_files = []
        file_timestamps = []
        search_dir = TEST_WORKING_DIR if TEST_WORKING_DIR else self.exp_class.WORKING_DIR
        for root, dirs, files in os.walk(search_dir):
            for fname in files:
                print(fname)
                if "history" in fname and fname.startswith(self.dataset) and not fname.endswith(".png") and not 'BIN_' in fname:
                    print("*#(*#&$(*#&$(*#&$(&))))")
                    file_path = os.path.join(root, fname)
                    hist_files.append(file_path)
                    file_timestamps.append(os.path.getmtime(file_path))
        logging.info("Found %i history files in %s for dataset %s.", len(hist_files), search_dir, self.dataset)
        return hist_files, file_timestamps

    def _get_plot_info(self, hist_dict):
        keys = [key for key in hist_dict.keys()]
        validation_keys = [k for k in keys if k.startswith("val")]
        loss_keys = [k for k in keys if any([(val_key.endswith(k) and k not in validation_keys)
                                            for val_key in validation_keys])]
        extra_keys = [k for k in keys if (k not in loss_keys and k not in validation_keys)]
        return loss_keys, validation_keys, extra_keys

    def _load_data(self):
        history_data = {}
        for file_ind in range(len(self.hist_files)):
            with open(self.hist_files[file_ind], "r") as f:
                hist_data = json.load(f)
            history_data[self.hist_files[file_ind]] = hist_data
        return history_data


def test_multi_history():
    dataset = 'digits'
    exp_class = DenseExperiment
    hist = MultiHistory(dataset, exp_class)

    logging.info("Loaded %i files for %s using the %s dataset.", len(hist.hist_files), exp_class.__name__, dataset)


class HistoryPlotter(object):
    def __init__(self, dataset, exp_class, plot_keys=None):
        self.plot_keys = plot_keys if plot_keys is not None else None
        self._data = MultiHistory(dataset, exp_class)
        
        #self._filters,self._boxes, self._spa_args = self._make_filters()
        #import ipdb; ipdb.set_trace()
        self._plot()
    '''
    def _check_toggle(self, event):
        """
        Toggle the visibility of the lines based on the check button state.
        """
        label = event.label
        if label in self._data.keys_by_file:
            for line_data in self.lines:
                if line_data['hist_file'] == label:
                    line_data['line'].set_visible(event.get_status())
            plt.draw()

    def _make_filters(self):
        """
        For each of the filter parameters, create a check button box with one button per possible value of
        the parameter.  The text of the button is the "<name>=<value>  (<file count>)"
        :returns: 
           - filters: A dictionary mapping parameter names to their filter options.
           - boxes: A dictionary mapping parameter names to their corresponding CheckButtons.
           - spa_args: kwargs to subplots_adjust to make room for the check buttons.
        """
        filters = {}
        
        spa_args = {}
        label_boxes = {}  # backwards mapping for lookup

        for par, values in self._data.files_by_param_val.items():
            filters[par] = []
            for value in values:
                file_count = len(self._data.files_by_param_val[par][value])
                filters[par].append(f"{par}={value}  ({file_count})")
        # Create the checkboxes, down the right side of the plot, scale vertically be number of options.
        v_size_per_line = 0.04
        v_sep = 0.025
        x_left = 0.8
        width = 0.15  # width of the checkboxes
        top_y = 0.15  # move this down as buttons are added
        for par, options in filters.items():
            n_lines = len(options)
            box_y_bottom = top_y - n_lines * (v_size_per_line)
            options_str = [f"{par}={opt}" for opt in options]  # add the parameter name to each option
            top_y = box_y_bottom - v_sep  # move the top down for the next box
        spa_args = {'left': 0.0, 'right': x_left - 0.01, 'bottom': 0.0}
        return filters, boxes, spa_args
    '''
    def _get_plot_info(self, hist_data):
        loss_keys = [k for k in hist_data.keys() if ("loss" in k and 'val' not in k)]
        val_loss_keys = [k for k in hist_data.keys() if "val-loss" in k]
        extra_keys = [k for k in hist_data.keys() if k not in loss_keys and k not in val_loss_keys]
        return loss_keys, val_loss_keys, extra_keys

    def _plot(self):
        
        fig, ax = None, None
        plot_axes = {}  # history key to axis
        loss_keys, val_loss_keys, extra_keys = [], [], []
        lines = []  # dict with (line_artist,original_color,highlighted, hist_file, key)

        highlight_colors = {'mouseover': 'green',
                            'same_file': 'red'}
        
        for file_name, hist_data in self._data.history.items():
            loss_keys, val_loss_keys, extra_keys = self._get_plot_info(hist_data)

            if fig is None:
                print("SETTING UP AXES __-----------____")
                # Set up the plot when we get the first data loaded.
                if self.plot_keys is not None:
                    n_loss = len([k for k in loss_keys if k in self.plot_keys])
                    n_extra = len([k for k in extra_keys if k in self.plot_keys])
                else:
                    n_loss = len(loss_keys)
                    n_extra = len(extra_keys)
                height_ratios = (3*np.ones(n_loss)).tolist() + np.ones(n_extra).tolist()
                fig, ax = plt.subplots(n_loss + n_extra, 1, figsize=(10, 8), sharex=True,
                                       gridspec_kw={'height_ratios': height_ratios})
                n_ax = [0]

                def _try_key(key):
                    if self.plot_keys is None or key in self.plot_keys:
                        plot_axes[key] = ax[n_ax[0]]
                        n_ax[0] += 1
                # figure out which keys we're going to plot
                for key in loss_keys:
                    _try_key(key)
                for key in val_loss_keys:
                    train_key = key.replace("val-", "") if "-" in key else key.replace("val_", "")
                    if train_key not in plot_axes:
                        raise ValueError(
                            f"Missing training key for validation key: {key}.\nHistory file has these keys: {list(hist_data.keys())}")
                    plot_axes[key] = plot_axes[train_key]
                for key in extra_keys:
                    _try_key(key)

            for key, values in hist_data.items():
                if key in plot_axes:
                    if key in val_loss_keys:
                        plot_kind = "--"
                    else:
                        plot_kind = "-"
                    line = plot_axes[key].plot(values, plot_kind, picker=True)[0]
                    lines.append({'line': line,
                                  'original_color': line.get_color(),
                                  'highlight_color': None,
                                  'hist_file': file_name,
                                  'key': key})

        for key in loss_keys + extra_keys:
            if key in plot_axes:
                plot_axes[key].set_ylabel(key)

                plot_axes[key].grid(which='both')
                # plot_axes[key].set_xscale('log')
                if key in loss_keys:
                    plot_axes[key].set_yscale('log')
        n_histories = len(self._data.hist_files)
        history_prefix = f"Training History for {self._data.exp_class.__name__} on {self._data.dataset} dataset, N_experiments:  {n_histories}"
        default_sup_title = history_prefix + "\nMouse-over a curve to see which experiment it belongs to."
        last_title = [default_sup_title]

        def on_check(event):
            """"""

        def on_motion(event):
            if event.inaxes:  # Check if mouse is within the axes
                line_under_mouse = [line_data for line_data in lines if line_data['line'].contains(event)[0]]
                hist_file = line_under_mouse[0]['hist_file'] if any(line_under_mouse) else None

                # set highlight state:
                for line_data in lines:
                    if hist_file is not None and line_data['hist_file'] == hist_file:
                        if line_data['key'] == line_under_mouse[0]['key']:
                            line_data['highlight_color'] = highlight_colors['mouseover']
                        else:
                            line_data['highlight_color'] = highlight_colors['same_file']
                    else:
                        line_data['highlight_color'] = None
                # set title:
                if hist_file is not None:
                    print("Mouse over:", hist_file, "key:", line_under_mouse[0]['key'])
                    key = line_under_mouse[0]['key']
                    x = event.xdata  # TODO:  Update to use epoch index if storing that in history.
                    print(x)
                    ind = np.sum(np.arange(len(self._data.history[hist_file][key])) < x)
                    ind = np.clip(ind, 0, len(self._data.history[hist_file][key]) - 1)
                    loss_value = self._data.history[hist_file][key][ind]
                    suptitle_suffix = "\nFile:  %s\nValue: %.6f" % (hist_file, loss_value)
                    new_title = history_prefix + suptitle_suffix
                else:
                    new_title = default_sup_title
                if new_title != last_title[0]:
                    plt.suptitle(new_title, fontsize=12)
                    last_title[0] = new_title

                # Set all lines to their correct color:
                for line_data in lines:
                    line = line_data['line']
                    highlight_color = line_data['highlight_color']
                    original_color = line_data['original_color']
                    if highlight_color is not None:
                        # and not highlighted:
                        # Highlight the line if it's hovered and not already highlighted
                        line.set_color(highlight_color)
                        line.set_linewidth(3)
                        line_data['highlighted'] = True
                        fig.canvas.draw_idle()  # Redraw the figure
                    else:
                        # Revert to original appearance if not hovered but was highlighted
                        line.set_color(original_color)
                        line.set_linewidth(1)  # Default linewidth
                        line_data['highlighted'] = False
                        fig.canvas.draw_idle()

        plt.suptitle(default_sup_title, fontsize=14)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        #fig.subplots_adjust(**self._spa_args)
        fig.tight_layout()
        plt.show()


def do_plot(dataset, exp_class_name, plot_keys):

    exp_class = EXP_CLASSES.get(exp_class_name)
    if exp_class is None:
        logging.error("Unknown experiment class: %s", exp_class_name)
        sys.exit(1)
    HistoryPlotter(dataset, exp_class, plot_keys)


def get_args():
    parser = argparse.ArgumentParser(description="Plot training history")
    parser.add_argument("dataset", type=str, help="Dataset name:  'digits' or 'fashion'")
    parser.add_argument("experiment_class", type=str, help="Experiment class name: one of %s " % (EXP_CLASSES.keys(),))
    parser.add_argument("--plot_keys", nargs="*", type=str, default=None, help="Keys to plot (default all)")
    args = parser.parse_args()
    return args.dataset, args.experiment_class, args.plot_keys


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    do_plot(*get_args())
