"""
Interactively partition the set of fonts into similar groups

Process:  
  1. Specify clustering parameters with the UI,
  2. convert each font into a vector using the selected characters (row of selected/unselected boxes),
  3. compute the clustering, goto 1 until it looks useful, then
  4. save the clustering to a json file.

Clustering algorithms and parameters:

    - Preprocessing (PCA):
        - P, the number of principle components to use
        - Raw/Whitened, does every component have unit variance, or are they 
          in proportion to the original magnitudes?

    - Clustering:
        - K, the number of clusters
        - Outlier filtering method:  (TODO)
            - remove small clusters (below a threshold size) & re-cluster up to N times.
        - Algorithms:
            - k-means, params:
                - Distance metric:  cosine, euclidean\
            - spectral, params:
                - Similarity graph type:
                    - KNN (k-nearest neighbors), shared-only / inclusive
                - Affinity:  nearest_neighbors, precomputed
                - Number of clusters:  K

"""
import logging
import numpy as np
import cv2
from color_blit import draw_color_tiles_cython as color_blit
from colors import COLORS
from load_typographyMNIST import load_alphanumeric, GOOD_CHAR_SET
from pca import PCA
import json
import os
import sys
import argparse
import pprint
from mnist import AlphaNumericMNISTData
from util import fit_spaced_intervals, draw_bbox, get_font_size, scale_bbox, write_lines, split_bbox
from abc import ABC, abstractmethod
from tools import RadioButtons, Slider, Button, ToggleButton
from clustering import KMeansAlgorithm, SpectralAlgorithm, DBScanAlgorithm, DBScanManualAlgorithm
from similarity import NNSimGraph
from img_util import make_assign_gallery, make_digit_mosaic

SIM_GRAPHS = {'KNN': {'type': NNSimGraph,
                      'param': 'k'}}
#   'Epsilon': {'type':EpsilonSimGraph,
#              'param': 'epsilon_rel'},
#   'Full': {'type':FullSimGraph,
#            'param': 'sigma_rel'}}

ALGS = {'KMeans': KMeansAlgorithm,
        'Spectral': SpectralAlgorithm,
        'DBScan-auto': DBScanAlgorithm,
        'DBScan': DBScanManualAlgorithm}

DISTANCE_LABELS = {'euclidean': "EUC-D",
                   'cosine': 'COS-D'}


class ClusterWindow(ABC):
    def __init__(self, app, bbox_rel=None):
        """
        Initialize the cluster window.
        :param app: The main application instance.
        :param bbox_rel: The relative bounding box for the window,
          {'x': (x_min, x_max), 'y': (y_min, y_max)} all in [0, 1]
        """
        self.app = app
        self._bbox_rel = bbox_rel if bbox_rel is not None else {'x': [0.0, 1.0], 'y': [0.0, 1.0]}
        self._last_size = None

    @abstractmethod
    def _draw(self, image):
        pass

    @abstractmethod
    def resize(self, new_size):
        """
        Calculate necessary adjustments for the new size.
        """
        pass

    def draw(self, image):
        if self._last_size is None or self._last_size != (image.shape[1], image.shape[0]):
            self.resize((image.shape[1], image.shape[0]))
            self._last_size = (image.shape[1], image.shape[0])
        return self._draw(image)


class ControlWindow(ClusterWindow):
    """
            +--------------------+
            | Clustering alg     |
            |  - Kmeans          | <- unselected
            |  * Spectral        | <- selected
            |                    |
            | Common Params:     |
            |  +---------K----+  |  <- slider
            |  +--pca-----+ [W]  |  <- slider, whitened/unwhitened toggle button
            |                    |
            |                    |
            | Sim-graph params:  |
            |  +--k-------+ [M]  |  <- updates for selected sim-graph type.
            |                    |
            |   CLUSTER FONTS    |  <- button
            +--------------------+
    """
    _LAYOUT = {'indent_px': 0,
               'area_names': ['alg', 'param', 'sim-param', 'action'],
               'area_weights': [5, 4, 3, 1],  # 5 vertical areas of the control window
               'pad_weight': .25}

    _ALGORITHMS = [k for k in ALGS.keys()]
    _SIMGRAPHS = [k for k in SIM_GRAPHS.keys()]

    _PARAM_RANGES = {'K': (2, 42),
                     'knn_k': (1, 30),
                     'PCA': (0, 784)}

    def __init__(self, app, bbox_rel=None):
        super().__init__(app, bbox_rel)
        self._cur_alg = None
        self.widgets = None

    def _init_widgets(self):
        int_fmt_str = ": %i"
        self.widgets = {
            'alg': RadioButtons(self._area_bboxes['alg'], title='Algorithm', callback=self._alg_change,
                                options=self._ALGORITHMS),
            'k_slider': Slider(self._k_slider_bbox, label='N. Clusters', callback=self._k_change, default=2, range=self._PARAM_RANGES['K'], format_str=int_fmt_str),
            'pca_slider': Slider(self._pca_slider_bbox, label='PCA Dims', callback=self._preproc_change, default=100, range=self._PARAM_RANGES['PCA'], format_str=int_fmt_str),
            'whiten_toggle': ToggleButton(self._whiten_toggle_bbox, label='Norm', callback=self._preproc_change),
            'knn_k': Slider(self._simgraph_param_split_bbox, label='KNN K', callback=self._param_change, default=5, range=self._PARAM_RANGES['knn_k'], format_str=int_fmt_str),
            'knn_mutual_toggle': ToggleButton(self._simgraph_toggle_bbox, label='Mutual', callback=self._param_change),
            'dbscan_min_size': Slider(self._db_scan_min_size_bbox, label='Min Size', callback=self._param_change, default=5, range=(1, 20), format_str=int_fmt_str),
            'dbscan_eps': Slider(self._db_scan_eps_bbox, label='Epsilon', callback=self._param_change, default=0.05, range=(0.0, 1.0), format_str=": %.2f"),
            'dist_metric_toggle': ToggleButton(self._dist_metric_toggle_bbox,
                                               label=DISTANCE_LABELS['euclidean'],
                                               alt_label=DISTANCE_LABELS['cosine'],
                                               callback=self._param_change),
            'run': Button(self._run_bbox, label='Cluster', callback=self._run, border_indent=4),
            'save': Button(self._save_bbox, label='Save', callback=self.app.save_clusters, border_indent=6)
        }
        # set defaults here
        self.widgets['dist_metric_toggle'].set_value(DISTANCE_LABELS['euclidean'])
        self.widgets['pca_slider'].set_value(0)
        self._cur_alg = self.widgets['alg'].get_value()
        self.app.refresh_alg()

    def resize(self, new_size):
        left, right = int(self._bbox_rel['x'][0] * new_size[0]), int(self._bbox_rel['x'][1] * new_size[0])
        top, bottom = int(self._bbox_rel['y'][0] * new_size[1]), int(self._bbox_rel['y'][1] * new_size[1])
        width, height = right - left, bottom - top
        self._bbox = {'x': (left, right),
                      'y': (top, bottom)}
        unit_height = height / (self._LAYOUT['pad_weight'] *
                                (len(self._LAYOUT['area_weights']) - 1) + sum(self._LAYOUT['area_weights']))

        self._area_bboxes = {}
        y = top
        for area_name, area_weight in zip(self._LAYOUT['area_names'], self._LAYOUT['area_weights']):
            area_height = unit_height * area_weight
            self._area_bboxes[area_name] = {'x': (left, right), 'y': (int(y), int(y + area_height))}
            y += area_height + unit_height * self._LAYOUT['pad_weight']

        # These slider areas need to be split
        self._k_slider_bbox, temp = split_bbox(self._area_bboxes['param'], weight=0.4, orient='h')
        self._pca_slider_bbox, temp = split_bbox(temp, weight=0.66, orient='v')
        self._whiten_toggle_bbox, self._dist_metric_toggle_bbox = split_bbox(temp, weight=0.5, orient='h')
        self._simgraph_param_split_bbox, self._simgraph_toggle_bbox = split_bbox(
            self._area_bboxes['sim-param'], weight=0.75, orient='v')
        self._simgraph_param_full_bbox = self._area_bboxes['sim-param']
        self._run_bbox, self._save_bbox = split_bbox(self._area_bboxes['action'], weight=0.66, orient='v')

        self._db_scan_min_size_bbox, self._db_scan_eps_bbox = split_bbox(
            self._area_bboxes['sim-param'], weight=0.5, orient='h')

        # initialize or move widgets
        if self.widgets is None:
            self._init_widgets()
            self._alg_change(None)  # for ui consistency
        else:
            self.widgets['alg'].move_to(self._area_bboxes['alg'])
            self.widgets['k_slider'].move_to(self._k_slider_bbox)
            self.widgets['pca_slider'].move_to(self._pca_slider_bbox)
            self.widgets['whiten_toggle'].move_to(self._whiten_toggle_bbox)
            self.widgets['dist_metric_toggle'].move_to(self._dist_metric_toggle_bbox)
            self.widgets['knn_k'].move_to(self._simgraph_param_split_bbox)
            self.widgets['knn_mutual_toggle'].move_to(self._simgraph_toggle_bbox)
            self.widgets['run'].move_to(self._run_bbox)
            self.widgets['save'].move_to(self._save_bbox)
            self.widgets['dbscan_min_size'].move_to(self._db_scan_min_size_bbox)
            self.widgets['dbscan_eps'].move_to(self._db_scan_eps_bbox)

    def get_params(self):
        return {'pca_dims': int(self.widgets['pca_slider'].get_value()),
                'whiten_toggle': self.widgets['whiten_toggle'].get_value(),
                'alg': self.widgets['alg'].get_value(),
                'k_slider': int(self.widgets['k_slider'].get_value()),
                'dist_metric_name': self.widgets['dist_metric_toggle'].get_value(),
                'knn_k': int(self.widgets['knn_k'].get_value()),
                'knn_mutual_toggle': self.widgets['knn_mutual_toggle'].get_value(),
                'dbscan_min_size': int(self.widgets['dbscan_min_size'].get_value()),
                'dbscan_eps': float(self.widgets['dbscan_eps'].get_value())
                }

    def _draw(self, image):
        # draw_bbox(image, self._bbox, 3, COLORS['DARK_NAVY_RGB'])
        # for bbox in self._area_bboxes.values():
        #     draw_bbox(image, bbox, 2, COLORS['DARK_GRAY'])
        # for bbox in [self._k_slider_bbox, self._pca_slider_bbox, self._whiten_toggle_bbox,self._simgraph_param_split_bbox, self._simgraph_toggle_bbox]:
        #     draw_bbox(image, bbox, 1, COLORS['LIGHT_GRAY'])

        for widget_name, widget in self.widgets.items():
            widget.render(image)

        return image

    def _alg_change(self, _):

        alg_name = self.widgets['alg'].get_value()

        if alg_name == 'KMeans':
            self.widgets['knn_k'].set_visible(False)
            self.widgets['knn_mutual_toggle'].set_visible(False)
            self.widgets['dbscan_min_size'].set_visible(False)
            self.widgets['dbscan_eps'].set_visible(False)
            self.widgets['k_slider'].set_visible(True)  

        elif alg_name == 'Spectral':
            self.widgets['knn_k'].set_visible(True)
            self.widgets['knn_mutual_toggle'].set_visible(True)
            self.widgets['dbscan_min_size'].set_visible(False)
            self.widgets['dbscan_eps'].set_visible(False)
            self.widgets['k_slider'].set_visible(True)  

        elif alg_name == 'DBScan-auto':
            self.widgets['knn_k'].set_visible(False)
            self.widgets['knn_mutual_toggle'].set_visible(False)
            self.widgets['dbscan_min_size'].set_visible(True)
            self.widgets['dbscan_eps'].set_visible(False)
            self.widgets['k_slider'].set_visible(False)  

        elif alg_name == 'DBScan':
            self.widgets['knn_k'].set_visible(False)
            self.widgets['knn_mutual_toggle'].set_visible(False)
            self.widgets['dbscan_min_size'].set_visible(True)
            self.widgets['dbscan_eps'].set_visible(True)
            self.widgets['k_slider'].set_visible(False)  # K not used for DBScan

        else:
            raise ValueError("Unknown algorithm name: %s" % alg_name)

        self.widgets['run'].set_visible(True)
        self._cur_alg = alg_name
        self.app.refresh_alg()

    def _param_change(self, value):
        print("CHANGING PARAMS:", value)
        # enable run button
        self.widgets['run'].set_visible(True)
        self.app.update_params(self.get_params())

    def _k_change(self, value):
        value = int(value)
        print("CHANGING K:", value)
        # enable run button
        self.widgets['run'].set_visible(True)
        self.app.update_k(value)

    def _preproc_change(self, value):
        print("CHANGING PREPROC PARAMS:", value)
        # enable run button
        self.widgets['run'].set_visible(True)
        self.app.clear_preprocessing()

    def _run(self):
        self.app.recluster()
        # disable run button
        self.widgets['run'].set_visible(False)

    def on_mouse(self, event, x, y, flags, param):
        for _, widget in self.widgets.items():
            widget.on_mouse(event, x, y, flags, param)


class CharsetWindow(ClusterWindow):
    _LAYOUT = {'indent_px': 28,
               'x_div_rel': .15,
               'char_spacing_frac': 0.03,
               'button_spacing_frac': 0.5,
               'bbox_color': COLORS['DARK_NAVY_RGB'],
               'text_color': COLORS['DARK_NAVY_RGB'],
               'unused_color': COLORS['SKY_BLUE'],
               'mouseover_color': COLORS['DARK_RED_RGB']}

    def __init__(self, app, bbox_rel=None, shape=[5, 20]):
        """
        Pick from the 94 characters to use in the clustering.
            Quick-buttons:  "All, None, Good", select all, none or those in the GOOD_CHAR_SET
            Initialize the character set artist.

        +------------------------------------------+ 
        | All      0 1 2 3 4 5 6 7 8 9 - = ` | ] [ | 
        |          A B C D E F G H I J K L M N O P |
        | None     Q R S T U V W X Y Z a b c d e f |
        |          h i j k l m n o p ~ ! @ # $ % ^ |
        | Good     & * ( ) _ + / < > ? / . ,       |
        +------------------------------------------+
          ^indent  ^x-div                        ^ indent

        mouse-down / dragging with mouse down over a char toggles its state.

        :param app: The main application instance.
        :param size: Will draw in box of this size (width, height)
        :param shape: The grid shape for the character set buttons (rows, cols):
        """
        super().__init__(app, bbox_rel)
        self.chars = sorted(np.unique(app.data.labels_train))
        self.char_states = {char: (char in app.char_set) for char in self.chars}
        self.nrows, self.ncols = shape
        self._mouse_down = False
        self._mouse_over_text = None  # button text

    def resize(self, size):
        """
        """
        left, right = int(self._bbox_rel['x'][0] * size[0]), int(self._bbox_rel['x'][1] * size[0])
        top, bottom = int(self._bbox_rel['y'][0] * size[1]), int(self._bbox_rel['y'][1] * size[1])
        width, height = right - left, bottom - top
        self._bbox = {'x': (left, right),
                      'y': (top, bottom)}

        ind = self._LAYOUT['indent_px']
        x_mid = int(width * self._LAYOUT['x_div_rel']) + left

        # multi-char buttons:
        button_left, button_right = left + ind, x_mid - ind
        button_top, button_bottom = top + ind, bottom - ind
        button_y = fit_spaced_intervals((button_top, button_bottom), 3, self._LAYOUT['button_spacing_frac'],
                                        fill_extent=False)
        button_x = (button_left, button_right)
        button_h, button_w = button_y[0][1] - button_y[0][0], button_x[1] - button_x[0]
        bpad = min(14, button_h // 4)
        button_font_size, xy_rel, thickness = get_font_size('Good', (button_w, button_h),
                                                            incl_baseline=False, pad=bpad)

        self._buttons = {'All': {'text': 'All',
                                 'bbox': {'x': button_x, 'y': button_y[0]},
                                 'text_pos': (button_x[0] + xy_rel[0], button_y[0][0] + xy_rel[1])},
                         'None': {'text': 'None',
                                  'bbox': {'x': button_x, 'y': button_y[1]},
                                  'text_pos': (button_x[0] + xy_rel[0], button_y[1][0] + xy_rel[1])},
                         'Good': {'text': 'Good',
                                  'bbox': {'x': button_x, 'y': button_y[2]},
                                  'text_pos': (button_x[0] + xy_rel[0], button_y[2][0] + xy_rel[1])}}

        for button in self._buttons:
            self._buttons[button]['mouseover'] = False
            self._buttons[button]['font_size'] = button_font_size
            self._buttons[button]['color'] = self._LAYOUT['bbox_color']
            self._buttons[button]['thickness'] = thickness

        # Character buttons
        char_button_left, char_button_right = x_mid + ind, right - ind
        char_button_top, char_button_bottom = top + ind, bottom - ind
        char_button_x = fit_spaced_intervals((char_button_left, char_button_right), self.ncols, self._LAYOUT['char_spacing_frac'],
                                             fill_extent=False)
        char_button_y = fit_spaced_intervals((char_button_top, char_button_bottom), self.nrows, self._LAYOUT['char_spacing_frac'],
                                             fill_extent=False)
        char_button_h = char_button_y[0][1] - char_button_y[0][0]
        char_button_w = char_button_x[0][1] - char_button_x[0][0]
        cpad = min(4, char_button_h // 3)

        char_font_size, xy_rel, thickness = get_font_size("#", (char_button_w, char_button_h),
                                                          incl_baseline=False, pad=cpad)

        self._char_buttons = {}
        for c_ind, char in enumerate(self.chars):

            col = c_ind % self.ncols
            row = c_ind // self.ncols
            bbox = {'x': char_button_x[col], 'y': char_button_y[row]}
            (width, height), baseline = cv2.getTextSize(str(char), cv2.FONT_HERSHEY_COMPLEX, char_font_size, thickness)
            text_x = (char_button_x[col][0] + char_button_x[col][1]) // 2 - width // 2
            text_y = (char_button_y[row][0] + char_button_y[row][1]) // 2 + height // 2
            char_button = {'char': char,
                           'text': char,
                           'thickness': thickness,
                           'bbox': bbox,
                           'text_pos': (text_x, text_y),
                           'font_size': char_font_size,
                           'mouseover': False,
                           'color': self._LAYOUT['bbox_color']}

            self._char_buttons[char] = char_button

    def push_button(self, char):

        if char in self.char_states:
            self.char_states[char] = not self.char_states[char]
        elif char == 'All':
            self.char_states = {c: True for c in self.chars}
        elif char == 'None':
            self.char_states = {c: False for c in self.chars}
        elif char == 'Good':
            self.char_states = {c: (c in GOOD_CHAR_SET) for c in self.chars}
        self.app.char_set = [c for c in self.chars if self.char_states[c]]
        logging.info("CharsetWindow: toggled '%s', now %i chars:", char, len(self.app.char_set))
        logging.info("\t%s", self.app.char_set)
        self.app.windows['ctrl_window'].widgets['run'].set_visible(True)
        self.app.clear_preprocessing()

    def _draw(self, image):

        def _draw_box(button, color, text_color):
            if color is not None:
                draw_bbox(image, button['bbox'], color=color, thickness=button['thickness'], inside=True)
            if button['text'] is not None and text_color is not None:
                cv2.putText(image, button['text'], button['text_pos'],
                            cv2.FONT_HERSHEY_COMPLEX, button['font_size'], text_color, button['thickness'], cv2.LINE_AA)

        for button_text, button in self._buttons.items():
            color = button['color'] if not button['mouseover'] else self._LAYOUT['mouseover_color']
            text_color = self._LAYOUT['text_color'] if not button['mouseover'] else self._LAYOUT['mouseover_color']
            _draw_box(button, color, text_color)

        for _, char_button in self._char_buttons.items():

            if not self.char_states[char_button['char']]:
                text_color = self._LAYOUT['unused_color']
                color = self._LAYOUT['unused_color']
            else:
                text_color = self._LAYOUT['text_color']
                color = self._LAYOUT['text_color']

            text_color = text_color if not char_button['mouseover'] else self._LAYOUT['mouseover_color']
            color = None if not char_button['mouseover'] else self._LAYOUT['mouseover_color']

            _draw_box(char_button, color, text_color)

        return image

    def _update_mouseover(self, x, y):
        """
        Update internal button states, return new mouse over button or None
        """
        def _check_button(bbox):
            return bbox['x'][0] <= x <= bbox['x'][1] and bbox['y'][0] <= y <= bbox['y'][1]

        new_mouse_over_text = None

        for button in self._buttons.values():
            if _check_button(button['bbox']):
                #logging.info("Mouse over button: %s", button['text'])
                button['mouseover'] = True
                new_mouse_over_text = button['text']
            else:
                button['mouseover'] = False

        if new_mouse_over_text is None:
            for char_button in self._char_buttons.values():
                if _check_button(char_button['bbox']):
                    #logging.info("Mouse over char button: %s", char_button['text'])
                    char_button['mouseover'] = True
                    new_mouse_over_text = char_button['text']
                else:
                    char_button['mouseover'] = False

        return new_mouse_over_text

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            new_mouse_over_text = self._update_mouseover(x, y)
            if new_mouse_over_text is not None and (self._mouse_over_text is None or new_mouse_over_text != self._mouse_over_text):
                if new_mouse_over_text in self._char_buttons and self._mouse_down:
                    self.push_button(new_mouse_over_text)
            self._mouse_over_text = new_mouse_over_text

        elif event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_down = True
            if self._mouse_over_text in self._buttons or self._mouse_over_text in self._char_buttons:
                self.push_button(self._mouse_over_text)
        elif event == cv2.EVENT_LBUTTONUP:
            self._mouse_down = False


class ResultsWindow(ClusterWindow):
    """
    Show the clusters sorted by size, w/in each cluster sorted by distance from cluster mean.
    When clusters are showing, user selects each cluster to include by clicking on it, how much
    of each cluster to include by dragging up/down.  User can un-select by dragging to the left.
    Selected clusters are outlined in red, shaded proportionately.
    """
    _LAYOUT = {'indent_px': 10,
               'max_samples_per_cluster': 15*15,
               'thickness': 2
               }

    def __init__(self, app, bbox_rel=None):
        super().__init__(app, bbox_rel)
        self._assignments = None
        self._distances = None
        self._train_vec_info = None
        self._img = None
        self._bbox = None
        self._bboxes = None  # list of bboxes for each cluster
        self._fg_color = np.array(COLORS['DARK_NAVY_RGB']).reshape((1, 1, 3))
        self._bkg_color = np.array(COLORS['OFF_WHITE_RGB']).reshape((1, 1, 3))
        self._clusters = []  # each is dict for each cluster:  bbox, is_selected, image

        self._mouseover_ind = None
        self._held_ind = None
        self._click_xy = None

        self._fractions = None

    def update_results(self, assignments, distances, train_vec_info):
        print("_---------results window: update_results()")
        
        self._labels = np.unique(assignments)
        self._assignments = assignments
        self._distances = distances
        self._train_vec_info = train_vec_info
        self._clusters = [{'bbox': None, 'label': lab, 'is_selected': False, 'image': None} for lab in self._labels]
        self._fractions = np.zeros(len(self._labels))  # how much of each cluster to select
        if self._bbox is not None or self._bboxes is None:
            self.redraw()


    def get_selection(self):
        """
        For each selected cluster return a dict with:
            'font_inds': indices into the training set for fonts in this cluster
            'font_dists': list of distances for each font in 'font_inds'
            'frac': selected fraction used
        for clusters with frac<1.0, use the most distant fonts first

        """
        selection = []
        if self._assignments is None:
            return selection

        for ind, cluster in enumerate(self._clusters):
            # if ind==len(self._clusters)-1:
            label = self._cluster_labels[ind]
            if cluster['is_selected']:

                font_inds = np.where(self._assignments == label)[0].tolist()
                font_dists = self._distances[self._assignments == label].tolist()
                frac = self._fractions[ind]
                font_names = [self._train_vec_info['font_names'][i] for i in font_inds]
                if frac < 1.0:
                    sorted_inds = np.argsort(font_dists)[::-1]
                    if len(sorted_inds) == 0:
                        continue
                    n_select = max(1, int(len(sorted_inds) * frac))
                    font_inds = [font_inds[i] for i in sorted_inds[:n_select]]
                    font_dists = [font_dists[i] for i in sorted_inds[:n_select]]
                    font_names = [font_names[i] for i in sorted_inds[:n_select]]
                    # Adjust the fraction to account for the selected fonts
                    
                    frac = n_select / len(sorted_inds)
                selection.append({'font_inds': font_inds,
                                  'font_dists': font_dists,
                                  'font_names': font_names,
                                  'frac': frac})
        # pprint.pprint(selection)
        return selection

    def redraw(self):

        w, h = self._bbox['x'][1]-self._bbox['x'][0], self._bbox['y'][1]-self._bbox['y'][0]
        aspect = w/h if h > 0 else 1.0
        image, bboxes, labels = make_assign_gallery(size=(w, h), n_max=200,
                                                    tiles=np.array(self._train_vec_info['disp_icons']),
                                                    distances=self._distances,
                                                    assignments=self._assignments,
                                                    aspect=aspect,
                                                    bgk_color=self._bkg_color)

        if image.shape[0] != h or image.shape[1] != w:
            bboxes = [scale_bbox(bbox, (w/image.shape[1], h/image.shape[0])) for bbox in bboxes]
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        self._bboxes = bboxes
        self._img = image
        self._cluster_labels = labels

    def _update_mouseover(self, x, y):
        self._mouseover_ind = None
        for ind, cluster in enumerate(self._clusters):
            bbox = self._bboxes[ind]
            if bbox['x'][0] <= x <= bbox['x'][1] and bbox['y'][0] <= y <= bbox['y'][1]:
                self._mouseover_ind = ind
                break
        return self._mouseover_ind

    def translate(self, x, y):
        if self._bbox is None:
            return (x, y)
        return (x - self._bbox['x'][0], y - self._bbox['y'][0])

    def on_mouse(self, event, x, y, flags, param):
        x, y = self.translate(x, y)
        y_scale = 150  # pixels per 100 %
        changed=False
        self._update_mouseover(x, y)
        if event == cv2.EVENT_LBUTTONDOWN and self._mouseover_ind is not None:
            self._click_xy = (x, y)
            self._held_ind = self._mouseover_ind
            self._clusters[self._held_ind]['is_selected'] = True  # not self._clusters[self._held_ind]['is_selected']
            changed=True
            self._fractions[self._held_ind] = 0.5
            logging.info("Selecting cluster %i, setting fraction to %.2f", self._cluster_labels[self._held_ind], self._fractions[self._held_ind])
        elif event == cv2.EVENT_MOUSEMOVE and self._held_ind is not None and self._click_xy is not None:
            dy = (self._click_xy[1] - y) / y_scale
            frac = np.clip(dy, 0, 1)
            self._fractions[self._held_ind] = frac
            logging.info("Setting cluster %i fraction to %.2f", self._cluster_labels[self._held_ind], frac)
            changed=True

        elif event == cv2.EVENT_LBUTTONUP:
            if self._held_ind is not None:
                if self._fractions[self._held_ind] == 0:
                    self._clusters[self._held_ind]['is_selected'] = False
                    logging.info("Un-selecting cluster %i", self._held_ind)
                    changed=True
            self._held_ind = None
            self._click_xy = None
        if changed:
            self.app.windows['dataset_window'].update_dataset()

    def resize(self, size):
        left, right = int(self._bbox_rel['x'][0] * size[0]), int(self._bbox_rel['x'][1] * size[0])
        top, bottom = int(self._bbox_rel['y'][0] * size[1]), int(self._bbox_rel['y'][1] * size[1])
        width, height = right - left, bottom - top
        self._bbox = {'x': (left, right),
                      'y': (top, bottom)}
        self._aspect = width / height if height > 0 else 0
        if self._assignments is not None:
            self.redraw()

    def _draw(self, image):
        if self._assignments is None:
            draw_bbox(image, self._bbox, 3, COLORS['DARK_NAVY_RGB'])
        else:
            if self._img is not None:
                image[self._bbox['y'][0]:self._bbox['y'][1], self._bbox['x'][0]:self._bbox['x'][1]] = self._img
            for ind, cluster in enumerate(self._clusters):
                bbox = self._bboxes[ind]
                bbox = {'x': (bbox['x'][0]+self._bbox['x'][0], bbox['x'][1]+self._bbox['x'][0]+1),
                        'y': (bbox['y'][0]+self._bbox['y'][0], bbox['y'][1]+self._bbox['y'][0]+1)}
                color = None
                if self._mouseover_ind is not None and ind == self._mouseover_ind:
                    color = COLORS['NEON_GREEN']
                if cluster['is_selected'] or (self._held_ind is not None and ind == self._held_ind):
                    color = COLORS['DARK_RED_RGB'] if (
                        self._held_ind is not None and ind == self._held_ind) else COLORS['NEON_BLUE']
                    frac_pct_str = f"{int(self._fractions[ind]*100)}%"
                    text_x = bbox['x'][0] + 5
                    size = .5
                    (w, h), b = cv2.getTextSize(frac_pct_str, cv2.FONT_HERSHEY_SIMPLEX, size, 1)

                    text_y = bbox['y'][1] - 7 - h
                    image[text_y:text_y+h+b, text_x:text_x+w+2] = COLORS['OFF_WHITE_RGB']
                    cv2.putText(image, frac_pct_str, (text_x, text_y+h),
                                cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)
                if color is not None:
                    draw_bbox(image, bbox, color=color, thickness=3, inside=True)
        return image


class StatusWindow(ClusterWindow):
    """
    Print stats from the clustering algorithm.
    """

    def __init__(self, app, bbox_rel=None):
        super().__init__(app, bbox_rel)
        self._frame_out = None
        self._bkg_color = COLORS['OFF_WHITE_RGB']
        self._draw_color = COLORS['DARK_NAVY_RGB']

    def resize(self, size):
        left, right = int(self._bbox_rel['x'][0] * size[0]), int(self._bbox_rel['x'][1] * size[0])
        top, bottom = int(self._bbox_rel['y'][0] * size[1]), int(self._bbox_rel['y'][1] * size[1])
        width, height = right - left, bottom - top
        self._bbox = {'x': (left, right),
                      'y': (top, bottom)}
        self._aspect = width / height if height > 0 else 0
        self._frame_out = np.zeros((height, width, 3), dtype=np.uint8)
        self._frame_out[:] = self._bkg_color

    def on_mouse(self, event, x, y, flags, param):
        pass

    def _draw(self, image):
        self.app.stats_artist.draw_stats(image, self._bbox, color=self._draw_color)
        return image


class DatasetWindow(StatusWindow):
    """
    Show stats about the selected clusters (saved as the dataset when user clicks "Save"):
        - n_clusters selected
        - n_fonts selected
        - n_train_samples with the "good" character set
        - n_train_samples with all characters.

    """
    def __init__(self, app, bbox_rel=None):
        super().__init__(app, bbox_rel)
        self._selection = None
    
    def update_dataset(self):
        self._selection = self.app.windows['results_window'].get_selection()
        total_fonts = sum([len(s['font_inds']) for s in self._selection]) if self._selection is not None else 0
        logging.info("DatasetWindow: %i clusters selected, %i fonts total.", len(self._selection) if self._selection is not None else 0, total_fonts)

    def _draw(self, image):
        
        if self._selection is not None:
            lines = ['n clust: %i' % len(self._selection),
                     'n fonts: %i' % sum([len(s['font_inds']) for s in self._selection]),
                     'DS-S size: %i' % (sum([len(s['font_inds']) for s in self._selection]) * len(self.app.char_set)),
                     'DS-G size: %i' % (sum([len(s['font_inds']) for s in self._selection]) * len(GOOD_CHAR_SET)),
                     'DS-A size: %i' % (sum([len(s['font_inds']) for s in self._selection]) * 94)]
            # print font names in the last cluster (highest label)
            # if len(sel)>0:
            #     print("Font names in last cluster (%i):"%len(sel[-1]['font_names']))
            #     for font_name in sel[-1]['font_names']:
            #         print(" - %s"%font_name)

        else:
            lines = ['Select clusters']
        write_lines(image, self._bbox, lines, pad_px=5, color=self._draw_color)
        return image

    def _draw_chars(self, selection):
        """
        Make an image showing all characters in the training set.
        """
        chars = self.app.char_set
        tw, th = 28, 28
        n_cols = len(chars)
        n_rows = sum([len(s['font_inds']) for s in selection])
        indent_x = 0
        img = np.ones((n_rows*th, n_cols*tw+indent_x, 3), dtype=np.uint8)*255
        img[:, :] = self._bkg_color
        font_names = [fn for cluster_info in selection for fn in cluster_info['font_names']]
        logging.info("Drawing %i fonts with up to %i chars each." % (len(font_names), len(chars)))
        for row, font_name in enumerate(font_names):
            char_mask = self.app.data.font_names_train == font_name
            x_train = self.app.data.x_train[char_mask]
            y_train = self.app.data.labels_train[char_mask]
            for col in range(n_cols):
                ind = np.where(y_train == chars[col])[0]
                if len(ind) == 0:
                    continue
                tile = x_train[ind].reshape((28, 28, 1))
                tile_image = (self._draw_color * tile + self._bkg_color * (1-tile)).astype(np.uint8)
                x0, y0 = indent_x + col*tw, row*th
                x1, y1 = x0+tw, y0+th
                img[y0:y1, x0:x1] = tile_image
        return img

    def save(self):
        filename = "font_set.json"
        sel = self.app.windows['results_window'].get_selection()
        out_data = {'char_set': self.app.char_set,
                    'clusters': sel}

        image = self._draw_chars(sel)


        with open(filename, 'w') as f:
            json.dump(out_data, f)
        img_filename = "font_chars.png"
        cv2.imwrite(img_filename, image[:, :, ::-1])
        logging.info("Saved dataset (%i fonts, %i chars each) to %s and %s",
                     len(sel), len(self.app.char_set), filename, img_filename)


class FontClusterApp(object):
    """
    +-----------+---------------------------+
    |  Controls |     Cluster Results       |
    |           |                           |
    |           |                           |
    |           |                           |
    +-----------+                           |
    |  status   |                           |
    |           |                           |
    +------+----+---------------------------+
    |select|                                |
    | info |    Charset Window              |
    +---------------------------------------+

    """

    def __init__(self, size=(1600, 950)):

        self.data = AlphaNumericMNISTData(use_good_subset=False, test_train_split=0.0)
        self.disp_font_ind = 0
        self.char_set = []
        self.size = size
        self._last_pca_dims = -1
        # self._pca = PCA()
        # self._clusters = None
        self.clust_alg = None
        self.sim_graph = None
        self.windows = {'cs_window': CharsetWindow(self, bbox_rel={'x': (.14, 1.0), 'y': (.8, 1.0)}),
                        'ctrl_window': ControlWindow(self, bbox_rel={'x': (.01, .14), 'y': (.02, .5)}),
                        'results_window': ResultsWindow(self, bbox_rel={'x': (.15, 1.0), 'y': (.02, .8)}),
                        'status_window': StatusWindow(self, bbox_rel={'x': (.01, .14), 'y': (.51, .864)}),
                        'dataset_window': DatasetWindow(self, bbox_rel={'x': (.01, .14), 'y': (.865, 1.0)})
                        }
        self._bkg_color = COLORS['OFF_WHITE_RGB']
        self._draw_color = COLORS['DARK_NAVY_RGB']
        self.win_name = "Character Set"
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_name, self.size[0], self.size[1])
        cv2.setMouseCallback(self.win_name, self.on_mouse)
        self.x_train = None

    def vec_to_tile(self, vec):

        tile = vec.reshape((28, 28, 1))
        tile_image = (self._draw_color * tile + self._bkg_color * (1-tile)).astype(np.uint8)
        return tile_image

    def update_k(self, k):
        if self.clust_alg is not None:
            self.clust_alg.set_k(k)

    def clear_preprocessing(self):
        self.x_train = None
        self._last_pca_dims = -1

    def update_preprocessing(self, params):
        logging.info("Preprocessing...") # with params:\n%s", pprint.pformat(params))
        self.x_train, self.fonts_train, self.train_vec_info = self._preprocess(params)
        self.refresh_alg()

    def recluster(self):
        self.refresh_alg()
        params = self.windows['ctrl_window'].get_params()
        #logging.info("Clustering with params:\n%s", pprint.pformat(params))
        # TODO:  implement clustering
        if self.x_train is None:
            self.update_preprocessing(params)
        self._assignments, self._distances = self._cluster(self.x_train)
        print("-------------------------> ",self._assignments.min(), self._assignments.max())
        # DEBUG
        # test_lab = np.max(self._assignments)-1
        # test_mask = self._assignments == test_lab
        # test_imgs = np.array(self.train_vec_info['disp_icons'])[test_mask]
        # test_img = make_digit_mosaic(test_imgs, 1.0, bkg=0)
        # print("Test cluster (%i) has %i fonts, showing %i." % (test_lab, test_mask.sum(), len(test_imgs)))
        # cv2.imshow("Test", test_img[:, :, ::-1])
        # print("Test cluster fonts:", np.array(self.fonts_train)[test_mask])
        # END DEBUG
        logging.info("\tComplete, found %i clusters.", len(np.unique(self._assignments)))

        self.windows['results_window'].update_results(self._assignments, self._distances, self.train_vec_info)
        self.windows['dataset_window'].update_dataset()

    def update_params(self, params):
        params = self.windows['ctrl_window'].get_params()
        alg = params['alg']
        if alg == 'KMeans':
            if params['k_slider'] != self.clust_alg.k or params['dist_metric_name'] != self.clust_alg.which:
                self.refresh_alg()
        elif alg == 'Spectral':
            sim_graph_name = 'KNN'
            if not isinstance(self._sim_graph, SIM_GRAPHS[sim_graph_name]['type']):
                raise Exception("This shouldn't be called for alg changes.")
            else:
                self._sim_graph.set_param(**self._get_sim_graph_params(params))

            self.clust_alg.set_k(params['k_slider'])

    def _get_sim_graph_params(self, params):
        sim_graph_name = 'KNN'
        knn_k = params['knn_k']
        knn_mutual = params['knn_mutual_toggle']
        distance_metric = params['dist_metric_name']
        dist_metric = [k for k in DISTANCE_LABELS.keys() if DISTANCE_LABELS[k] == distance_metric][0]
        sim_graph_params = {'k': knn_k, 'mutual': knn_mutual, 'distance_metric': dist_metric}
        return sim_graph_params

    def refresh_alg(self):
        params = self.windows['ctrl_window'].get_params()
        alg = params['alg']
        k = params['k_slider']
        dist_metric_name = params['dist_metric_name']
        # reverse-lookup to get the metric:
        try:
            dist_metric = [k for k in DISTANCE_LABELS.keys() if DISTANCE_LABELS[k] == dist_metric_name][0]
        except:
            raise ValueError("Unknown distance metric name: %s" % dist_metric_name)

        if alg == 'KMeans':
            self.clust_alg = KMeansAlgorithm(k=k, distance_metric=dist_metric, n_init=1, max_iter=1000)
            self._sim_graph = None
            self.stats_artist = self.clust_alg

        elif alg == 'Spectral':
            sim_graph_name = 'KNN'
            sim_graph_params = self._get_sim_graph_params(params)

            self._sim_graph = SIM_GRAPHS[sim_graph_name]['type'](**sim_graph_params)
            self.clust_alg = SpectralAlgorithm(k=k)
            self.stats_artist = self._sim_graph
        elif alg == 'DBScan-auto':
            min_samples = params['dbscan_min_size']
            self.clust_alg = DBScanAlgorithm(min_nn_samples=min_samples,
                                             metric=dist_metric)
            self._sim_graph = None
            self.stats_artist = self.clust_alg
        elif alg == 'DBScan':
            eps = params['dbscan_eps']
            min_samples = params['dbscan_min_size']
            self.clust_alg = DBScanManualAlgorithm(epsilon_rel=eps,
                                                   min_nn_samples=min_samples,
                                                   metric=dist_metric)
            self._sim_graph = None
            self.stats_artist = self.clust_alg
        else:
            raise ValueError("Unknown algorithm name: %s" % alg)

    def _cluster(self, x_train):
        logging.info("Clustering %i samples...", x_train.shape[0])
        if self._sim_graph is not None:
            self._sim_graph.fit(x_train)  # already update in update_params
            img = self._sim_graph.make_img()
            # cv2.imshow("Similarity Graph", img)
            # cv2.waitKey(0)
            # print(self._sim_graph._get_graph_stats())
            data = self._sim_graph
        else:
            data = x_train
        self.clust_alg.fit(data, verbose=True)
        assignments, distances = self.clust_alg.assign(self.x_train)

        return assignments, distances

    def _make_img(self, train_vecs):
        width = 28 * len(self.char_set)
        height = 28 * len(train_vecs['font_names'])
        img = np.zeros((height, width), dtype=np.uint8)
        img[:] = 255

        for f_ind, font in enumerate(train_vecs['font_names']):
            for c_ind, char in enumerate(self.char_set):
                data_ind = c_ind*28 * 28
                img[f_ind * 28:(f_ind + 1) * 28, c_ind * 28:(c_ind + 1) * 28] = (train_vecs['font_data']
                                                                                 [f_ind][data_ind:data_ind+28*28].reshape((28, 28)) * 255).astype(np.uint8)
        return img

    def _preprocess(self, params):
        """
        Step 1, subset characters, 
        step 2  convert to "font vectors":
           For each font with all selected chars represented, concatenate the character images into a single vector.
        Step 3  apply PCA (if pca_dims > 0).

        Also:
           * Make images for representatives of each font for displaying the clustering result. 
             (The image is the first character in the selected set.)
        """
        # step 1,2, make vectors
        train_vecs = {'font_names': [],
                      'font_data': [],
                      'disp_icons': []}
        fonts = np.sort(np.unique(self.data.font_names_train))
        logging.info("Data has %i fonts.", len(fonts))
        chars = np.sort(self.char_set)
        char_mask = np.isin(self.data.labels_train, chars)
        n_chars = len(chars)
        logging.info("Using %i chars, %i total samples exist.", n_chars, char_mask.sum())

        for font in fonts:
            font_mask = (self.data.font_names_train == font) & char_mask
            font_labels = self.data.labels_train[font_mask]
            if n_chars == font_labels.size:
                order = np.argsort(font_labels)
                font_data = self.data.x_train[font_mask][order]
                icon_data = font_data[0].reshape((28, 28, 1))
                train_vecs['disp_icons'].append(self.vec_to_tile(icon_data))
                train_vecs['font_names'].append(font)
                train_vecs['font_data'].append(font_data.flatten())

        test_img = self._make_img(train_vecs)
        cv2.imwrite("test_vecs.png", test_img)
        logging.info("Wrote test vector image with %i samples to 'test_vecs.png'", len(train_vecs['font_names']))

        logging.info("After subsetting, %i fonts have all %i chars.", len(train_vecs['font_names']), len(self.char_set))

        x_train = np.array([fdata.flatten() for fdata in train_vecs['font_data']])
        fonts_train = np.array(train_vecs['font_names'])

        logging.info("\tX-train shape: %s", x_train.shape)
        logging.info("\tFonts shape: %s", fonts_train.shape)

        # step 3, PCA
        pca_dims = params['pca_dims']
        if pca_dims >= 0 and pca_dims != self._last_pca_dims:
            logging.info("Applying PCA, dims=%i", pca_dims)
            pca = PCA(dims=params['pca_dims'])
            x_train = pca.fit_transform(x_train)
        elif pca_dims == self._last_pca_dims:
            logging.info("Using cached PCA, dims=%i", pca_dims)
            x_train = self.x_train
            fonts_train = self.fonts_train
        else:
            raise ValueError("pca_dims must be >= 0, <= 784")

        self._last_pca_dims = pca_dims
        return x_train, fonts_train, train_vecs

    def run(self):

        while True:
            try:
                x, y, current_width, current_height = cv2.getWindowImageRect(self.win_name)
            except cv2.error:
                logging.info("Window closed, exiting...")
                break
            if (current_width, current_height) != self.size:
                self.size = (current_width, current_height)
                for _, window in self.windows.items():
                    window.resize(self.size)

            frame = self.make_blank()
            for _, window in self.windows.items():
                frame = window.draw(frame)
            cv2.imshow(self.win_name, frame[:, :, ::-1])
            k = cv2.waitKey(10)
            if k == 27:
                break
        logging.info("Exiting...")

    def save_clusters(self):
        self.windows['dataset_window'].save()

    def make_blank(self):
        blank = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        blank[:] = COLORS['OFF_WHITE_RGB']
        return blank

    def on_mouse(self, event, x, y, flags, param):
        for _, window in self.windows.items():
            window.on_mouse(event, x, y, flags, param)


def run_app():
    app = FontClusterApp().run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_app()
