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
                    - Epsilon (binary, within max distance threshold)
                    - KNN (k-nearest neighbors), shared-only / inclusive
                    - Full, using gaussian distance function with parameter sigma.
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
import os
import sys
import argparse
import pprint
from sklearn.cluster import KMeans, AgglomerativeClustering
from mnist import AlphaNumericMNISTData
from util import fit_spaced_intervals, draw_bbox, get_font_size
from abc import ABC, abstractmethod
from tools import RadioButtons, Slider, Button, ToggleButton


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


def split_bbox(bbox, weight=0.5, orient='v'):
    """
    Split a bounding box into two parts.
    :param bbox: The bounding box to split.
    :param weight: The relative weight of the first part.
    :param orient: The orientation of the split ('v' for vertical, 'h' for horizontal).
    :return: Two bounding boxes representing the split.
    """
    if orient == 'v':
        # Vertical split
        (x1, x2), (y1, y2) = bbox['x'], bbox['y']
        split_x = int(x1 + (x2 - x1) * weight)
        rv = {'x': (x1, split_x), 'y': (y1, y2)}, {'x': (split_x, x2), 'y': (y1, y2)}
    else:
        # Horizontal split
        (x1, x2), (y1, y2) = bbox['x'], bbox['y']
        split_y = int(y1 + (y2 - y1) * weight)
        rv = {'x': (x1, x2), 'y': (y1, split_y)}, {'x': (x1, x2), 'y': (split_y, y2)}
    print("Split", bbox, "into BBoxes", rv, orient)
    return rv


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
            | Sim-graph type:    |
            |   - Epsilon        |
            |   * KNN            |
            |   - Full           |
            |                    |
            | Sim-graph params:  |
            |  +--k-------+ [M]  |  <- updates for selected sim-graph type.
            |                    |
            |   CLUSTER FONTS    |  <- button
            +--------------------+
    """
    _LAYOUT = {'indent_px': 10,
               'area_names': ['alg', 'param', 'sim-graph', 'sim-param', 'action'],
               'area_weights': [2, 3, 3.5, 1.5, 1.],  # 5 vertical areas of the control window
               'pad_weight': .25}
    
    _ALGORITHMS = ['KMeans', 'Spectral']
    _SIMGRAPHS = ['Epsilon', 'KNN', 'Full']

    _PARAM_RANGES = {'K': (2, 42),
                     'knn_k': (1, 30),
                     'epsilon_sigma': (0., 1.),
                     'PCA': (0, 256)}

    def __init__(self, app, bbox_rel=None):
        super().__init__(app, bbox_rel)
        self._widgets = None

    def _init_widgets(self):
        int_fmt_str = ": %i"
        self._widgets = {
            'alg': RadioButtons(self._area_bboxes['alg'], title='Algorithm', callback=self._alg_change,
                                options=self._ALGORITHMS),
            'k_slider': Slider(self._k_slider_bbox, label='N. Clusters', callback=self._param_change, default=2, range=self._PARAM_RANGES['K'], format_str=int_fmt_str),
            'pca_slider': Slider(self._pca_slider_bbox, label='N. PCA Dims', callback=self._param_change, default=100, range=self._PARAM_RANGES['PCA'], format_str=int_fmt_str),
            'whiten_toggle': ToggleButton(self._whiten_toggle_bbox, label='Norm', callback=self._param_change),
            'simgraph-picker': RadioButtons(self._area_bboxes['sim-graph'], title='Similarity Graph', callback=self._alg_change,
                                            options=self._SIMGRAPHS),
            'epsilon_sigma': Slider(self._simgraph_param_full_bbox, label='Epsilon/Sigma', callback=self._param_change, default=0.05, range=self._PARAM_RANGES['epsilon_sigma']),
            'knn_k': Slider(self._simgraph_param_split_bbox, label='KNN K', callback=self._param_change, default=5, range=self._PARAM_RANGES['knn_k'], format_str=int_fmt_str),
            'knn_mutual_toggle': ToggleButton(self._simgraph_toggle_bbox, label='Mutual', callback=self._param_change),

            'action': Button(self._area_bboxes['action'], label='Cluster Fonts', callback=self._run),
        }

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

        # THe slider areas need to be split
        self._k_slider_bbox, temp = split_bbox(self._area_bboxes['param'], weight=0.5, orient='h')
        self._pca_slider_bbox, self._whiten_toggle_bbox = split_bbox(temp, weight=0.66, orient='v')
        self._simgraph_param_split_bbox, self._simgraph_toggle_bbox = split_bbox(
            self._area_bboxes['sim-param'], weight=0.75, orient='v')
        self._simgraph_param_full_bbox = self._area_bboxes['sim-param']

        # initialize or move widgets
        if self._widgets is None:
            self._init_widgets()
            self._alg_change(None)  # for ui consistency
        else:
            self._widgets['alg'].move_to(self._area_bboxes['alg'])
            self._widgets['k_slider'].move_to(self._k_slider_bbox)
            self._widgets['pca_slider'].move_to(self._pca_slider_bbox)
            self._widgets['whiten_toggle'].move_to(self._whiten_toggle_bbox)
            self._widgets['simgraph-picker'].move_to(self._area_bboxes['sim-graph'])
            self._widgets['epsilon_sigma'].move_to(self._simgraph_param_full_bbox)
            self._widgets['knn_k'].move_to(self._simgraph_param_split_bbox)
            self._widgets['knn_mutual_toggle'].move_to(self._simgraph_toggle_bbox)
            self._widgets['action'].move_to(self._area_bboxes['action'])

    def get_params(self):
        return {'alg': self._widgets['alg'].get_value(),
                'k_slider': self._widgets['k_slider'].get_value(),
                'pca_slider': self._widgets['pca_slider'].get_value(),
                'whiten_toggle': self._widgets['whiten_toggle'].get_value(),
                'simgraph-picker': self._widgets['simgraph-picker'].get_value(),
                'epsilon_sigma': self._widgets['epsilon_sigma'].get_value(),
                'knn_k': self._widgets['knn_k'].get_value(),
                'knn_mutual_toggle': self._widgets['knn_mutual_toggle'].get_value()}

    def _draw(self, image):
        # draw_bbox(image, self._bbox, 3, COLORS['DARK_NAVY_RGB'])
        # for bbox in self._area_bboxes.values():
        #     draw_bbox(image, bbox, 2, COLORS['DARK_GRAY'])
        # for bbox in [self._k_slider_bbox, self._pca_slider_bbox, self._whiten_toggle_bbox,self._simgraph_param_split_bbox, self._simgraph_toggle_bbox]:
        #     draw_bbox(image, bbox, 1, COLORS['LIGHT_GRAY'])

        for widget_name, widget in self._widgets.items():
            widget.render(image)

        return image

    def _alg_change(self, _):
        alg_name = self._widgets['alg'].get_value()
        if alg_name=='KMeans': 
            self._widgets['simgraph-picker'].set_visible(False)
            self._widgets['epsilon_sigma'].set_visible(False)
            self._widgets['knn_k'].set_visible(False)
            self._widgets['knn_mutual_toggle'].set_visible(False)

        elif alg_name=='Spectral':
            self._widgets['simgraph-picker'].set_visible(True)
            simgraph = self._widgets['simgraph-picker'].get_value()
            if simgraph == 'Epsilon' or simgraph == 'Full':
                self._widgets['epsilon_sigma'].set_visible(True)
                self._widgets['knn_k'].set_visible(False)
                self._widgets['knn_mutual_toggle'].set_visible(False)
            elif simgraph == 'KNN':
                self._widgets['epsilon_sigma'].set_visible(False)
                self._widgets['knn_k'].set_visible(True)
                self._widgets['knn_mutual_toggle'].set_visible(True)
            else:
                raise ValueError("Unknown sim-graph name: %s" % simgraph)
        else:
            raise ValueError("Unknown algorithm name: %s" % alg_name)
        

    def _param_change(self, value):
        print("CHANGING PARAMS:", value)

    def _run(self):
        self.app.update_clusters(self.get_params())

    def on_mouse(self, event, x, y, flags, param):
        for _, widget in self._widgets.items():
            widget.on_mouse(event, x, y, flags, param)


class CharsetWindow(ClusterWindow):
    _LAYOUT = {'indent_px': 28,
               'x_div_rel': .2,
               'char_spacing_frac': 0.1,
               'button_spacing_frac': 0.25,
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
        logging.info("CharsetWindow: toggled '%s', now %i chars on", char, len(self.app.char_set))

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
                logging.info("Mouse over button: %s", button['text'])
                button['mouseover'] = True
                new_mouse_over_text = button['text']
            else:
                button['mouseover'] = False

        if new_mouse_over_text is None:
            for char_button in self._char_buttons.values():
                if _check_button(char_button['bbox']):
                    logging.info("Mouse over char button: %s", char_button['text'])
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


class FontClusterApp(object):
    """
    +-----------+---------------------------+
    |  Controls |     Cluster view          |
    |           |                           |
    |           |                           |
    |           |                           |
    +-----------+                           |
    |  status   |                           |
    |           |                           |
    +-----------+---------------------------+
    |                                       |
    |           Charset Window              |
    +---------------------------------------+

    """

    def __init__(self, size=(1600, 950)):
        self.data = AlphaNumericMNISTData(use_good_subset=False, test_train_split=0.0)
        self.disp_font_ind = 0
        self.char_set = GOOD_CHAR_SET
        self.size = size
        # self._pca = PCA()
        # self._clusters = None

        self.windows = {'cs_window': CharsetWindow(self, bbox_rel={'x': (.1, .9), 'y': (.74, 1.0)}),
                        'ctrl_window': ControlWindow(self, bbox_rel={'x': (.025, .15), 'y': (.02, .56)})}

        self.win_name = "Character Set"
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_name, self.size[0], self.size[1])
        cv2.setMouseCallback(self.win_name, self.on_mouse)

    def update_clusters(self, clustering_params):
        logging.info("Clustering with params:\n%s", pprint.pformat(clustering_params))
        # TODO:  implement clustering

    def run(self):

        while True:

            x, y, current_width, current_height = cv2.getWindowImageRect(self.win_name)
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

    def make_blank(self):
        blank = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        blank[:] = COLORS['OFF_WHITE_RGB']
        return blank

    def on_mouse(self, event, x, y, flags, param):
        for _, window in self.windows.items():
            window.on_mouse(event, x, y, flags, param)


def test_windows():

    app = FontClusterApp().run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_windows()
