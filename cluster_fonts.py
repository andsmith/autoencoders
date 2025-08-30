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
        - Outlier filtering method:
            - remove small clusters (below a threshold size) & re-cluster up to N times.
        - Algorithms:
            - k-means, params:
                - Distance metric:  cosine, euclidean
            - agglomerative, params:
                - Distance metric:  cosine, euclidean
                - Linkage:  ward, complete, average
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
from sklearn.cluster import KMeans, AgglomerativeClustering
from mnist import AlphaNumericMNISTData
from util import fit_spaced_intervals, draw_bbox, get_font_size


class CharsetWindow(object):
    _LAYOUT = {'indent_px': 28,
               'x_div_rel': .2,
               'char_spacing_frac': 0.1,
               'button_spacing_frac': 0.15,
               'bbox_color': COLORS['DARK_NAVY_RGB'],
               'text_color': COLORS['DARK_NAVY_RGB'],
               'unused_color': COLORS['SKY_BLUE'],
               'mouseover_color': COLORS['DARK_RED_RGB']}

    def __init__(self, app, size=(1000, 300), shape=[5, 20]):
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
        self.chars = sorted(np.unique(app.data.labels_train))
        self.char_states = {char: (char in app.char_set) for char in self.chars}
        self.app = app
        self.nrows, self.ncols = shape
        self._mouse_down = False
        self._mouse_over = None
        self.set_size(size)
        logging.info("CharsetWindow:  %i chars, %.2f %% on, shape %s", len(self.chars),
                     100 * sum(self.char_states.values()) / len(self.char_states), shape)

    def set_size(self, new_size):
        """
        Get layout of window.
        :returns:
            buttons:  list of dicts with {'text': <button_text>
                                          'bbox': ((x_min, x_max),( y_min, y_max))}
            char_buttons:  same list ,for each of the character buttons
        """
        self.size = new_size

        ind = self._LAYOUT['indent_px']
        x_mid = int(self.size[0] * self._LAYOUT['x_div_rel'])

        # multi-char buttons:
        button_left, button_right = ind, x_mid - ind
        button_top, button_bottom = ind, self.size[1] - ind
        button_y = fit_spaced_intervals((button_top, button_bottom), 3, self._LAYOUT['button_spacing_frac'],
                                        fill_extent=False)
        button_x = (button_left, button_right)
        button_h, button_w = button_y[0][1] - button_y[0][0], button_x[1] - button_x[0]
        bpad = min(14, button_h // 4)
        button_font_size, xy_rel, thickness = get_font_size('Good', (button_w, button_h),
                                                            incl_baseline=False, pad=bpad)
        self._buttons = [{'text': 'All',
                          'bbox': {'x': button_x, 'y': button_y[0]},
                          'text_pos': (button_x[0] + xy_rel[0], button_y[0][0] + xy_rel[1])},
                         {'text': 'None',
                          'bbox': {'x': button_x, 'y': button_y[1]},
                          'text_pos': (button_x[0] + xy_rel[0], button_y[1][0] + xy_rel[1])},
                         {'text': 'Good',
                          'bbox': {'x': button_x, 'y': button_y[2]},
                          'text_pos': (button_x[0] + xy_rel[0], button_y[2][0] + xy_rel[1])}]

        for button in self._buttons:
            button['mouseover'] = False
            button['font_size'] = button_font_size
            button['color'] = self._LAYOUT['bbox_color']
            button['thickness'] = thickness

        # Character buttons
        char_button_left, char_button_right = button_right + ind, self.size[0] - ind
        char_button_top, char_button_bottom = button_top, button_bottom
        char_button_x = fit_spaced_intervals((char_button_left, char_button_right), self.ncols, self._LAYOUT['char_spacing_frac'],
                                             fill_extent=False)
        char_button_y = fit_spaced_intervals((char_button_top, char_button_bottom), self.nrows, self._LAYOUT['char_spacing_frac'],
                                             fill_extent=False)
        char_button_h = char_button_y[0][1] - char_button_y[0][0]
        char_button_w = char_button_x[0][1] - char_button_x[0][0]
        cpad = min(4, char_button_h // 3)

        char_font_size, xy_rel, thickness = get_font_size("#", (char_button_w, char_button_h),
                                                          incl_baseline=False, pad=cpad)

        self._char_buttons = []
        for c_ind, char in enumerate(self.chars):

            col = c_ind % self.ncols
            row = c_ind // self.ncols
            bbox = {'x': char_button_x[col], 'y': char_button_y[row]}
            (width, height), baseline = cv2.getTextSize(str(char), cv2.FONT_HERSHEY_COMPLEX, char_font_size, thickness)
            text_x = (char_button_x[col][0] + char_button_x[col][1]) // 2 - width // 2
            text_y = (char_button_y[row][0] + char_button_y[row][1]) // 2  + height // 2
            char_button = {'char': char,
                           'text': char,
                           'thickness': thickness,
                           'bbox': bbox,
                           'text_pos': (text_x, text_y),
                           'font_size': char_font_size,
                           'mouseover': False,
                           'color': self._LAYOUT['bbox_color']}
            self._char_buttons.append(char_button)

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

    def draw(self, image):

        if image.shape[0] != self.size[1] or image.shape[1] != self.size[0]:
            self.set_size((image.shape[1], image.shape[0]))

        def _draw_box(button, color, text_color):
            if color is not None:
                draw_bbox(image, button['bbox'], color=color, thickness=button['thickness'], inside=True)
            if button['text'] is not None and text_color is not None:
                cv2.putText(image, button['text'], button['text_pos'],
                            cv2.FONT_HERSHEY_COMPLEX, button['font_size'], text_color, button['thickness'], cv2.LINE_AA)

        for button in self._buttons:
            color = button['color'] if not button['mouseover'] else self._LAYOUT['mouseover_color']
            text_color = self._LAYOUT['text_color'] if not button['mouseover'] else self._LAYOUT['mouseover_color']
            _draw_box(button, color, text_color)

        for char_button in self._char_buttons:

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

        new_mouse_over = None

        for button in self._buttons:
            if _check_button(button['bbox']):
                logging.info("Mouse over button: %s", button['text'])
                button['mouseover'] = True
                new_mouse_over = button
            else:
                button['mouseover'] = False

        if new_mouse_over is None:
            for char_button in self._char_buttons:
                if _check_button(char_button['bbox']):
                    logging.info("Mouse over char button: %s", char_button['text'])
                    char_button['mouseover'] = True
                    new_mouse_over = char_button
                else:
                    char_button['mouseover'] = False
        return new_mouse_over

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            new_mouse_over = self._update_mouseover(x, y)
            if new_mouse_over is not None and new_mouse_over != self._mouse_over:
                if new_mouse_over in self._char_buttons and self._mouse_down:
                    self.push_button(new_mouse_over['text'])
            self._mouse_over = new_mouse_over

        elif event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_down = True
            if self._mouse_over in self._buttons or self._mouse_over in self._char_buttons:

                self.push_button(self._mouse_over['text'])
        elif event == cv2.EVENT_LBUTTONUP:
            self._mouse_down = False


class FakeApp(object):
    """

    """

    def __init__(self):
        self.data = AlphaNumericMNISTData(use_good_subset=False, test_train_split=0.0)
        self.disp_font_ind = 0
        self.char_set = GOOD_CHAR_SET


def test_charset_window():
    app = FakeApp()
    window = CharsetWindow(app)
    win_name = "Character Set"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, window.size[0], window.size[1])
    cv2.setMouseCallback(win_name, window.on_mouse)

    def make_blank(size):
        blank = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        blank[:] = COLORS['OFF_WHITE_RGB']
        return blank

    while True:
        try:
            x, y, current_width, current_height = cv2.getWindowImageRect(win_name)
        except:
            logging.info("Window closed.")
            break

        frame = window.draw(make_blank((current_width, current_height)))

        cv2.imshow(win_name, frame[:, :, ::-1])
        k = cv2.waitKey(10)
        if k == 27:
            break
    logging.info("Exiting...")


class FontClusterApp(object):
    def __init__(self):
        self.data = AlphaNumericMNISTData(use_good_subset=False, test_train_split=0.0)
        self.disp_font_ind = 0
        self.char_set = GOOD_CHAR_SET
        self._pca = PCA()
        self._clusters = None

        self.charset_window = CharsetWindow(self)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_charset_window()
