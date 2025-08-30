"""
Toolbox sub-window of the cluster editor.
"""

import cv2
import numpy as np
from colors import COLORS
from util import get_font_size, draw_bbox, fit_spaced_intervals
from abc import ABC, abstractmethod

import logging


COLOR_OPTIONS = {'unselected': COLORS['GRAY'],
                 'selected': COLORS['DARK_NAVY_RGB'],
                 'mouseover': COLORS['FOREST_GREEN_RGB'],
                 'idle': COLORS['DARK_NAVY_RGB'],
                 'held': COLORS['NEON_RED'],
                 'tab': COLORS['DARK_GRAY'],
                 'active_toggle': COLORS['LIGHT_GRAY'],
                 'inactive_toggle': COLORS['DARK_NAVY_RGB'],
                 'border': COLORS['DARK_GRAY'], }


def bbox_contains(bbox, x, y):
    return bbox['x'][0] <= x <= bbox['x'][1] and bbox['y'][0] <= y <= bbox['y'][1]


def calc_font_size(lines, bbox, font, indent, incl_baseline=False):
    w, h = bbox['x'][1] - bbox['x'][0], bbox['y'][1] - bbox['y'][0]
    size, pos_xy, thickness = get_font_size(lines[0], (w, h), font=font, pad=indent)
    (width, height), baseline = cv2.getTextSize(lines[0], font, size, thickness)
    return size, (height if not incl_baseline else height + baseline)


class Tool(ABC):
    """
    Abstract class for tools in the cluster creator app.
    """

    def __init__(self, bbox, label, callback=None, visible=True, spacing_px=6):
        """
        Create a tool with the given bounding box.
        :param bbox: {'x': [left, right], 'y': [top, bottom]}
        :param label: Text label for rendering the tool. (i.e. on a button)
        :param visible: Whether the tool is visible initially.
        :param callback: Function to call when the tool is clicked.
        :param spacing_px: Spacing between elements in the tool (text lines, etc)
           NOTE:  The base class does nothing with this, inheriting classes must implement the callback.
        """
        self._bbox = bbox
        self._spacing_px = spacing_px
        self._visible = visible
        self._callback = callback
        self._txt_name = label
        self._calc_dims()

    @abstractmethod
    def _render(self, img):
        """
        Render the tool onto the image.
        """
        pass

    @abstractmethod
    def _mouse_click(self, x, y):
        """
        Handle a mouse click.
        """
        pass

    @abstractmethod
    def _mouse_move(self, x, y):
        """
        Handle a mouse move.
        """
        pass

    @abstractmethod
    def _mouse_unclick(self, x, y):
        """
        Handle a mouse unclick.
        """
        pass

    def on_mouse(self, event, x, y, flags, param):
        inside = bbox_contains(self._bbox, x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if inside:
                return self.mouse_click(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            return self.mouse_move(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if inside:
                return self.mouse_unclick(x, y)

    # above methods will not be called if tool is not visible:
    def render(self, img):
        if self._visible:
            # draw_bbox(img, self._bbox, color=(0,0,0),thickness=3)

            self._render(img)

    def mouse_click(self, x, y):
        if self._visible:
            return self._mouse_click(x, y)

    def mouse_move(self, x, y):
        if self._visible:
            return self._mouse_move(x, y)

    def mouse_unclick(self, x, y):
        if self._visible:
            return self._mouse_unclick(x, y)

    def set_visible(self, visible):
        self._visible = visible

    def move_to(self, bbox):
        """
        Move the tool to the new bbox.
        """
        self._bbox = bbox
        self._calc_dims()

    @abstractmethod
    def _calc_dims(self):
        """
        Calculate the dimensions of the tool after resizing.
        """
        pass

    @abstractmethod
    def get_value(self):
        """
        Return the current value.
        """
        pass


class Slider(Tool):
    """
    Slider, orient='horizontal' looks like this:

        Value = 0.4
        [---|------]

    Vertical:

        V = .4
        -----
          |
          +
          |
          |
          |
        -----



    """

    def __init__(self, bbox, label, callback=None, visible=True, range=(0, 1), default=None, format_str='=%.2f', spacing_px=3, orient='horizontal'):
        """
        Create a slider with the given bounding box.
        :param format_str: Format string for the value display:  label + format_str % (value,) 
        """
        logging.info("Initializing Slider '%s' at position: %s" % (label, bbox))
        self._format_str = format_str
        self._t_horiz_fact = 0.6   # fraction of slider that is for title
        self._t_vert_frac = 0.15
        self._slider_width_px = 10
        self._orient = orient
        self._range = range
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._colors = {c_opt: COLOR_OPTIONS[c_opt] for c_opt in COLOR_OPTIONS}

        self._moused_over = False
        self._held = False

        if default is None:
            self._slider_pos = 0.0  # smallest (biggest is 1.0)
        else:
            self._slider_pos = (default - self._range[0]) / (self._range[1] - self._range[0])
        super().__init__(bbox, label, callback, visible, spacing_px)

    def get_value(self):
        """
        Return the current value.
        """
        return self._range[0] + self._slider_pos * (self._range[1] - self._range[0])

    def set_value(self, val):
        """
        Set the value of the slider.
        """
        self._slider_pos = (val - self._range[0]) / (self._range[1] - self._range[0])

    def _calc_dims(self):

        if self._orient == 'horizontal':
            self._calc_dims_horiz()
        else:
            self._calc_dims_vert()

    def _calc_dims_horiz(self):
        self._y_midline = int((self._bbox['y'][1] * self._t_horiz_fact + self._bbox['y'][0] * (1 - self._t_horiz_fact)))
        # title position
        self._text_bbox = {'x': (self._bbox['x'][0]+self._spacing_px, self._bbox['x'][1]-self._spacing_px),
                           'y': [self._bbox['y'][0]+self._spacing_px, self._y_midline]}

        # test font size with random value
        test_val = self._range[1]
        test_str = self._txt_name + self._format_str % (test_val,)

        self._font_size, _ = calc_font_size([test_str], self._text_bbox, self._font, 0)
        title_dims, baseline = cv2.getTextSize(test_str, self._font, self._font_size, 1)
        left = self._bbox['x'][0] + self._spacing_px
        top = self._bbox['y'][0] + title_dims[1] + self._spacing_px + baseline
        self._title_pos = (left, top)
        # slider dims
        self._s_left = left
        self._s_right = self._bbox['x'][1] - self._spacing_px
        self._s_top = self._y_midline + self._spacing_px
        self._s_bottom = self._bbox['y'][1] - self._spacing_px

        self._slider_bbox = {'x': [self._s_left, self._s_right],
                             'y': [self._s_top, self._s_bottom]}

    def _calc_dims_vert(self):
        self._x_midline = int((self._bbox['x'][1] + self._bbox['x'][0])/2)
        slider_top_y = int((self._bbox['y'][0] + self._spacing_px) *
                           (1 - self._t_vert_frac) + self._bbox['y'][1] * self._t_vert_frac)
        # title position
        self._text_bbox = {'x': [self._bbox['x'][0]+self._spacing_px, self._bbox['x'][1]-self._spacing_px],
                           'y': [self._bbox['y'][0]+self._spacing_px, slider_top_y]}
        # test font size with random value
        test_val = self._range[1]
        test_str = self._txt_name + self._format_str % (test_val,)
        self._font_size, _ = calc_font_size([test_str], self._text_bbox, self._font, 0)
        title_dims, baseline = cv2.getTextSize(test_str, self._font, self._font_size, 1)
        pad = (self._text_bbox['y'][1] - self._text_bbox['y'][0] - title_dims[1]) // 2
        title_y = self._text_bbox['y'][0] + title_dims[1] + pad
        self._title_pos = (self._text_bbox['x'][0] + self._spacing_px, title_y)
        # slider dims
        self._s_left = self._bbox['x'][0] + self._spacing_px
        self._s_right = self._bbox['x'][1] - self._spacing_px
        self._s_top = slider_top_y + self._spacing_px
        self._s_bottom = self._bbox['y'][1] - self._spacing_px
        tab_half_width = int(0.35 * (self._s_right - self._s_left))
        self._slider_tab_x_span = self._x_midline - tab_half_width, self._x_midline + tab_half_width

        self._slider_bbox = {'x': [self._s_left, self._s_right],
                             'y': [self._s_top, self._s_bottom]}

    def _render(self, img):
        """
        Render the slider.
        """
        tab_color = self._colors['tab']
        if self._held:
            tab_color = self._colors['held']
        elif self._moused_over:
            tab_color = self._colors['mouseover']

        def _draw_bbox(bbox, color_name):
            p0 = (bbox['x'][0], bbox['y'][0])
            p1 = (bbox['x'][1], bbox['y'][1])
            cv2.rectangle(img, p0, p1, COLORS[color_name], 1)
        # _draw_bbox(self._bbox, 'red')
        # _draw_bbox(self._text_bbox, 'green')
        # _draw_bbox(self._slider_bbox, 'blue')

        # title
        val = self.get_value()
        slider_str = self._txt_name + self._format_str % (val,)

        cv2.putText(img, slider_str, self._title_pos, self._font, self._font_size, self._colors['idle'], 1, cv2.LINE_AA)
        # slider bar
        if self._orient == 'horizontal':
            slider_y = (self._slider_bbox['y'][0] + self._slider_bbox['y'][1]) // 2
            cv2.line(img, (self._s_left, slider_y), (self._s_right, slider_y), self._colors['idle'], 1)

            # slider tab
            slider_xpos = int(self._slider_pos * (self._s_right - self._s_left)) + \
                self._s_left - self._slider_width_px // 2

            img[self._slider_bbox['y'][0]:self._slider_bbox['y'][1],
                slider_xpos: slider_xpos + self._slider_width_px] = np.array(tab_color, dtype=np.uint8)
        else:
            cv2.line(img, (self._x_midline, self._s_top), (self._x_midline, self._s_bottom), self._colors['idle'], 1)
            # slider tab
            slider_ypos = int(self._slider_pos * (self._s_bottom - self._s_top)) + \
                self._s_top - self._slider_width_px // 2
            img[slider_ypos: slider_ypos + self._slider_width_px,
                self._slider_tab_x_span[0]:self._slider_tab_x_span[1]] = np.array(tab_color, dtype=np.uint8)

    def _mouse_click(self, x, y):
        """
        Check if the click is within the slider.
        """
        if bbox_contains(self._bbox, x, y):
            self._held = True
            self._move_slider(x, y)
            return True

    def _mouse_move(self, x, y):
        """
        Check if the mouse is over the slider.
        """
        self._moused_over = False
        if bbox_contains(self._bbox, x, y):
            self._moused_over = True
        if self._held:
            self._move_slider(x, y)
            return True
        return False

    def _mouse_unclick(self, x, y):
        self._held = False

    def _move_slider(self, x, y):
        """
        Move the slider to the new position.
        """
        old_pos = self._slider_pos
        if self._orient == 'horizontal':
            rel_x = (x - self._s_left) / (self._s_right - self._s_left)
            rel_x = np.clip(rel_x, 0, 1)
            self._slider_pos = rel_x
        else:
            rel_y = (y - self._s_top) / (self._s_bottom - self._s_top)
            rel_y = np.clip(rel_y, 0, 1)
            self._slider_pos = rel_y

        if old_pos != self._slider_pos and self._callback is not None:
            self._callback(self.get_value())


class Button(Tool):
    """
    Rectangular area with text label.
    left-click calls callback function.
    """

    def __init__(self, bbox,  label, callback, visible=True, border_indent=2, spacing_px=4):
        """
        Create a button with a label.
        :param bbox: {'x': [left, right], 'y': [top, bottom]} bounding box of region with button
        :param label: Text label for the button.
        :param callback: Function to call when the button is clicked.
        :param visible: Whether the button is visible initially.
        :param border_indent: Number of pixels to indent the rectangle of the button.
        :param spacing_px: Vertical spacing between elements in the button (text lines, etc),
            and horizontally from the border.
        """
        self._text = label
        self._border_indent = border_indent
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._moused_over = False
        self._held = False
        self._colors = {c_opt: list(COLOR_OPTIONS[c_opt]) for c_opt in COLOR_OPTIONS}
        super().__init__(bbox, label, callback, visible, spacing_px)
        logging.info("Initialized Button '%s' at position: %s" % (label, bbox))

    def _get_button_text_color(self):
        """
        Return the color of the text.
        """
        if self._held:
            return self._colors['held']
        elif self._moused_over:
            return self._colors['mouseover']
        else:
            return self._colors['idle']

    def move(self, bbox):
        """
        Move the button to the new bbox.
        """
        self._bbox = bbox
        self._calc_dims()

    def _calc_dims(self):
        """
        Determine the text size and position.
        """
        self._text_bbox = {'x': [self._bbox['x'][0] + self._spacing_px, self._bbox['x'][1] - self._spacing_px],
                           'y': [self._bbox['y'][0] + self._spacing_px, self._bbox['y'][1] - self._spacing_px]}
        self._border_indent = self._border_indent
        self._font_size, _ = calc_font_size([self._text], self._text_bbox, self._font, self._border_indent)

        text_dims = cv2.getTextSize(self._text, self._font, self._font_size, 1)[0]
        x0, x1 = self._text_bbox['x']
        y0, y1 = self._text_bbox['y']
        self._text_pos = (x0 + (x1 - x0 - text_dims[0]) // 2, y0 + (y1 - y0 + text_dims[1]) // 2)

    def _render(self, img):
        """
        Render the button.
        """
        p0 = (self._bbox['x'][0], self._bbox['y'][0])
        p1 = (self._bbox['x'][1], self._bbox['y'][1])
        color = self._get_button_text_color()
        # bbox
        #  cv2.rectangle(img, p0, p1, COLORS['orange'].tolist(), 1)

        # text box
        cv2.rectangle(img, (self._text_bbox['x'][0], self._text_bbox['y'][0]),
                      (self._text_bbox['x'][1], self._text_bbox['y'][1]), self._colors['idle'], 1)
        cv2.putText(img, self._text, self._text_pos, self._font, self._font_size, color, 1, cv2.LINE_AA)

    def _mouse_click(self, x, y):
        """
        Check if the click is within the button.
        """
        if bbox_contains(self._bbox, x, y):
            self._held = True
            return True

    def _mouse_move(self, x, y):
        """
        Check if the mouse is over the button.
        """
        self._moused_over = False
        if bbox_contains(self._bbox, x, y):
            self._moused_over = True
            return self._held
        else:
            if self._held:
                self._held = False
        return False

    def _mouse_unclick(self, x, y):
        if self._held and bbox_contains(self._bbox, x, y):
            self._callback()
        self._held = False

    def get_value(self):
        return self._text


class ToggleButton(Button):
    """
    Like a button, but has state (on/off), and renders different color to indicate state, and triggers on unclick.
    """

    def __init__(self, bbox, label, callback, visible=True, border_indent=2, spacing_px=4, default=False):
        self._state = default
        super().__init__(bbox, label, callback, visible, border_indent, spacing_px)

    def _mouse_unclick(self, x, y):
        """
        Check if the click is within the button.
        """
        if bbox_contains(self._bbox, x, y):
            self._held = False
            self._state = not self._state
            self._callback(self._state)
            return True
        return False

    def _get_button_text_color(self):
        if self._held:
            return self._colors['held']
        elif self._moused_over:
            return self._colors['mouseover']
        elif self._state:
            return self._colors['active_toggle']
        else:
            return self._colors['inactive_toggle']

    def render(self, img):
        if not self._state and self._visible:
            # draw an X through the box
            p0 = (self._text_bbox['x'][0], self._text_bbox['y'][0])
            p1 = (self._text_bbox['x'][1], self._text_bbox['y'][1])
            p2 = (self._text_bbox['x'][0], self._text_bbox['y'][1])
            p3 = (self._text_bbox['x'][1], self._text_bbox['y'][0])
            color = self._colors['inactive_toggle']
            cv2.line(img, p0, p1, color, 1)
            cv2.line(img, p2, p3, color, 1)
        super().render(img)

    def get_value(self):
        return self._state


class RadioButtons(Tool):
    """
    Looks like this:

        Title
        -------
         * option 1
           option 2
           option 3

    Color indicates slection/mouseover (no checkbox, etc.)

    """

    def __init__(self, bbox, title, callback, visible=True, options=('1', '2', '3'), texts=None, default_selection=None, spacing_px=6):
        """
        Create list of mutually exclusive items to select.
        """
        self._font = cv2.FONT_HERSHEY_DUPLEX
        self._texts = texts if texts is not None else options
        self._options = options
        logging.info("Initialized RadioButtons '%s' at position: %s" % (title, bbox))
        logging.info("\toptions: %s" % self._options)
        logging.info("\ttexts: %s" % self._texts)

        self._title = title
        self._bbox = bbox
        self._selected_ind = default_selection if default_selection is not None else 0  # currently selected index
        self._mouseover_ind = None  # index of item the mouse is over
        self._colors = {c_opt: list(COLOR_OPTIONS[c_opt]) for c_opt in COLOR_OPTIONS}
        super().__init__(bbox, title, callback, visible, spacing_px=spacing_px)

    def move(self, bbox):
        """
        Move the button to the new bbox.
        """
        self._bbox = bbox
        self._calc_dims()

    def _calc_dims(self):
        """
        Determine the text size, the X,Y positions of each option.
        """

        n_text_lines = (1 if self._title is not None else 0) + len(self._texts)
        height, width = self._bbox['y'][1] - self._bbox['y'][0], self._bbox['x'][1] - self._bbox['x'][0]
        y_intervals = fit_spaced_intervals(self._bbox['y'], n_text_lines, 0.0, fill_extent=True)
        line_h = y_intervals[0][1] - y_intervals[0][0]
        title_h = int(line_h * 1.5)
        self._title_font_size, pos_rel, self._title_thickness = get_font_size(
            self._title, (width-self._spacing_px*2, title_h), incl_baseline=False, font=self._font)
        self._title_pos = (self._spacing_px + self._bbox['x'][0], pos_rel[1] + self._bbox['y'][0])
        body_h = height - title_h
        top, bottom = self._bbox['y']
        left, right = self._bbox['x']
        y_intervals = fit_spaced_intervals((top+title_h, bottom), len(self._texts), 0.0, fill_extent=True)
        self._div_lines = [y_int[0] for y_int in y_intervals] + [y_intervals[-1][1]]
        line_h = y_intervals[0][1] - y_intervals[0][0]
        longest_line = np.argmax([len(t) for t in self._texts])
        self._font_size, pos_rel, self._thickness = get_font_size(
            self._texts[longest_line], (width-self._spacing_px*3, line_h), incl_baseline=False, font=self._font)
        self._font_size = min(self._font_size, self._title_font_size)
        self._line_coords = [(left, top),
                             (left+width, top)]
        self._text_pos = []
        # indent for indicator-dot
        self._dot_size = 2
        left += self._dot_size + self._spacing_px * 2

        for i, text in enumerate(self._texts):
            text_x = self._bbox['x'][0] + self._spacing_px*2
            text_y = self._div_lines[i] + pos_rel[1]
            self._text_pos.append((text_x, text_y))

    def _render(self, img):
        """
        Render the radio buttons.
        """
        p0 = (self._bbox['x'][0], self._bbox['y'][0])
        p1 = (self._bbox['x'][1], self._bbox['y'][1])
        # cv2.rectangle(img, p0, p1, self._colors['unselected'], 1)  # draw bbox
        cv2.putText(img, self._txt_name, self._title_pos, self._font,
                    self._title_font_size, self._colors['unselected'], 1, cv2.LINE_AA)
        #cv2.line(img, self._line_coords[0], self._line_coords[1], self._colors['unselected'])

        for i, text_pos in enumerate(self._text_pos):
            if i == self._selected_ind:
                color = self._colors['selected']
                # indicator dot

                dot_y = (text_pos[1]-self._dot_size-self._spacing_px)
                cv2.circle(img, (self._bbox['x'][0] + self._spacing_px//2, dot_y), self._dot_size, color, -1)
            elif i == self._mouseover_ind:
                color = self._colors['mouseover']
            else:
                color = self._colors['unselected']
            cv2.putText(img, self._texts[i], text_pos, self._font, self._font_size, color, 1, cv2.LINE_AA)

    def _get_item_at(self, y):
        """
        Return the index of the item at the given y-coordinate.
        """
        for i in range(len(self._div_lines)-1):
            if self._div_lines[i] < y < self._div_lines[i+1]:
                return i
        return None

    def _mouse_click(self, x, y):
        """
        Check if the click is within the radio buttons.
        :
        """
        if bbox_contains(self._bbox, x, y):
            ind = self._get_item_at(y)
            if ind is not None:
                self._selected_ind = ind
                if self._callback is not None:
                    self._callback(self.get_value())
        return False

    def _mouse_move(self, x, y):
        """
        Check if the mouse is over the radio buttons.

        """
        if not bbox_contains(self._bbox, x, y):

            self._mouseover_ind = None
        else:
            for i, text_pos in enumerate(self._text_pos):
                self._mouseover_ind = None
                item = self._get_item_at(y)
                if item is not None:
                    self._mouseover_ind = item
                    break
        return False

    def get_value(self):
        """
        Return the selected index.
        """
        return self._options[self._selected_ind]

    def _mouse_unclick(self, x, y):
        pass

    def get_selection(self, name=False):
        """
        Return the selected index.
        """
        return self._selected_ind if not name else self._texts[self._selected_ind]
