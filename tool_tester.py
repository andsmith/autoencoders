"""
Lightweight interface with tools.
"""

import cv2
import numpy as np
from tools import RadioButtons, Tool, Button, Slider
from colors import COLORS


class ToolTester(object):
    """
    Test a tool.
    """

    def __init__(self, tools, img_size=(640, 480)):
        """
        Create a tool and test it.
        """
        self._img_size = img_size
        self._tools = tools
        self._test()

    def _test(self):
        """
        Test the tool.
        """

        blank = np.zeros((500, 500, 3), dtype=np.uint8) 
        blank[:] = COLORS['OFF_WHITE_RGB']
        # draw_color = COLORS['black'].tolist()
        cv2.namedWindow("Test")
        cv2.setMouseCallback("Test", self._mouse_callback)
        while True:
            img = blank.copy()
            for tool in self._tools:
                tool.render(img)
            cv2.imshow("Test", img[:,:,::-1])
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break

        cv2.destroyAllWindows()

    def _mouse_callback(self, event, x, y, flags, param):
        """
        Handle mouse events.
        """
        for tool in self._tools:
            if event == cv2.EVENT_LBUTTONDOWN:
                tool.mouse_click(x, y)
            elif event == cv2.EVENT_MOUSEMOVE:
                tool.mouse_move(x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                tool.mouse_unclick(x, y)


def test_radio():
    def callback(selection):
        print("Radio button selected:", selection)
    params = {'title': "Cluster Type",
              'callback': callback,
              'options': ['Elliptical', 'Gaussian', 'Annular', 'taco', 'nacho', 'burrito supreme'],
              'default_selection': 1,
              'bbox': {'x': [10, 220], 'y': [10, 220]}}
    tt = ToolTester([RadioButtons(**params)])


def test_button():
    def callback():
        print("Button pressed")
    params = {'label': "Button",
              'callback': callback,
              'bbox': {'x': [50, 125], 'y': [100, 160]}}
    tt = ToolTester([Button(**params)])


def test_slider():
    params = {'label': "Numy Points",
              'range': [0, 100],
              'default': 100,
              'bbox': {'x': [10, 330], 'y': [10, 120]}}
    #import ipdb; ipdb.set_trace()
    tt = ToolTester([Slider(**params)])


if __name__ == "__main__":
    test_radio()
    test_button()
    test_slider()
