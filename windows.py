import numpy as np
import cv2
import time
from abc import ABC, abstractmethod
import logging
from embedding_drawing import EmbeddingPanZoom
from colors import COLOR_SCHEME
from util import draw_bbox

WIN_SIZES = {'embedding': {'size_wh': (1400, 700)},
             'info_control': {'size_wh': (400, 800)},
             'analogy': {'size_wh': (1400, 250)}}


class Window(ABC):
    """
    App subwindow.
    """

    def __init__(self,  title, init_size=(640, 480)):
        self._win_name = title
        self.size = init_size if init_size is None else init_size
        logging.info(f"Initializing window: {self._win_name} with size {self.size}")

    @abstractmethod
    def _get_frame(self):
        """
        Return the current frame to be displayed.
        """
        pass

    @abstractmethod
    def mouse_callback(self, event, x, y, flags, param):
        """
        Handle mouse events.
        """
        pass

    def _init_cv2(self):
        # NOTE:  Only call from main thread
        cv2.namedWindow(self._win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._win_name, *self.size)
        cv2.setMouseCallback(self._win_name, self.mouse_callback)
        logging.info(f"Window display {self._win_name} initialized with size {self.size}")
        return self._win_name

    def refresh(self):
        # NOTE:  Only call from main thread
        self._check_size()
        t0 = time.perf_counter()
        frame = self._get_frame()
        duration = time.perf_counter() - t0
        cv2.imshow(self._win_name, frame)
        return duration

    def _check_size(self):
        cur_size = cv2.getWindowImageRect(self._win_name)
        if cur_size[2:] != self.size:
            self.size = cur_size[2:]
            self._resize(self.size)

    def _resize(self, new_img_size):
        self._img_size = new_img_size
        logging.info(f"Window {self._win_name} resized to {self._img_size}")


class EmbedWindow(Window):
    """"

    main_app:
    +--------------------------------+ 
    | Embedding pan/zoom             |
    |  interact, start no selections |
    |   -mouseover/select samp1,     |
    |   -then samp2 to form vector.  |
    |   -select N analogy targets to |
    |    show analogies (not here)   |  (shown in AnalogyWindow)
    +--------------------------------+
    mouseover:  "floating" analogy target w/live updates as moused over sample changes.
    right-click:  toggle analogy targets
    left-click: toggle vector endpoint, left-double-click to switch direction.

    Communicates w/pan_zoom using sample_ids, select/deselect, etc.

    """

    def __init__(self, pan_zoom):
        super().__init__(title='Embedding space')
        self._pan_zoom = pan_zoom.set_embed_win(self)
        self._embed = pan_zoom.embed
        self._grab_pos_xy = None

        # pan-zoom reads these to draw frames:
        self.mouse_held = None
        self.mouseover = None
        self.a_inputs = []
        self.vec_points = [None, None]

    def _get_frame(self):
        # Get the current frame from the pan-zoom object, send current state to
        return self._pan_zoom.get_frame()

    def mouse_callback(self, event, x, y, flags, param):
        over = self._pan_zoom.get_over(x, y)
        if over is None:
            self.mouseover = None


class InfoWindow(Window):
    pass


class AnalogyWindow(Window):
    pass

