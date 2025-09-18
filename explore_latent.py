"""
main_app:


+-------------------------------+----------+
| Embedding pan/zoom            | results  |
|                               | window   |
|                               | [exp 1]  |  <- change experiments button
|                               |          |
|                               |          | <- experiment buttons
|                               |          | 
+-------------------------------+----------+

    For panning/zooming around in embedded latent space.
    For selecting samples for experiments.


"""

import cv2
import numpy as np
import logging
import sys

from mnist import MNISTData
import time
from enum import IntEnum
from abc import ABC, abstractmethod
from embedding_drawing import EmbeddingPanZoom
from embed import LatentRepEmbedder
from util import fit_spaced_intervals, draw_bbox
from colors import COLORS, MPL_CYCLE_COLORS, COLOR_SCHEME


from abc import ABC, abstractmethod


class ResultPanel(ABC):
    def __init__(self, bbox, app):

        y_div = app.LAYOUT['dims']['header_y_px'] + bbox['y'][0]
        self._title_bbox = {'x': bbox['x'],
                            'y': (bbox['y'][0], y_div)}
        self.bbox = {'x': bbox['x'],
                     'y': (y_div, bbox['y'][1])}

        self.app = app
        self._info = self.app.EXPERIMENTS['interp']
        self._fg = np.array(COLOR_SCHEME['fg'], dtype=np.uint8).reshape(1, 1, 3)
        self._bkg = np.array(COLOR_SCHEME['bkg'], dtype=np.uint8).reshape(1, 1, 3)
        self._t0 = time.perf_counter()
        self._startup()

    def _startup(self):
        self._src_boxes, self._tgt_boxes = self._calc_dims()
        self._n_sources, self._n_targets = self._get_slot_counts()
        self.reset_ui_state()
        self._title = self._TITLE

        self._cur_box_ind = 0  # set by user keypress up/down to change individual interpolation sources

    @abstractmethod
    def _get_slot_counts(self):
        pass

    @abstractmethod
    def _render(self, frame):
        pass

    def render(self, frame):

        # Write the title
        self._write_centered_caption(frame, self._title_bbox, self._title, text_indent=0, font_scale=.75)
        # Draw box around the current box
        if self._cur_box_ind < len(self._src_boxes):
            box = self._src_boxes[self._cur_box_ind]
        else:
            box = self._tgt_boxes[self._cur_box_ind - self._n_sources]
        draw_bbox(frame, box, thickness=2, color=COLOR_SCHEME['mouseover'], inside=False)
        return self._render(frame)

    @abstractmethod
    def _reset_ui_state(self):
        # Custom reset for each experiment
        pass

    @abstractmethod
    def _calc_dims(self):
        """
        Compute how many tiles we can fit across, vertical/horizontal spacing of each tile.
        :RETURNS:
            number of user-selectable source boxes,
            number of user-selectable target boxes
        """
        pass

    @abstractmethod
    def _update_results(self, src_ind=None, tgt_ind=None):
        """
        Compute results for source index src_ind --> target_ind
        :param src_ind: index of source box that was updated, or None if target box was updated
        :param tgt_ind: index of target box that was updated, or None if source box was updated
        """
        pass

    def move_cur_box(self, direction):
        if direction not in (-1, 1):
            raise ValueError("Direction must be -1 or +1")
        self._cur_box_ind += direction
        if self._cur_box_ind < 0:
            self._cur_box_ind = 0
        elif self._cur_box_ind >= self._n_sources + self._n_targets:
            self._cur_box_ind = self._n_sources + self._n_targets - 1
        print("Current box index now ", self._cur_box_ind)

    def reset_ui_state(self):
        # common reset for all experiments
        self._cur_box_ind = 0
        self._source_indices = [None] * self._n_sources  # indices into embedding for each interpolation source image
        self._source_images = {}  # Key is index [0 - n_sources-1], values is image to display or None to show bbox
        self._target_images = {}
        self._target_indices = [None] * self._n_targets  # index into embedding for target image

        self._n_src_updates = 0
        self._n_tgt_updates = 0

        self._reset_ui_state()

    def set_box_sample(self, sample_ind):
        """
        User clicked a sample image, set the current box to that sample.
        :param sample_ind: index into embedding.autoencoder.[training data]
        """

        def _set_target(target_index):
            if sample_ind == self._target_indices[target_index]:
                return False
            self._target_indices[target_index] = sample_ind
            self._target_images[target_index] = self._make_train_image(sample_ind)
            self._n_tgt_updates += 1
            return True

        def _set_source(source_index):
            if self._source_indices[source_index] == sample_ind:
                return None
            self._source_indices[source_index] = sample_ind
            self._source_images[source_index] = self._make_train_image(sample_ind)
            self._n_src_updates += 1
            return source_index

        if self._cur_box_ind >= self._n_sources:
            box_ind = self._cur_box_ind - self._n_sources
            update = _set_target(box_ind)

            # Cycle through all target boxes:
            self._cur_box_ind += 1
            if self._cur_box_ind >= self._n_sources + self._n_targets:
                self._cur_box_ind = self._n_sources

            src_update_ind = None
            tgt_update_ind = box_ind
        else:
            update_ind = _set_source(self._cur_box_ind)

            # move to next box, or target if at end of sources:
            self._cur_box_ind += 1

            update = update_ind is not None
            src_update_ind = update_ind
            tgt_update_ind = None

        if update:
            self._update_results(src_ind=src_update_ind, tgt_ind=tgt_update_ind)

    def _render_box(self, frame, bbox, tile_img, box_color):
        if tile_img is None:
            draw_bbox(frame, bbox, thickness=1, color=box_color, inside=True)
        else:
            ih, iw = tile_img.shape[0:2]
            box_w, box_h = bbox['x'][1] - bbox['x'][0], bbox['y'][1] - bbox['y'][0]
            if iw != box_w or ih != box_h:
                raise ValueError(f"Image size {iw}x{ih} does not match box size {box_w}x{box_h}")
            frame[bbox['y'][0]:bbox['y'][1], bbox['x'][0]:bbox['x'][1], :] = tile_img

    def _make_train_image(self, src_ind):
        if src_ind is None:
            return None

        img = self.app.embedding.images_in[src_ind].reshape((28, 28, 1))
        color_img = self._fg * img + self._bkg * (1.0 - img)
        img = (color_img).astype(np.uint8)
        img = cv2.resize(img, (self._info['tile_size'], self._info['tile_size']), interpolation=cv2.INTER_NEAREST)
        return img


    def _write_centered_caption(self, frame, box, text, text_indent=5, font_scale = 0.4):
        font = self.app.LAYOUT['font']
        thickness = 1
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_w, text_h = text_size
        box_w, box_h = box['x'][1] - box['x'][0], box['y'][1] - box['y'][0]
        text_x = box['x'][0] + (box_w - text_w) // 2 - text_indent
        text_y = box['y'][0] + (box_h + text_h) // 2
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, COLOR_SCHEME['text'], thickness, cv2.LINE_AA)



class InterpExtrapResultPanel(ResultPanel):
    """
    Result window, experiment 1:  Interpolations and Extrapolations

        +------------------------------------------+
        |                                          |
        |   [src 1]   [src 2]   [src 3]   [src 4]  |  <- source images (t = 0.0)
        |                                          |
        |    [i11]     [i12]     [i13]     [i14]   |  t = 0.1
        |                                          |
        |    [i21]     [i22]     [i23]     [i24]   |  t = 0.2
        |                                          |
        |                   ...                    |  t = 0.3 - 0.8
        |                                          |
        |    [i91]     [i92]     [i93]     [i94]   |  t = 0.9
        |                                          |
        |    [tgt]     [tgt]     [tgt]     [tgt]   |  <- target image, all same (t = 1.0)
        |                                          |
        |    [e11]     [e12]     [e13]     [e14]   |  t = 1.1
        |                                          |
        |    [e21]     [e22]     [e23]     [e24]   |  t = 1.5
        |                                          |
        |    [e31]     [e32]     [e33]     [e34]   |  t = 2.0
        |                                          |
        |            [clear sources]               |
        |                                          |
        +------------------------------------------+

        User selects N sources (N is shown as 4 here), and a single target.

        For each source, we show interpolations between source and target 
        (at intervals of 0.1) and extrapolations beyond the target and in 
        the same direction (at 1.1, 1.5, 2.0 times the distance).
    """

    _TITLE = "Interpolation / Extrapolation"

    def _reset_ui_state(self):

        self._interp_images = {}  # key is col, value is {row: image}
        self._extrap_images = {}  # key is col, value is {row: image}

    def _calc_dims(self):
        """
        Compute how many tiles we can fit across, vertical/horizontal spacing of each tile.
        """
        tile_s = self._info['tile_size']
        pad = self.app.LAYOUT['small_pad']
        indent = self.app.LAYOUT['outer_pad']
        self._interp_vals = self._info['interp_factors']
        self._extrap_vals = self._info['extrap_factors']

        x_left_margin = self._info['left_margin_px']

        top, bottom = self.bbox['y'][0] + indent, self.bbox['y'][1] - indent
        left, right = self.bbox['x'][0] + indent + x_left_margin, self.bbox['x'][1] - indent
        print("Result panel bbox", left, top, right, bottom)
        h, w = bottom - top, right - left

        n_across = (w + pad) // (tile_s + pad)
        print("N across", n_across)

        self.n_sources = n_across

        across_x_intervals = fit_spaced_intervals((left, right), n_across, 0.0, min_spacing=pad, fill_extent=False)
        assym_right = right - across_x_intervals[-1][-1]
        assym_left = across_x_intervals[0][0] - left
        diff = (assym_right - assym_left) // 2
        across_x_intervals = [(axi[0] + diff, axi[1] + diff) for axi in across_x_intervals]

        x_tile_locs = [(axi[0], axi[0] + tile_s) for axi in across_x_intervals]

        # the linearly interpolated part is equally spaced
        height_for_interp = int((bottom-top) * (len(self._interp_vals) /
                                (len(self._interp_vals) + len(self._extrap_vals) + 1)))

        interp_y_intervals = fit_spaced_intervals((top, top + height_for_interp),
                                                  len(self._interp_vals) + 2,
                                                  tile_s,
                                                  min_spacing=pad,
                                                  fill_extent=False)

        def _make_box_row(y_interval):
            return [{'x': x_span, 'y': y_interval} for x_span in x_tile_locs]

        y_interp_locs = [(iyi[0], iyi[0] + tile_s) for iyi in interp_y_intervals]
        src_boxes = _make_box_row(y_interp_locs[0])
        interp_box_rows = [_make_box_row(y_interval) for y_interval in y_interp_locs[1:-1]]
        self._interp_boxes = list(zip(*interp_box_rows))

        interp_box_y_spacing = self._interp_boxes[0][1]['y'][0] - self._interp_boxes[0][0]['y'][1]
        print(interp_box_y_spacing)

        # the extrapolated part is more spaced out
        y_top = y_interp_locs[-1][1] + interp_box_y_spacing
        y_extrap_intervals = []

        y_spacing = int((bottom - y_top - tile_s * len(self._extrap_vals)) / (len(self._extrap_vals) + .5))
        for ev in self._extrap_vals:
            y_extrap_intervals.append((y_top, y_top + tile_s))
            y_top += tile_s + y_spacing
        y_extrap_bbox = [(eyi[0], eyi[0] + tile_s) for eyi in y_extrap_intervals]

        # rotate so first index is source 1, ...

        tgt_boxes = _make_box_row(y_interp_locs[-1])
        extrap_box_rows = [_make_box_row(y_interval) for y_interval in y_extrap_bbox]
        self._extrap_boxes = list(zip(*extrap_box_rows))

        box_left = self.bbox['x'][0] + indent
        box_right = x_tile_locs[0][0] - pad
        top = y_interp_locs[0][0]
        bottom = y_extrap_bbox[-1][1]

        self._caption_bbox = {'x': (box_left, box_right),
                              'y': (top, bottom)}

        def _make_cap_box(y_interp_loc):
            return {'x': (box_left, box_right),
                    'y': y_interp_loc}

        self._caption_boxes = {'source': _make_cap_box(y_interp_locs[0]),
                               'target': _make_cap_box(y_interp_locs[-1]),
                               'interp': [_make_cap_box(y_interval) for y_interval in y_interp_locs[1:-1]],
                               'extrap': [_make_cap_box(y_interval) for y_interval in y_extrap_bbox]}
        return src_boxes, tgt_boxes

    def _update_results(self, src_ind=None, tgt_ind=None):
        # Compute interpolations and extrapolations for source index src_ind --> target_ind
        print("Recomputing interp/extrap for source ", src_ind, " target ", tgt_ind)
        pass

    def _render(self, frame):
        """
        Render the interpolation/extrapolation results.
        """

        # Captions in the right column
        self._write_centered_caption(frame, self._caption_boxes['source'], self._info['captions']['source'])
        self._write_centered_caption(frame, self._caption_boxes['target'], self._info['captions']['target'])
        for i, box in enumerate(self._caption_boxes['interp']):
            caption = self._info['captions']['interp'] % (self._interp_vals[i],)
            self._write_centered_caption(frame, box, caption)
        for i, box in enumerate(self._caption_boxes['extrap']):
            caption = self._info['captions']['extrap'] % (self._extrap_vals[i],)
            self._write_centered_caption(frame, box, caption)

        for i in range(self.n_sources):
            target_img = self._target_images.get(0, None) if 0 in self._target_images else None
            self._render_box(frame, self._tgt_boxes[i], target_img,  COLOR_SCHEME['a_output'])
            self._render_box(frame, self._src_boxes[i], self._source_images.get(i, None), COLOR_SCHEME['a_source'])
            for j in range(len(self._interp_vals)):
                interp_img = self._interp_imgs[i].get(j, None) if i in self._interp_images else None
                self._render_box(frame, self._interp_boxes[i][j], interp_img, COLOR_SCHEME['a_dest'])
            for j in range(len(self._extrap_vals)):
                extrap_img = self._extrap_images[i].get(j, None) if i in self._extrap_images else None
                self._render_box(frame, self._extrap_boxes[i][j], extrap_img, COLOR_SCHEME['a_input'])

    def _get_slot_counts(self):
        return len(self._src_boxes), 1


class AnalogyResultPanel(ResultPanel):
    """
    Experiment 2:  Analogies

        +------------------------------------------------------+
        |                                                      |
        |                [Image A] : [Image B]                 |
        |                                                      |
        |  [Image C1] : [Image D1]    [Image C5] : [Image D5]  |
        |                                                      |
        |  [Image C2] : [Image D2]    [Image C6] : [Image D6]  |
        |                                                      |
        |  [Image C3] : [Image D3]    [Image C7] : [Image D7]  |
        |                                                      |
        |  [Image C4] : [Image D4]    [Image C8] : [Image D8]  |
        |                                                      |
        +------------------------------------------------------+

        User select 8 "C" images for analogy completion, then selects
        the "A" and "B" target images to define the analogy/transformation.

        The completed analogies are shown in the "D" images.
    """
    _TITLE = "Analogy Completion"

    def _get_slot_counts(self):
        return len(self._src_boxes), 2

    def _calc_dims(self):
        # How many columns can we fit given the padding and the size of 2 tiles for each analogy?
        self._info = self.app.EXPERIMENTS['analogy']
        tile_s = self._info['tile_size']
        a_space = self._info['a_space']
        col_space = self._info['col_space']
        row_space = self._info['row_space']
        indent = self.app.LAYOUT['outer_pad']

        top, bottom = self.bbox['y'][0] + indent, self.bbox['y'][1] - indent
        left, right = self.bbox['x'][0] + indent, self.bbox['x'][1] - indent

        width, height = right - left, bottom - top

        analogy_width = tile_s * 2 + a_space
        n_cols = (width + col_space) // (analogy_width + col_space)
        n_cols = max(1, n_cols)
        n_rows = (height + row_space) // (tile_s + row_space)
        n_rows = max(2, n_rows)

        center_x = (left + right) // 2

        target_x_interval = (center_x - analogy_width // 2, center_x - analogy_width // 2 + tile_s)  # at the top

        across_x_intervals = fit_spaced_intervals(
            (left, right), n_cols, 0.0, min_spacing=col_space, fill_extent=False)
        across_y_intervals = fit_spaced_intervals(
            (top, bottom), n_rows, 0.0, min_spacing=row_space, fill_extent=False)
        if len(across_x_intervals) == 1:
            # exactly under the target pair
            across_x_intervals = [target_x_interval]
            diff = 0
        else:
            # center it horizontally
            assym_right = right - across_x_intervals[-1][-1]
            assym_left = across_x_intervals[0][0] - left
            diff = (assym_right - assym_left) // 2
            print("Left extra: ", assym_left, "Right extra: ", assym_right, "Diff:", diff)

        def _make_bbox_pair(x0, y0):
            return [{'x': (x0, x0 + tile_s), 'y': (y0, y0 + tile_s)},
                    {'x': (x0 + tile_s + a_space, x0 + 2*tile_s + a_space), 'y': (y0, y0 + tile_s)}]

        target_bboxes = _make_bbox_pair(target_x_interval[0], top)
        self._analogy_bboxes = []  # both source and response boxes
        self._response_bboxes = []
        source_bboxes = []

        for yi in across_y_intervals[1:]:
            for xi in across_x_intervals:
                box_pair = _make_bbox_pair(xi[0] + diff, yi[0])
                self._analogy_bboxes.append(box_pair)
                source_bboxes.append(box_pair[0])
                self._response_bboxes.append(box_pair[1])
        self.n_targets = 2
        self.n_analogies = len(self._analogy_bboxes)
        print(f"Can fit {n_cols} columns and {n_rows} rows, total {self.n_analogies} analogies")
        return source_bboxes, target_bboxes

    def _do_analogy(self):
        a_source_code = self.app.embedder.latent_codes[self.samples[0]]
        a_dest_code = self.app.embedder.latent_codes[self.samples[1]]
        a_input_code = self.app.embedder.latent_codes[self.samples[2]]
        a_output_code = a_input_code + (a_dest_code - a_source_code)
        # Perform analogy operation here
        a_source_img = self.app.embedder.images_in[self.samples[0]]
        a_dest_img = self.app.embedder.images_in[self.samples[1]]
        a_input_img = self.app.embedder.images_in[self.samples[2]]
        a_output_img = self.app.embedder.autoencoder.decode_samples(a_output_code.reshape(1, -1))

    def _render(self, frame):

        for a_ind, a_box_pair in enumerate(self._analogy_bboxes):
            box_a, box_b = a_box_pair
            self._render_box(frame, box_a, self._source_images.get(a_ind, None), COLOR_SCHEME['a_input'])
            self._render_box(frame, box_b, self._analogy_images.get(a_ind, None), COLOR_SCHEME['a_output'])

        self._render_box(frame, self._tgt_boxes[0], self._target_images.get(0, None), COLOR_SCHEME['a_source'])
        self._render_box(frame, self._tgt_boxes[1], self._target_images.get(1, None), COLOR_SCHEME['a_dest'])

    def _update_results(self, src_ind=None, tgt_ind=None):
        print("Recomputing analogy results for source ", src_ind, " target ", tgt_ind)
        # Compute analogy results for all analogy source indices --> target_indices
        pass

    def _reset_ui_state(self):
        self._analogy_images = {}  # key is analogy index, value is image to display or None to show bbox


class ExploreDims:

    LAYOUT = {'dims': {'x_div_rel': 0.7, 'header_y_px': 70},   # division between embedding and results
              'small_pad': 6,     # between tiles in analogy display
              'outer_pad': 20,

              'font': cv2.FONT_HERSHEY_SIMPLEX, }

    EXPERIMENTS = {'interp': {'name': "Interpolation / Extrapolation",
                              'tile_size': 28*3,  # in results display
                              'interp_factors': np.linspace(0.0, 1.0, 6 + 1).tolist()[1:-1],  # exclude 0.0 and 1.0
                              'extrap_factors': [1.1, 1.5, 2.0],
                              'left_margin_px': 50,   # print interpolation parameter (t=.3) in this column
                              'captions': {'source': 'Source',
                                           'target': 'Target',
                                           'interp': 't=%.2f',
                                           'extrap': 't=%.2f'},
                              'n_targets': 1, },  # morph target

                   'analogy': {'name': "Analogy Completion",
                               'tile_size': 28*4,
                               'a_space': 4,  # space between analogy pairs
                               'col_space': 10,  # space between analogy columns
                               'row_space': 10,
                               'n_targets': 2, }  # analogy A and B vector endpoints
                   }
    # embedder = LatentRepEmbedder.from_filename(sys.argv[1])  # DEBUG, use for test_result_panel


def test_result_panel(cls):
    win_size = [(776, 950)]
    x_div = int(.7 * win_size[0][0])
    results_bbox = {'x': (x_div, win_size[0][0]), 'y': (0, win_size[0][1])}
    panel = [cls(results_bbox, ExploreDims())]

    img = [np.zeros((win_size[0][1], win_size[0][0], 3), dtype=np.uint8)]
    img[0][:] = COLORS['OFF_WHITE_RGB']
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("test", *win_size[0])
    n_frames = 0

    def _check_win_size():
        width, height = cv2.getWindowImageRect("test")[2:4]

        if (width, height) != win_size[0]:
            win_size[0] = (width, height)
            img[0] = np.zeros((win_size[0][1], win_size[0][0], 3), dtype=np.uint8)
            img[0][:] = COLORS['OFF_WHITE_RGB']
            x_div = int(.7 * win_size[0][0])
            bbox = {'x': (x_div, width), 'y': (0, height)}
            print("Window resized to %i x %i, new bbox %s" % (width, height, str(bbox)))
            return bbox

        return None

    while True:
        frame = img[0].copy()
        panel[0].render(frame)
        bbox = _check_win_size()
        if bbox is not None:
            panel[0] = cls(bbox, ExploreDims())

        cv2.imshow("test", frame[:, :, ::-1])
        key = cv2.waitKey(10)
        if key == 27 or key == ord('q'):  # ESC key
            break
        # Up/down to change bbox in UI
        # elif key == ord('u'):  # up
        n_frames += 1

        # if n_frames % 10 == 0:
        #     print(f"Rendered {n_frames} frames")


class Explore(ExploreDims):

    def __init__(self, embedding, win_size=(1900, 950)):
        self.embedding = embedding
        self.win_size = win_size

        self._blank = np.zeros((win_size[1], win_size[0], 3), dtype=np.uint8)
        self._blank[:] = COLORS['OFF_WHITE_RGB']

        x_div = int(self.LAYOUT['dims']['x_div_rel'] * win_size[0])
        self._embed_bbox = {'x': (0, x_div), 'y': (0, win_size[1])}
        self._results_bbox = {'x': (x_div, win_size[0]), 'y': (0, win_size[1])}
        embed_size = (self._embed_bbox['x'][1] - self._embed_bbox['x'][0],
                      self._embed_bbox['y'][1] - self._embed_bbox['y'][0])

        images = self.embedding.images_in.reshape(-1, 28, 28)
        labels = self.embedding.labels_in
        embed_pos_2d = self.embedding.embedded_latent
        self.epz = EmbeddingPanZoom(embed_size, embed_pos_2d, images, labels, MPL_CYCLE_COLORS)

        self._experiments = [InterpExtrapResultPanel(bbox=self._results_bbox, app=self),
                             AnalogyResultPanel(bbox=self._results_bbox, app=self)]
        self._cur_experiment = 0

        self.offset = np.zeros(2)
        self._win_name = "Embedding Viewer"

        self.timing_info = {'fps': 0,
                            'n_frames': 0,
                            't_start': time.perf_counter(),
                            't_update_interval_sec': 2.0}

        # self.mouse_info = {'clicked_pos': None,
        #                    'button_held': None  # "left" or 'right'
        #                    }
        cv2.namedWindow(self._win_name, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(self._win_name, self.size[0], self.size[1])
        cv2.setMouseCallback(self._win_name, self._mouse_callback)

    def _get_bbox(self):
        center = np.array([0.5, 0.5]) + self.offset
        box_wh = np.array([0.5, 0.5]) / self.scale
        return {'x': (center[0] - box_wh[0], center[0] + box_wh[0]),
                'y': (center[1] - box_wh[1], center[1] + box_wh[1])}
    '''
    def _check_window_size(self):
        w, h= cv2.getWindowImageRect(self._win_name)[2:4]
        if (w, h) != self.win_size:
            self.win_size = (w, h)
            self._blank = np.zeros((h, w, 3), dtype=np.uint8)
            self._blank[:] = COLORS['OFF_WHITE_RGB']
            x_div = int(self.LAYOUT['dims']['x_div_rel'] * w)
            self._embed_bbox = {'x': (0, x_div), 'y': (0, h)}
            self._results_bbox = {'x': (x_div, w), 'y': (0, h)}
            embed_size = (self._embed_bbox['x'][1] - self._embed_bbox['x'][0],
                                       self._embed_bbox['y'][1] - self._embed_bbox['y'][0])
            self._render.set_size(embed_size) ???
    '''

    def run(self):

        # cv2.namedWindow(self._win_name, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(self._win_name, *self.win_size)
        # cv2.setMouseCallback(self._win_name, self.mouse_callback)

        self._click_px = None
        self._pan_offset = None
        self._moused_over = None  # index into points

        while True:

            # self._check_window_size()
            img = self._blank.copy()
            self._experiments[self._cur_experiment].render(img)
            embed_img = self.epz.get_frame(self._pan_offset, moused_over=self._moused_over)
            img[self._embed_bbox['y'][0]:self._embed_bbox['y'][0]+embed_img.shape[0],
                self._embed_bbox['x'][0]:self._embed_bbox['x'][0]+embed_img.shape[1], :] = embed_img

            cv2.imshow(self._win_name, img[:, :, ::-1])
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC key
                break

            elif key & 0xFF == ord('['):  # reset view
                self._experiments[self._cur_experiment].move_cur_box(-1)
            elif key & 0xFF == ord(']'):  # reset view
                self._experiments[self._cur_experiment].move_cur_box(1)

            elif key & 0xFF == ord(','):
                self.epz.shrink(-1)
            elif key & 0xFF == ord('.'):
                self.epz.shrink(1)

            elif key & 0xFF == ord(';'):
                self.epz.sample(-1)
            elif key & 0xFF == ord('\''):
                self.epz.sample(1)

            elif key == ord('e'):  # next experiment
                self._cur_experiment = (self._cur_experiment + 1) % len(self._experiments)
                self._experiments[self._cur_experiment].reset_ui_state()
                logging.info(f"Experiment changed to {self._experiments[self._cur_experiment].__class__.__name__}")
            # up/down arrows to change current box in results panel
            elif key != 255:
                logging.info(f"Key {key} pressed, no action assigned")

            # Update timing information
            self.timing_info['n_frames'] += 1
            now = time.perf_counter()
            if now - self.timing_info['t_start'] > self.timing_info['t_update_interval_sec']:
                self.timing_info['fps'] = self.timing_info['n_frames'] / (now - self.timing_info['t_start'])
                logging.info(f"Current FPS: {self.timing_info['fps']:.2f}")
                self.timing_info['t_start'] = now
                self.timing_info['n_frames'] = 0

        cv2.destroyAllWindows()

    def _set_sample(self, sample_ind):
        self._experiments[self._cur_experiment].set_box_sample(sample_ind)

    def _mouse_callback(self, event, x, y, flags, param):
        pos_px = np.array((x, y))
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click_px = pos_px
            if self._moused_over is not None:
                self._set_sample(self._moused_over)

        elif event == cv2.EVENT_LBUTTONUP:
            self._click_px = None
            self.epz.pan_by(self._pan_offset)
            self._pan_offset = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._click_px is not None:
                self._pan_offset = pos_px - self._click_px
                print("Panning by ", self._pan_offset)
            else:
                self._moused_over = self.epz.get_moused_over(pos_px)

                print("Mouse moved, now over: ", self._moused_over)

        elif event == cv2.EVENT_MOUSEWHEEL:
            direction = int(np.sign(flags))
            self.epz.zoom_at(direction, pos_px=pos_px)


def run_app():

    if len(sys.argv) < 2:
        print("Usage: python explore_latent.py <embedding.pkl>")
        sys.exit(1)
    embedder = LatentRepEmbedder.from_filename(sys.argv[1])
    explorer = Explore(embedder)
    explorer.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_app()  # Run the test function to verify the implementation
    # test_result_panel(InterpExterpResultPanel)
    # test_result_panel(AnalogyResultPanel)
