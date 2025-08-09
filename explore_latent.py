import cv2
import numpy as np
import logging
from embedding_render import MNISTEmbedRenderer
from embeddings import PassThroughEmbedding, PCAEmbedding, TSNEEmbedding, UMAPEmbedding
from mnist import MNISTData
import time


class PanZoomEmbedding(object):
    def __init__(self, embedding, win_size):
        self.embedding = embedding
        self.win_size = win_size
        self.offset = np.zeros(2)
        self._win_name = "Embedding Viewer"
        self.scale = 1.0
        self.spread = 1.0
        self.timing_info = {'fps': 0,
                            'n_frames': 0,
                            't_start': time.perf_counter(),
                            't_update_interval_sec': 2.0}

        self.mouse_info = {'clicked_pos': None,
                           'button_held': None  # "left" or 'right'
                           }
        self._blank = np.zeros((win_size[1], win_size[0], 3), dtype=np.uint8)
        self._render = MNISTEmbedRenderer(embedding)

    def _get_bbox(self):
        center = np.array([0.5, 0.5]) + self.offset
        box_wh = np.array([0.5, 0.5]) / self.scale
        return {'x': (center[0] - box_wh[0], center[0] + box_wh[0]),
                'y': (center[1] - box_wh[1], center[1] + box_wh[1])}

    def start(self):

        cv2.namedWindow(self._win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._win_name, *self.win_size)
        cv2.setMouseCallback(self._win_name, self.mouse_callback)

        while True:
            img = self._blank.copy()
            self._render.render_embedding(img, self._get_bbox(), spread=self.spread)
            cv2.imshow(self._win_name, img)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):  # ESC key
                break
            elif key == ord('i'):  # increase spread
                self.spread *= 1.1
                logging.info(f"Spread increased to {self.spread}")
            elif key == ord('d'):  # decrease spread
                self.spread /= 1.1
                logging.info(f"Spread decreased to {self.spread}")

            # Update timing information
            self.timing_info['n_frames'] += 1
            now = time.perf_counter()
            if now - self.timing_info['t_start'] > self.timing_info['t_update_interval_sec']:
                self.timing_info['fps'] = self.timing_info['n_frames'] / (now - self.timing_info['t_start'])
                logging.info(f"Current FPS: {self.timing_info['fps']:.2f}")
                self.timing_info['t_start'] = now
                self.timing_info['n_frames'] = 0

        cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.scale *= 1.1
            else:
                self.scale /= 1.1
            logging.info(f"Scale changed to {self.scale}")
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_info['clicked_pos'] = (x, y)
            self.mouse_info['button_held'] = 'left'
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.mouse_info['clicked_pos'] = (x, y)
            self.mouse_info['button_held'] = 'right'
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mouse_info['button_held'] == 'left':
                dx = (x - self.mouse_info['clicked_pos'][0]) / self.win_size[0]
                dy = (y - self.mouse_info['clicked_pos'][1]) / self.win_size[1]
                self.offset -= np.array([dx, dy]) / self.scale
                self.mouse_info['clicked_pos'] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            self.mouse_info['button_held'] = None
            self.mouse_info['clicked_pos'] = None



def test_get_bbox():
    pze = PanZoomEmbedding(None, win_size=(800, 600))
    import matplotlib.pyplot as plt

    bbox_0 = pze._get_bbox()
    pze.scale = 2.0
    bbox_1 = pze._get_bbox()
    pze.offset = (0.25, 0.25)
    bbox_2 = pze._get_bbox()

    def plot_box(ax, bbox, color, label):
        x_range = bbox['x']
        y_range = bbox['y']
        box_coords = np.array([[x_range[0], y_range[0]],
                               [x_range[1], y_range[0]],
                               [x_range[1], y_range[1]],
                               [x_range[0], y_range[1]],
                               [x_range[0], y_range[0]]])
        ax.plot(box_coords[:, 0], box_coords[:, 1], color=color, label=label)

    fig, ax = plt.subplots()
    plot_box(ax, bbox_0, color='r', label='original(unit)')
    plot_box(ax, bbox_1, color='r', label='smaller')
    plot_box(ax, bbox_2, color='g', label='moved and smaller')
    plt.legend()
    plt.ylim(-1, 2)
    plt.xlim(-1, 2)
    plt.show()


def pan_zoom(n_sample=3000):
    data = MNISTData()
    sample_inds = np.random.choice(len(data.x_train), n_sample, replace=False)
    x_train = data.x_train[sample_inds].reshape(n_sample, -1)
    y_train = data.y_train[sample_inds]
    labels = np.argmax(y_train, axis=1)
    x_latent = x_train
    embed = PCAEmbedding(x_latent, class_labels=labels, inputs=x_train)
    pan_zoom_embed = PanZoomEmbedding(embed, win_size=(800, 600))
    pan_zoom_embed.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pan_zoom()  # Run the test function to verify the implementation
