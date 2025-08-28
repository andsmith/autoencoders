import logging
from windows import Window, EmbedWindow
import numpy as np
import cv2
import logging
import time
from colors import COLOR_SCHEME 

class TestWindow(Window):
    def __init__(self, t_max=5.0):
        self._t_max = t_max
        super().__init__(title='test window', init_size=(640, 480))
        self._t0 = time.perf_counter()

    def _get_frame(self):
        # Return a test frame (e.g., a blank image)
        frame= np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        txt_str = "Elapsed:  %.3f" % (time.perf_counter() - self._t0)
        cv2.putText(frame, txt_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        text_str = "Remaining: %.3f" % (self._t_max - (time.perf_counter() - self._t0))
        (width, height), _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.putText(frame, text_str, (self.size[0]-width-10   ,self.size[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame

    def mouse_callback(self, event, x, y, flags, param):
        # Handle mouse events (e.g., print coordinates)
        if event == cv2.EVENT_LBUTTONDOWN:
            logging.info(f"Mouse clicked at ({x}, {y}) in {self._win_name}")

def _test_win_base(n_sec=10):
    logging.basicConfig(level=logging.INFO)
    test_window = TestWindow(t_max=n_sec)
    test_window._init_cv2()
    t0 = time.perf_counter()
    while time.perf_counter()-t0 < n_sec:

        test_window.refresh()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return test_window



def test_embed_window():
    from load_ae import load_autoencoder
    #test_exp = load_autoencoder('VAE-results\digits_VAE-TORCH(digits-PCA(25,UW)_Encoder-64_Dlatent=16_RegLambda=0.0100).weights')
    test_exp = load_autoencoder(r'Dense-results\digits_Dense(digits-PCA(784,UW)_units=256-2048-256-2_dec-units=256_Drop(l=1,r=0.50)).weights.h5')
    # Simulate some interactions with the embed window
    tester.embed_window.mouse_callback(event=None, x=100, y=100, flags=None, param=None)
    # Add assertions to verify the expected behavior


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    #_test_win_base()
    test_embed_window()
    logging.info("Test window completed.")