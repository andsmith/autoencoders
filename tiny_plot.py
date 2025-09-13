"""
wrap matplotlib to get png of a plot
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

def tiny_plot(size_px, x,y,x_label=None, y_label=None,title=None,adjust_params=None, *args, **kwargs):

    dpi = 100
    figsize = (size_px[0]/dpi, size_px[1]/dpi)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, *args, **kwargs)
    if title:
        ax.set_title(title)
    if x_label:
        ax.set_xlabel(x_label)
    else:
        # turn off
        ax.set_xticks([])
    if y_label:
        ax.set_ylabel(y_label)
    else:
        # turn off
        ax.set_yticks([])

    
    # Adjust left margin smaller 50%
    if not adjust_params:
        fig.subplots_adjust(left=.33, bottom=.23)
    else:

        fig.subplots_adjust(**adjust_params)
    # Save the plot to a PNG in memory
    from io import BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    

    # Convert PNG buffer to numpy array
    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    plt.close(fig)  # Close the figure to free memory

    if img.shape[0]!= size_px[1] or img.shape[1]!= size_px[0]:
        img = cv2.resize(img, size_px)

    return img[:,:,::-1]  # convert BGR to RGB



def test_tiny_plot():
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)
    size = (200, 200)
    img = tiny_plot(size, x, y, "X", "sin(X)", color='blue')
    cv2.imshow("Tiny Plot", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_tiny_plot()
