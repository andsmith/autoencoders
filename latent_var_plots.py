import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import logging
import seaborn as sns


class LatentDigitDist(object):
    """
    Plot a distribution of latent unit activations for each kind of digit.

    These will be like narrow-box plots with a thin line spanning +/- 3 standard deviations,
    drawn over a thick line spanning the interquartile range (IQR), and a dot representing the median.
    Outliers are drawn as single pixels.
    """

    def __init__(self, values, digit_labels, sts=2.5, colors=None):
        self.colors = colors if colors is not None else (np.array(sns.color_palette("husl", 10))*255.0).astype(int)
        self.values = values
        self.digit_labels = digit_labels
        self.sts = sts  # standard deviations for range around mean to draw line
        self._calc_stats()

    def _calc_stats(self):
        """
        Calculate statistics for the latent distributions.
        """
        self.stats = {'digits': {},
                      'range': (np.min(self.values), np.max(self.values))}

        def _mk_stats(values):

            mean = np.mean(values)
            std = np.std(values)
            iqr = np.percentile(values, 75) - np.percentile(values, 25)
            median = np.median(values)
            outliers = values[np.abs(values - mean) > self.sts * std]
            return {
                "mean": mean,
                "std": std,
                "iqr": iqr,
                "median": median,
                "outliers": outliers
            }

        for digit in np.unique(self.digit_labels):
            digit_values = self.values[self.digit_labels == digit]
            self.stats['digits'][digit] = _mk_stats(digit_values)
            if digit==2:
                self.stats['digits'][2]['outliers'] = np.array([-.5,0.5])

        self.stats['total'] = _mk_stats(self.values)

    def render(self,
               image,
               bbox,
               colors=None,
               digit_subset=None,
               thicknesses=(1, 3, 5),
               alphas=(.75, 0.65, 0.8, 1.0),
               val_range=None, centered=True, orient='vertical'):
        """
        Render the latent digit distribution on the given image.
        :param image:  H x W x 3  uint8 array
        :param bbox:   Dictionary with 'x' and 'y' keys containing tuples (min, max)
        :param colors:  Optional array of colors to use for each digit
        :param digit_subset: Optional list of digits to include in the plot
        :param thicknesses:  3-tuple, Line thicknesses for  3-sigma & outliers, IRQ, medians, respectively
        :param alphas:  4-tuple, Opacity values for the outliers, 3-sigma, IRQ, median respectively.
        :returns: bbox actually drawn in (can be smaller to make spacing even)
        """
        band_w = np.max(thicknesses) + thicknesses[1]
        dband_w = band_w*2  # double band for overall dist. plot.
        n_digits = 10 if digit_subset is None else len(digit_subset)+1  # include total distribution w/subsets
        width, height = bbox['x'][1] - bbox['x'][0], bbox['y'][1] - bbox['y'][0]
        px_scale = height if orient == 'vertical' else width
        px_b_range = list(bbox['x'] if orient == 'vertical' else bbox['y'])
        px_a_range = list(bbox['y'] if orient == 'vertical' else bbox['x'])
        locs = np.linspace(px_b_range[0], px_b_range[1], n_digits + 2)[1:-1]

        # make sure they're evenly spaced
        smallest = np.min(np.diff(locs.astype(int)))
        locs = int(locs[0]) + np.arange(n_digits+1, dtype=int) * smallest
        px_b_range[1] = locs[-1]  # adjust width so bounding box is tight
        locs = locs[:-1]

        low, high = self.stats['range'] if val_range is None else val_range

        if centered:
            dist = max(np.abs(low), np.abs(high))
            low = -dist
            high = dist

        val_range = high - low

        def draw_axis(thickness, alpha, color):
            t = int(thickness)
            if orient == 'vertical':
                y0 = int(px_a_range[0] + px_scale / 2.0 - t/2.0)
                y1 = y0 + t
                x0, x1 = px_b_range
            else:
                x0 = int(px_a_range[0] + px_scale / 2.0 - t/2.0)
                x1 = x0 + t
                y0, y1 = px_b_range
            logging.info("Drawing axis: (x0,x1,y0,y1)=%s", (x0, x1, y0, y1))
            color_shade = _fade_color(image[y0:y1, x0:x1], color, alpha)
            image[y0:y1, x0:x1] = color_shade

        def _draw_dist(digit_stats, x_loc, color, thick_mul=1.0):

            def _scale_value(val):
                return (((val - low) / val_range) * px_scale + px_a_range[0])

            def draw_dots(values, alpha, thickness=None):

                thickness = thicknesses[0] if thickness is None else thickness
                t = int(thickness * thick_mul)
                t += (1 - t % 2)
                x0 = int(x_loc - t/2)
                x1 = x0 + t

                y_vals = (_scale_value(values) + t/2).astype(int)-1
                y_vals = np.clip(y_vals, px_a_range[0], px_a_range[1] - t)  # Ensure within 

                
                for y in y_vals.reshape(-1,1):
                    small_y = int(y[0] - t/2.0-1)
                    big_y = small_y + t

                    big_y = min(big_y, px_a_range[1] - t)
                    small_y = max(small_y, px_a_range[0])

                    if orient == 'vertical':
                        color_shade = _fade_color(image[small_y:big_y, x0:x1], color, alpha)
                        image[small_y:big_y, x0:x1] = color_shade
                    else:
                        color_shade = _fade_color(image[x0:x1, small_y:big_y], color, alpha)
                        image[x0:x1, small_y:big_y] = color_shade

            def draw_line(small, big, thickness, alpha, min_h=0):
                t = int(thickness * thick_mul)
                t += (1 - t % 2)

                x0 = int(x_loc - t / 2)
                x1 = x0 + t
                small_y = int(_scale_value(small))
                big_y = int(_scale_value(big))
                if big_y < small_y:
                    small_y, big_y = big_y, small_y
                if min_h is not None and big_y-small_y < min_h:
                    big_y = small_y + min_h
                big_y = min(big_y, px_a_range[1] - t)
                small_y = max(small_y, px_a_range[0])
                if orient == 'vertical':
                    color_shade = _fade_color(image[small_y:big_y, x0:x1], color, alpha)
                    image[small_y:big_y, x0:x1] = color_shade
                else:
                    color_shade = _fade_color(image[x0:x1, small_y:big_y], color, alpha)
                    image[x0:x1, small_y:big_y] = color_shade

            mean = digit_stats['mean']
            std = digit_stats['std']
            iqr = digit_stats['iqr']
            outliers = digit_stats['outliers']
            median = digit_stats['median']

            # draw a zero-line first:
            draw_line(mean - self.sts * std, mean + self.sts * std, thicknesses[0], alphas[1])
            draw_line(mean - iqr, mean + iqr, thicknesses[1], alphas[2])
            draw_dots(median,  thickness=thicknesses[2], alpha=alphas[3])
            draw_dots(outliers, alphas[0], thickness=thicknesses[0])

        # Start drawing:

        bkg_color = image[bbox['y'][0], bbox['x'][0], :]
        total_color = (0, 0, 0) if np.mean(bkg_color) > 128 else (255, 255, 255)
        total_color = np.array(total_color)
        draw_axis(1, alpha=0.3, color=total_color)

        if digit_subset is None:
            for digit, x_pos in enumerate(locs):


                _draw_dist(self.stats['digits'][digit], x_pos, self.colors[digit])
        else:

            _draw_dist(self.stats['total'], locs[0], color=total_color, thick_mul=1.5)
            for ds_ind, digit in enumerate(digit_subset):
                _draw_dist(self.stats['digits'][digit], locs[ds_ind + 1], color=self.colors[digit])

        # Finally, return adjusted bbox
        if orient == 'vertical':
            new_bbox = {'x': (bbox['x'][0], px_b_range[1]),
                        'y': bbox['y']}
        else:
            new_bbox = {'x': bbox['x'],
                        'y': (bbox['y'][0], px_b_range[1])}

        return image, new_bbox


def draw_bbox(image, bbox, thickness=1, inside=True, color=(128, 128, 128)):
    """
    Set pixels just inside/outside the specified bounding box to the color indicated.
    :param image:  H x W x 3  uint8 array
    :param bbox:   Dictionary with 'x' and 'y' keys containing tuples (min, max)
    :param thickness: Thickness of the bounding box lines
    :param inside: If True, color the inside of the bbox; if False, color the outside
    :param color: Color to use for the bounding box
    """

    x_min, x_max = bbox['x']
    y_min, y_max = bbox['y']
    if inside:
        x_max -= thickness-1
        y_max -= thickness-1
        x_max += thickness-1
        y_max += thickness-1
    else:
        x_min -= thickness
        y_min -= thickness
        x_max += thickness
        y_max += thickness

    image[y_min:y_min+thickness, x_min:x_max] = color
    image[y_min:y_max, x_min:x_min+thickness] = color
    image[y_min:y_max, x_max-thickness:x_max] = color
    image[y_max-thickness:y_max, x_min:x_max] = color
    return image


def test_draw_bbox():
    blank = np.zeros((10, 10, 3), dtype=np.uint8)
    bboxes = [{'bbox': {'x': (2, 7), 'y': (2, 7)}, 'thickness': 1, 'inside': True},
              {'bbox': {'x': (2, 7), 'y': (2, 7)}, 'thickness': 1, 'inside': False},
              {'bbox': {'x': (2, 7), 'y': (2, 7)}, 'thickness': 2, 'inside': True},
              {'bbox': {'x': (2, 7), 'y': (2, 7)}, 'thickness': 2, 'inside': False}]
    fig, ax = plt.subplots(2, 2, figsize=(12, 3))
    ax = ax.flatten()
    for i, bbox in enumerate(bboxes):
        image = draw_bbox(blank.copy(), **bbox)
        ax[i].imshow(image)
        ax[i].set_title(f"bbox: {bbox['bbox']}, thickness: {bbox['thickness']}, inside: {bbox['inside']}")
    plt.show()


rand_state = [np.random.RandomState(42)]


def _mk_data(n=500):

    def _mk_digit(n):
        return rand_state[0].randn(1)*4.0 + rand_state[0].randn(n)

    digits = [i for i in range(10)]
    values = np.concatenate([_mk_digit(n) for _ in digits])
    digit_labels = np.concatenate([[d] * n for d in digits])
    return values, digit_labels


def _fade_color(bkg, color, alpha):
    return (1-alpha)*bkg + alpha*color


def test():
    image1 = np.zeros((70, 120, 3), dtype=np.uint8)
    image2 = np.zeros((120, 70, 3), dtype=np.uint8)
    image1[:] = 255
    image2[:] = 255
    image3, image4 = image1.copy(), image2.copy()
    # x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    # check = (x+y) % 2
    # image[check == 0] = (255, 255, 255)

    def _mk_flat_data(n, span=(-1, 1)):
        # like mk_data, but linearly spanning span[0] to span[1]
        values = np.linspace(span[0], span[1], n)
        value_mat = np.concatenate([values*(1/(i+1)) for i in range(10)], axis=0).flatten()
        value_mat[3*n:4*n] += 0.4
        label_mat = np.concatenate([i*np.ones(n) for i in range(10)]).astype(int)
        return value_mat, label_mat
    data = _mk_flat_data(400)
    bkg_color = image1[0, 0, :]
    dist = LatentDigitDist(*_mk_flat_data(500))
    bbox1 = {'y': (10, 60), 'x': (10, 110)}

    draw_bbox(image1, dist.render(image1, bbox1, val_range=(-2, 2), orient='horizontal')[1],
              thickness=1, inside=True, color=(256-bkg_color))

    draw_bbox(image3, dist.render(image3, bbox1, val_range=(-2, 2), thicknesses=[2, 2, 2], orient='horizontal')[1],
              thickness=1, inside=True, color=(256-bkg_color))

    bbox2 = {'x': (10, 60), 'y': (10, 110)}

    draw_bbox(image2,  dist.render(image2, bbox2, val_range=(-2, 2), orient='vertical')[1],
              thickness=1, inside=True, color=(256-bkg_color))

    draw_bbox(image4, dist.render(image4, bbox2, val_range=(-2, 2), orient='vertical', thicknesses=[2, 2, 2])[1],
              thickness=1, inside=True, color=(256-bkg_color))
    images = [image1, image2, image3, image4]
    fig, ax = plt.subplots(2, 2)
    ax = ax.flatten()
    for ax, img in zip(ax, images):
        ax.imshow(img)
    plt.show()


def test_latentdigitdist():
    image1 = np.zeros((480, 640, 3), dtype=np.uint8)
    image2 = np.zeros((640, 480, 3), dtype=np.uint8)
    image1[:] = 255
    image2[:] = 255
    bkg_color = image1[0, 0, :]

    def show_test(image, orient):
        spacing = 10

        if orient == 'vertical':
            width = 50
            height = 210
            n_bands = int(image.shape[1] / (width + spacing))
        else:
            width = 210
            height = 50
            n_bands = int(image.shape[0] / (height + spacing))

        x_offset = 20
        y_offset = 20

        for trial in range(n_bands * 2):
            rand_state[0] = np.random.RandomState(42)

            #if trial == n_bands:

            bbox = {'x': (x_offset, x_offset + width),
                    'y': (y_offset, y_offset + height)}
            dist = LatentDigitDist(*_mk_data(500))

            if trial >= n_bands:
                kwargs = dict(thicknesses=[3, 3, 3],
                              alphas=[0.8, .5, .75, 1.])
            else:
                kwargs = dict(thicknesses=[1, 3, 5],
                              alphas=[1, 1, 1, 1.])

            if trial % 2 == 0:
                kwargs['digit_subset'] = [0, 1, 3, 5, 8]

            d_bbox=dist.render(image, bbox, orient=orient, **kwargs)[1]

            draw_bbox(image, d_bbox, thickness=1, inside=True, color=(256-bkg_color))

            x_offset += width + spacing

            if x_offset + width > image.shape[1]:
                y_offset += height + spacing
                x_offset = 20

        plt.imshow(image)
        plt.show()

    show_test(image1, 'vertical')
    show_test(image2, 'horizontal')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_make_random_cov(n_points=10000, plot=True)
    # test_make_data(d=2, plot=True)

    test_latentdigitdist()

    #test()

    # Uncomment to test with higher dimensions
    # test_make_data(d=10, plot=True)
