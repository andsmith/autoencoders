import cv2
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import logging
import seaborn as sns
from torch import layout, std
from colors import COLORS, MPL_CYCLE_COLORS, bgr2rgb
from util import fit_spaced_intervals
import cv2


class LatentDigitDist(object):
    """
    Plot a distribution of latent unit activations for each kind of digit.

    These will be like narrow-box plots with a thin line spanning +/- 3 standard deviations,
    drawn over a thick line spanning the interquartile range (IQR), and a dot representing the median.
    Outliers are drawn as single pixels.
    """

    def __init__(self, values, digit_labels, sts=2.5):
        """
        :param values: Latent unit activations
        :param digit_labels: Corresponding digit labels
        :param sts: Standard deviations defining outlier threshold.
        """
        self.values = values
        self.digit_labels = digit_labels
        self.n_digits = len(np.unique(digit_labels))
        self.sts = sts  # standard deviations for range around mean to draw line
        self._calc_stats()

    def _calc_stats(self):
        """
        Calculate statistics for the latent distributions.
        """
        self.stats = {'digits': {},
                      'range': (np.min(self.values), np.max(self.values)),
                      'total': None}

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

        for digit in range(self.n_digits):
            digit_values = self.values[self.digit_labels == digit]
            self.stats['digits'][digit] = _mk_stats(digit_values)
            if digit == 2:
                self.stats['digits'][2]['outliers'] = np.array([-.5, 0.5])

        self.stats['total'] = _mk_stats(self.values)

    def calc_size_layout(self, orient, thicknesses, space, n_digits=10):
        """
        Compute where everything goes on the non-value axis, i.e. where
        the center of each boxplot is.
        """
        band_size = np.max(thicknesses)
        if orient == 'vertical':
            y = 0
            band_spans = []
            for d in range(n_digits):
                band_spans.append((y, y + band_size))
                y += band_size + space
            size = y-space
        else:
            x = 0
            band_spans = []
            for d in range(n_digits):
                band_spans.append((x, x + band_size))
                x += band_size + space
            size = x-space
        layout = {'bands': band_spans,
                  'size': size}

        return layout

    def render(self,
               image,
               loc_xy,
               scale,
               orient='vertical',
               thicknesses_px=(1, 3, 5, 7),
               separation_px=2,
               alphas=(.75, 0.6, 0.8, 1.0),
               digit_subset=None,
               val_span=None,
               colors=None,
               centered=False,
               show_axis=True):
        """
        Render the latent digit distribution on the given image.
        :param image:  H x W x 3  uint8 array
        :param loc_xy:  (x, y) coordinates for the top-left corner of the plot
        :param scale:   Width of the plot area if orient='horizontal', height if orient='vertical'
        :param digit_subset: Optional list of digits to include in the plot
        :param thicknesses_px:  4-tuple, Line thicknesses in pixels  for  outliers, 3-sigma , IRQ, medians, respectively
        :param separation_px:  space between lines in pixels.
        :param alphas:  4-tuple, Opacity values for the outliers, 3-sigma, IRQ, median respectively.
        :returns: bbox actually drawn in (can be smaller to make spacing even)
        """
        colors =  (np.array(sns.color_palette("husl", 12))*255.0).astype(int)  #colors if colors is not None else

        bkg_color = image[loc_xy[1], loc_xy[0], :]
        total_color = (0, 0, 0) if np.mean(bkg_color) > 128 else (255, 255, 255)
        total_color = np.array(total_color)
        n_digits = 10 if digit_subset is None else len(digit_subset)+1  # include total distribution w/subsets
        layout = self.calc_size_layout(orient, thicknesses_px, separation_px, n_digits=n_digits)
        low, high = self.stats['range'] if val_span is None else val_span
        val_span = (low, high)
        val_range = high-low

        plot_px_range = layout['size']
        value_px_range = scale

        if orient == 'vertical':
            plot_px_span = loc_xy[0], loc_xy[0] + layout['size']
            value_px_span = loc_xy[1], loc_xy[1] + scale
            width = layout['size']
            height = scale
        elif orient == 'horizontal':
            plot_px_span = loc_xy[1], loc_xy[1] + layout['size']
            value_px_span = loc_xy[0], loc_xy[0] + scale

            width = scale
            height = layout['size']
        else:
            raise ValueError("Unknown orientation: %s" % orient)

        def scale_value(val, flip=False):
            if flip:
                return (((high - val) / val_range) * value_px_range + value_px_span[0])
            return (((val - low) / val_range) * value_px_range + value_px_span[0])

        def _fade_color(x_span, y_span,  color, alpha):
            bkg = image[y_span[0]:y_span[1], x_span[0]:x_span[1]]
            patch = (1 - alpha) * bkg + alpha * color
            image[y_span[0]:y_span[1], x_span[0]:x_span[1]] = patch

        def draw_axis():
            t = thicknesses_px[0]
            axis_pos = int(scale_value(0.0))
            if orient == 'vertical':
                x0, x1 = plot_px_span
                y0 = axis_pos - t//2
                y1 = y0 + t
            elif orient == 'horizontal':
                x0 = axis_pos - t//2
                x1 = x0 + t
                y0, y1 = plot_px_span
            _fade_color((x0, x1), (y0, y1), total_color, alphas[0])

        def draw_line(value_span, width_span, thickness, alpha, color):
            small, big = value_span

            shift = ((width_span[1] - width_span[0]) - thickness) // 2

            small_v = int(scale_value(small))
            big_v = int(scale_value(big))
            if big_v < small_v:
                small_v, big_v = big_v, small_v
            big_v = np.clip(big_v, value_px_span[0], value_px_span[1])
            small_v = np.clip(small_v, value_px_span[0], value_px_span[1])

            if orient == 'vertical':
                x0 = width_span[0] + shift
                x1 = x0 + thickness
                y0, y1 = small_v, big_v

            else:
                x0, x1 = small_v, big_v
                y0 = width_span[0] + shift
                y1 = y0 + thickness
            _fade_color((x0, x1), (y0, y1), color, alpha)

        def draw_dots(values, width_span, size, alpha, color):
            shift = ((width_span[1] - width_span[0]) - size) // 2
            for value in values:
                x = (scale_value(value)).astype(int)

                small_v = x - size // 2
                big_v = small_v + size
                big_v = np.clip(big_v, value_px_span[0], value_px_span[1])
                small_v = np.clip(small_v, value_px_span[0], value_px_span[1])

                if orient == 'vertical':
                    x0 = width_span[0] + shift
                    x1 = x0 + size
                    y0, y1 = small_v, big_v

                else:
                    x0, x1 = small_v, big_v
                    y0 = width_span[0] + shift
                    y1 = y0 + size
                _fade_color((x0, x1), (y0, y1), color, alpha)

        def draw_dist(stats, band_ind, color):
            #  Draw a single row/column plot
            offset = (loc_xy[0] if orient == 'vertical' else loc_xy[1])
            band_span = np.array(layout['bands'][band_ind]) + offset
            mean, median, std = stats['mean'], stats['median'], stats['std']
            iqr, outliers = stats['iqr'], stats['outliers']
            width_span = np.array(layout['bands'][band_ind]) + offset
            draw_line(value_span=(mean - self.sts * std, mean + self.sts * std),
                      width_span=width_span,
                      thickness=thicknesses_px[1], alpha=alphas[1], color=color)

            draw_line(value_span=(mean - iqr, mean + iqr), width_span=width_span,
                      thickness=thicknesses_px[2], alpha=alphas[2], color=color)
            draw_dots([median], width_span=width_span, size=thicknesses_px[3],
                      alpha=alphas[3], color=color)
            draw_dots(outliers, width_span=width_span, size=thicknesses_px[0], alpha=alphas[0], color=color)

        draw_axis()
        # import ipdb; ipdb.set_trace()

        if digit_subset is None:

            for digit in range(10):
                draw_dist(self.stats['digits'][digit], band_ind=digit, color=colors[digit])
        else:
            draw_dist(self.stats['total'], band_ind=0, color=total_color)
            for ds_ind, digit in enumerate(digit_subset):
                draw_dist(self.stats['digits'][digit], band_ind=ds_ind + 1, color=colors[ds_ind])

        if orient=='vertical':
            bbox = {'x': plot_px_span,
                    'y': value_px_span}
            
        else:

            bbox = {'x': value_px_span,
                    'y': plot_px_span}
        return bbox
        """
        width, height = bbox['x'][1] - bbox['x'][0], bbox['y'][1] - bbox['y'][0]
        px_scale = height if orient == 'vertical' else width
        px_digit_range = list(bbox['x'] if orient == 'vertical' else bbox['y'])
        px_value_range = list(bbox['y'] if orient == 'vertical' else bbox['x'])

        band_spans = fit_spaced_intervals(px_digit_range,
                                          n_digits,
                                          spacing_fraction=.1,
                                          min_spacing=separation_px,
                                          fill_extent=False)
        band_width = band_spans[0][1] - band_spans[0][0]
        # adjust so they're all odd spaced.
        if band_width % 2 == 0:
            band_width -= 1

        band_centers = [(span[0] + span[1]) // 2 for span in band_spans]

        thicknesses_rel = thicknesses_rel / np.max(thicknesses_rel)
        thicknesses = (thicknesses_rel*band_width).astype(int)

        low, high = self.stats['range'] if val_range is None else val_range

        if centered:
            dist = max(np.abs(low), np.abs(high))
            low = -dist
            high = dist

        val_range = high - low
        print("--> Val Range:", val_range, px_digit_range)

        def _draw_dist(digit_stats, band_, color, thick_mul=1.0):

            def _scale_value(val):
                return (((val - low) / val_range) * px_scale + px_value_range[0])

            def draw_axis(thickness, alpha, color):
                t = int(thickness)

                if orient == 'vertical':
                    y0 = int(_scale_value(0.0)) - t//2
                    y1 = y0 + t
                    x0, x1 = px_digit_range
                else:
                    x0 = int(_scale_value(0.0)) - t//2
                    x1 = x0 + t
                    y0, y1 = px_digit_range
                color_shade = _fade_color(image[y0:y1, x0:x1], color, alpha)
                image[y0:y1, x0:x1] = color_shade

            def draw_dots(values, alpha, thickness=None):

                thickness = thicknesses[0] if thickness is None else thickness
                t = int(thickness * thick_mul)
                t += (1 - t % 2)
                x0 = int(x_loc)
                x1 = x0 + t

                y_vals = (_scale_value(values) + t/2).astype(int)-1
                y_vals = np.clip(y_vals, px_value_range[0], px_value_range[1] - t)  # Ensure within

                for y in y_vals.reshape(-1, 1):
                    small_y = int(y[0] - t/2.0-1)
                    big_y = small_y + t

                    big_y = min(big_y, px_value_range[1] - t)
                    small_y = max(small_y, px_value_range[0])

                    if orient == 'vertical':
                        color_shade = _fade_color(image[small_y:big_y, x0:x1], color, alpha)
                        image[small_y:big_y, x0:x1] = color_shade
                    else:
                        color_shade = _fade_color(image[x0:x1, small_y:big_y], color, alpha)
                        image[x0:x1, small_y:big_y] = color_shade

            def draw_line(band_ind, small, big, min_h=0):
                thickness, alpha = thicknesses[band_ind], alphas[band_ind]
                low, high = band_spans[band_ind]
                band_center = d
                small_y = int(_scale_value(small))
                big_y = int(_scale_value(big))
                if big_y < small_y:
                    small_y, big_y = big_y, small_y
                if min_h is not None and big_y-small_y < min_h:
                    big_y = small_y + min_h
                big_y = min(big_y, px_value_range[1] - t)
                small_y = max(small_y, px_value_range[0])
                if orient == 'vertical':
                    x0 = int(x_loc - thickness/2.0)
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
            draw_axis(thickness=1, color=total_color, alpha=0.3)
            draw_line(mean - self.sts * std, mean + self.sts * std, thicknesses[1], alphas[1])
            # draw_line(mean - iqr, mean + iqr, thicknesses[2], alphas[2])
            # draw_dots(median,  thickness=thicknesses[3], alpha=alphas[3])
            # draw_dots(outliers, alphas[0], thickness=thicknesses[0])

        # Start drawing:

        if digit_subset is None:
            for digit, span in enumerate(band_spans):

                _draw_dist(self.stats['digits'][digit], span, self.colors[digit])
        else:
            _draw_dist(self.stats['total'], band_spans[0], color=total_color)

            for ds_ind, digit in enumerate(digit_subset):

                digit_color = self.colors[ds_ind]
                _draw_dist(self.stats['digits'][digit], band_spans[ds_ind + 1], color=digit_color)

        # Finally, return adjusted bbox
        if orient == 'vertical':
            new_bbox = {'x': (bbox['x'][0], px_digit_range[1]),
                        'y': bbox['y']}
        else:
            new_bbox = {'x': bbox['x'],
                        'y': (bbox['y'][0], px_digit_range[1])}

        return image, new_bbox
        """


def test_latentdigitdist():
    image1 = np.zeros((650, 1000, 3), dtype=np.uint8)
    image2 = np.zeros((1000, 650, 3), dtype=np.uint8)
    image1[:] = 255
    image2[:] = 255
    bkg_color = image1[0, 0, :]
    rand_state[0] = np.random.RandomState(1)

    def show_test(image, orient):
        spacing = 10
        value_axis_scale = 200
        plot_axis_scale = 70

        if orient == 'vertical':
            width = plot_axis_scale
            height = value_axis_scale
            n_bands = int(image.shape[1] / (width + spacing))
        else:
            width = value_axis_scale
            height = plot_axis_scale
            n_bands = int(image.shape[0] / (height + spacing))

        x_offset = 20
        y_offset = 20

        for trial in range(n_bands * 2):
            # rand_state[0] = np.random.RandomState(3)  # each plot is the same

            # if trial == n_bands:

            bbox = {'x': (x_offset, x_offset + width),
                    'y': (y_offset, y_offset + height)}
            data = _mk_data(5)
            dist = LatentDigitDist(*data)

            if trial >= n_bands:
                kwargs = dict(thicknesses_px=[3, 3, 3, 3],
                              alphas=[0.8, .5, .75, 1.])
            else:
                kwargs = dict(thicknesses_px=[2, 1, 3, 5],
                              alphas=[0.8, .5, .75, 1.])

            if trial % 2 == 0:
                kwargs['digit_subset'] = [0, 1, 3, 5, 8]
            loc_xy = (bbox['x'][0], bbox['y'][0])

            d_bbox = dist.render(image, loc_xy, scale=value_axis_scale, orient=orient, **kwargs)
            cv2.putText(image, "%i" % trial, (d_bbox['x'][0], d_bbox['y'][0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, lineType=cv2.LINE_AA)

            draw_bbox(image, d_bbox, thickness=1, inside=False, color=(256-bkg_color))

            if orient == 'vertical':
                x_offset += width + spacing
                if x_offset + width > image.shape[1]:
                    y_offset += height + 200
                    x_offset = 20
            else:
                y_offset += height + spacing
                if y_offset + height > image.shape[0]:
                    y_offset = 20
                    x_offset += width + spacing
        plt.imshow(image)
        plt.show()
    show_test(image2, 'horizontal')
    show_test(image1, 'vertical')


class LatentCodeSet(object):
    """
    A LatentDigitDist object for each encoding unit.
    """

    def __init__(self, test_codes, test_labels, digit_subset=None, colors=None):
        self.test_codes = test_codes
        self.test_labels = test_labels
        self.digit_subset = digit_subset        
        if colors is None:
            if self.digit_subset is not None:
                colors = [np.array(MPL_CYCLE_COLORS[i]) for i in range(len(self.digit_subset))]
            else:
                colors = (np.array(sns.color_palette("husl", 10))*255.0).astype(int).tolist()
        self.colors = colors


        self.ldds, self.code_size = LatentCodeSet._init_dists(self.test_codes,
                                                              self.test_labels)

    @staticmethod
    def calc_thicknesses(n_codes):
        if n_codes < 10:
            return [3, 5, 7, 9], 4
        if n_codes < 30:
            return [1, 3, 5, 7], 1
        return [1, 1, 1, 1], 1

    @staticmethod
    def calc_size(n_codes, bbox, orient='vertical', sep_factor=.2, min_sep_px=2, equal_sizes=False):
        """
        Determine the scale (height/width) required to fit all codes in the bbox.
        :param bbox:  {'x': (x_min, x_max), 'y': (y_min, y_max)}
        :param orient: The orientation of the layout ('vertical' or 'horizontal')
        :param sep_factor: The factor by which to separate each code
        :param min_sep_px: The minimum separation in pixels
        :param equal_sizes: If True, all bands of the histogram will have equal sizes (good for
           large numbers of codes)
        :returns: (thicknesses_px, separation_px)
        """

        return LatentCodeSet.calc_thicknesses(n_codes), 2

        box_wh = bbox['x'][1] - bbox['x'][0], bbox['y'][1] - bbox['y'][0]
        scaling_dim_size = box_wh[0] if orient == 'vertical' else box_wh[1]
        if equal_sizes:
            rel_t = np.ones(4)
            size_incr = 1.0
        else:
            rel_t = np.array([1, 3, 5, 7], dtype=(float))
            size_incr = 2.0

        for size_base in range(1, scaling_dim_size, size_incr):
            thicknesses_rel = (rel_t + size_base)

    @staticmethod
    def _init_dists(codes, labels):
        ldds = []
        code_size = codes.shape[1]
        for code_unit in range(code_size):
            ldds.append(LatentDigitDist(codes[:, code_unit], labels))
        return ldds, code_size

    def get_latent_distribution(self):
        # Compute the latent distribution for each digit
        pass

    def _calc_layout(self, size_wh, bbox, orient, n_codes, pad_frac):
        """
        Determine the layout for the latent code visualizations:
            height of each code's bbox,
            thicknesses of lines
            padding between each bbox
        """
        x_span, y_span = bbox['x'], bbox['y']
        w, h = (x_span[1] - x_span[0], y_span[1] - y_span[0])

        if orient == 'horizontal':

            bbox_y = np.array(fit_spaced_intervals(y_span, n_codes, spacing_fraction=pad_frac))
            bbox_x = np.zeros_like(bbox_y) + np.array(x_span).reshape(1, 2)
        else:

            # Compute layout for horizontal orientation
            bbox_x = np.array(fit_spaced_intervals(x_span, n_codes, spacing_fraction=pad_frac))
            bbox_y = np.zeros_like(bbox_x) + np.array(y_span).reshape(1, 2)
        bboxes = [dict(x=tuple(x0.tolist()), y=tuple(y0.tolist())) for x0, y0 in zip(bbox_x, bbox_y)]
        return bboxes

    def _calc_lighting(self, size_wh, orient, n_codes, pad_frac):
        # STUB, TODO:  calculate good sizing for these
        thicknesses = (1, 3, 3, 3)
        alphas = (.75, 0.65, 0.8, 1.0)
        return thicknesses, alphas

    def render(self, image, bbox, orient='horizontal', same_scale=False, sep_factor=0.2, **kwargs):
        """
        Render the distribution of each code unit's activations.

        :param image: The image to render on
        :param bbox: The bounding box to draw inside
        :param orient: The orientation of each code's boxplot
        :param same_scale: If True, use the same scale for all code unit, otherwise center each plot & scale it to fit all points.
        :param sep_factor: Each code's set of boxplots will be separated by this factor.

        """
        scale = bbox['x'][1]-bbox['x'][0] if orient == 'horizontal' else bbox['y'][1]-bbox['y'][0]
        size_wh = image.shape[:2][::-1]
        # bboxes = self._calc_layout(size_wh, bbox, orient, n_codes=len(self.ldds), pad_frac=sep_factor)
        thicknesses, alphas = self._calc_lighting(size_wh, orient, len(self.ldds), pad_frac=sep_factor)
        # Draw each latent code's distribution
        x0, y0 = bbox['x'][0], bbox['y'][0]
        sep_px=None
        draw_bbox(image, bbox, thickness=1, inside=True, color=[0, 255, 0])
        
        for i, ldd in enumerate(self.ldds):
            # Compute the layout for this code
            # Render the distribution
            pos = (x0, y0)
            try:
                print("Rendering plot for code unit %d at position %s" % (i, pos)   )
                bbox_out = ldd.render(image, loc_xy=pos, scale=scale, orient=orient,
                                    thicknesses_px=thicknesses, alphas=alphas, separation_px=2,
                                    centered=True, show_axis=True,colors=self.colors)
            except Exception as e:
                print("Plot failed", sep_px)    
            draw_bbox(image, bbox_out, thickness=1, inside=True, color=[255, 0, 0])
            #import ipdb; ipdb.set_trace()
            if orient == 'vertical':
                sep_px = sep_px if sep_px is not None else  int((bbox_out['x'][1] - bbox_out['x'][0]) * sep_factor)
                x0 = bbox_out['x'][1] + sep_px
            elif orient=='horizontal':
                sep_px = sep_px if sep_px is not None else int((bbox_out['y'][1] - bbox_out['y'][0]) * sep_factor)
                y0 = bbox_out['y'][1] + sep_px
        return sep_px

def test_latent_codeset(code_size=10):
    image_size_wh = (300, 600)
    blankH = np.zeros((image_size_wh[1], image_size_wh[0], 3), dtype=np.uint8)
    blankV = np.zeros((image_size_wh[0], image_size_wh[1], 3), dtype=np.uint8)
    bkg_color = np.array(COLORS['OFF_WHITE_RGB'])
    blankH[:] = bkg_color
    blankV[:] = bkg_color
    labels, codes = mkd_dataset(code_size=code_size)
    subset = (0,1,3,5,8)
    def _test_orient(orient, bbox, image):

        draw_bbox(image, bbox, thickness=1, inside=True, color=(128, 128, 128))
        ls = LatentCodeSet(test_codes=codes, test_labels=labels, digit_subset=[0, 1, 3, 8])
        thic, sep = ls.calc_size(code_size, bbox, orient=orient, equal_sizes=True, sep_factor=0.2)
        print(thic,sep)
        loc_xy = bbox['x'][0], bbox['y'][0]
        ls.render(image, bbox, orient=orient,digit_subset=subset, same_scale=True, separation_px=sep, thicknesses_px=thic)
        return image

    bboxH = {'x': (10, image_size_wh[0]-10), 'y': (10, image_size_wh[1]-10)}
    bboxV = {'x': (10, image_size_wh[1]-10), 'y': (10, image_size_wh[0]-10)}
    img_h = _test_orient('horizontal', bboxH, blankH)
    img_v = _test_orient('vertical', bboxV, blankV)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img_h)
    ax[0].axis('off')
    ax[1].imshow(img_v)
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()


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


def test_Mk_data():
    data = _mk_data(500)
    for digit in range(10):
        plt.scatter(data[0][data[1] == digit], np.ones_like(data[0][data[1] == digit]) * digit, label=str(digit))
    plt.legend()
    plt.show()


def _fade_color(bkg, color, alpha):
    return (1-alpha)*bkg + alpha*color


def calc_scale(n_pixels, n_plots):
    if n_plots < 16:
        t = 5
    elif n_plots < 32:
        t = 3
    else:
        t = 3
    # Keep thickness odd!
    height = int(t*3)
    pad = int(max(3, t*2))
    return height, t, pad


def mkd_dataset(code_size=30, n_samples=10000):

    digit_labels = np.random.randint(0, 10, n_samples)
    codes = np.zeros((n_samples, code_size), dtype=np.float32)
    for c in range(code_size):
        for d in range(10):
            mask = digit_labels == d
            mean = np.random.rand() * 0.5 + 0.5
            std = np.random.rand() * 0.2
            codes[mask, c] = np.random.normal(mean, std, size=np.sum(mask))
    return digit_labels, codes


def test_latent_compact():
    """


    TODO:  MOVE TO CLASS  (is now LatentCodeSet)

    Create an image comparing VAE latent variable distributions.
    Each column is a model, under it are horizontally oriented LatentDigitDist plots of
        each latent variable in that model .
    """
    model_sizes = [1, 4, 8, 16, 32, 64,]
    grid_shape = np.array((1, len(model_sizes)))
    model_data = [[_mk_data(500) for _ in range(n)] for n in model_sizes]
    image_size = np.array((1800, 900))
    blank = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    bkg_color = np.array(COLORS['OFF_WHITE_RGB'])
    blank[:] = bkg_color
    dist_size_wh = np.array((200, 13))  # each LatentDigitDist plot
    digit_subset = [1, 3, 8]

    colors = [COLORS['MPL_BLUE_RGB'],
              COLORS['MPL_ORANGE_RGB'],
              COLORS['MPL_GREEN_RGB']]

    colors = [np.array(c) for c in colors]

    for i, (model_size, data) in enumerate(zip(model_sizes, model_data)):

        model_pad_x = ((image_size[0] - grid_shape[1] * dist_size_wh[0]) / (grid_shape[1] + 1))

        x = int(i * (dist_size_wh[0] + model_pad_x) + model_pad_x)
        y = 10
        thick, sep = calc_thicknesses(model_size)
        pad_y = int(1.5 * sep)

        for j, latent_data in enumerate(data):

            # draw_bbox(blank, bbox, thickness=1, inside=True, color=(256 - bkg_color))
            loc_xy = x, y
            scale = dist_size_wh[0]  # width of the plot area
            ldd = LatentDigitDist(latent_data[0], latent_data[1], colors=colors)
            d_bbox = ldd.render(blank, loc_xy=loc_xy, scale=scale,
                                orient='horizontal',
                                centered=True, show_axis=False,
                                thicknesses_px=thick,
                                separation_px=sep, alphas=[.2, .5, .5, 1],
                                digit_subset=digit_subset)
            y = d_bbox['y'][1] + pad_y

            # draw_bbox(blank, d_bbox, thickness=1, inside=True, color=(256 - bkg_color))
    plt.imshow(blank)
    plt.gca().set_axis_off()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_make_random_cov(n_points=10000, plot=True)
    # test_make_data(d=2, plot=True)
    # test_Mk_data()

    # test_latentdigitdist()
    # test_latent_compact()
    test_latent_codeset()

    # test()

    # Uncomment to test with higher dimensions
    # test_make_data(d=10, plot=True)
