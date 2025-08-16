# cython: boundscheck=False, wraparound=False, cdivision=True
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def draw_color_tiles_cython(np.ndarray[np.uint8_t, ndim=3] image,
                            np.ndarray[np.int32_t, ndim=2] locs_px,
                            np.ndarray[np.float32_t, ndim=3] gray_tiles,
                            np.ndarray[np.int32_t, ndim=1] color_labels,
                            np.ndarray[np.uint8_t, ndim=2] colors):
    """
    Cython version of draw_color_tiles_reference
    """
    cdef int N = gray_tiles.shape[0]
    cdef int Th = gray_tiles.shape[1]
    cdef int Tw = gray_tiles.shape[2]
    cdef int i, y, x, c
    cdef int img_x_low, img_x_high, img_y_low, img_y_high
    cdef int color_ind
    cdef float gray
    cdef np.uint8_t[:] color_row
    cdef np.uint8_t[:, :, :] img_view = image

    for i in range(N):
        img_x_low = locs_px[i, 0]
        img_y_low = locs_px[i, 1]
        img_x_high = img_x_low + Tw
        img_y_high = img_y_low + Th
        color_ind = color_labels[i]
        
        for y in range(Th):
            for x in range(Tw):
                gray = gray_tiles[i, y, x]
                for c in range(3):
                    img_view[img_y_low + y, img_x_low + x, c] = \
                        <np.uint8_t>((1.0 - gray) * img_view[img_y_low + y, img_x_low + x, c] +
                                     gray * colors[color_ind, c])