import numpy as np

from scipy.ndimage import shift
from scipy.ndimage import rotate

from color import to_grayscale


def find_best_shift(ref, input, min, max, step=1):
    bestx = 0
    besty = 0
    best_angle = 0
    bestdiff = float('inf')

    for x in np.arange(min, max, step):
        for y in np.arange(min, max, step):
            for angle in np.arange(min*10, max*10, step*10):
                out = rotate(input, angle, reshape=False)
                out = shift(out, (x, y))
                diff = (abs(ref-out)).sum()
                if diff < bestdiff:
                    bestx = x
                    besty = y
                    best_angle = angle
                    bestdiff = diff
        print((x+abs(min))/(abs(min)+max)*100, "%")

    return(bestx, besty, best_angle)


def find_alignment(loaded_images):
    alignments = [(0, 0, 0)]
    ref = to_grayscale(loaded_images[0])

    for i in range(1, len(loaded_images)):
        shift = find_best_shift(ref, to_grayscale(
            loaded_images[i]), -1, 1, step=1)
        alignments.extend([shift])

    return alignments
