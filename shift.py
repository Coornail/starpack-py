import numpy as np
from scipy.ndimage import shift
from scipy.ndimage import rotate
from scipy.optimize import minimize
from multiprocessing import Pool
import functools

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
                    print(x, y, angle, diff)
                    bestx = x
                    besty = y
                    best_angle = angle
                    bestdiff = diff
        print((x+abs(min))/(abs(min)+max)*100, "%")

    return(bestx, besty, best_angle)


def alignment_score(xshift, yshift, angle, ref, input):
    out = rotate(input, angle, reshape=False)
    out = shift(out, (xshift, yshift))
    diff = (abs(ref-out)).sum()
    print(xshift, yshift, angle, diff)
    return diff


class Minimizer(object):
    def __init__(self, ref, input):
        self.ref = ref
        self.input = input

    def __call__(self, guess):
        return alignment_score(guess[0], guess[1], guess[2], self.ref, self.input)


def find_best_shift_minimize(ref, input):
    print(minimize(Minimizer(ref, input), [0.0, 0.0, 0.0]))


def find_alignment(loaded_images):
    alignments = [(0, 0, 0)]
    ref = to_grayscale(loaded_images[0])

    for i in range(1, len(loaded_images)):
        shift = find_best_shift(ref, to_grayscale(
            loaded_images[i]), -1, 1, step=1)
        alignments.extend([shift])

    return alignments


def find_alignment_parallel(loaded_images):
    ref = to_grayscale(loaded_images[0])

    alignmenter = functools.partial(find_best_shift, ref=ref)

    alignments = [(0, 0, 0)]
    p = Pool(4)

    alignments.extend(p.map(lambda i: find_best_shift(
        ref, to_grayscale(loaded_images[i]), -1, 1, step=1), range(1, len(loaded_images))))

    return alignments
