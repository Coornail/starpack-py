import numpy as np
from scipy.ndimage import shift
from scipy.ndimage import rotate
from scipy.optimize import basinhopping

import numexpr as ne

from color import to_grayscale


def alignment_score(xshift, yshift, angle, ref, input):
    out = rotate(input, angle, reshape=False)
    out = shift(out, (xshift, yshift))
    # diff = (abs(ref-out)).sum()
    diff = ne.evaluate('abs(ref-out)').sum()
    print(xshift, yshift, angle, diff)
    return diff


class Minimizer(object):
    def __init__(self, ref, input):
        self.ref = ref
        self.input = input

    def __call__(self, guess):
        return round(alignment_score(guess[0], guess[1], 0, self.ref, self.input))


def find_best_shift_minimize(ref, input):
    # minimize_result = minimize(Minimizer(ref, input), [0.0, 0.0], bounds=[
    #    (-10, 10), (-10, 10)], options={'maxiter': 1000, 'disp': True, 'eps': 0.1})

    minimize_result = basinhopping(Minimizer(ref, input), [
                                   0.0, 0.0], niter=100, interval=10, disp=True, seed=1)
    print(minimize_result)
    return minimize_result.x


def find_alignment(loaded_images):
    alignments = [(0, 0, 0)]
    ref = to_grayscale(loaded_images[0])

    for i in range(1, len(loaded_images)):
        shift = find_best_shift(ref, to_grayscale(
            loaded_images[i]), -1, 1, step=1)
        alignments.extend([shift])

    return alignments
