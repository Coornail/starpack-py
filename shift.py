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
    minimize_result = basinhopping(Minimizer(ref, input), [
                                   0.0, 0.0], niter=100, interval=10, disp=True, seed=1, minimizer_kwargs={'options': {'eps': 0.5}})
    print(minimize_result)
    return minimize_result.x
