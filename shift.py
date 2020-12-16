from scipy.ndimage import shift
from scipy.optimize import basinhopping, brute
import numexpr as ne


def alignment_score(xshift, yshift, angle, ref, input):
    # out = rotate(input, angle, reshape=False)
    out = shift(input, (xshift, yshift), cval=-1)
    # diff = (abs(ref-out)).sum()
    diff = ne.evaluate('abs(ref-out)').sum()
    print(xshift, yshift, angle, diff)
    return diff


def find_best_shift_minimize(ref, input):
    """
    Find the best alignment for an image based on a reference image.
    """

    class Minimizer(object):
        def __init__(self, ref, input):
            self.ref = ref
            self.input = input

        def __call__(self, guess):
            return alignment_score(guess[0], guess[1], 0, self.ref, self.input)

    minimize_result = brute(Minimizer(ref, input), [
            (-400, 400), (-400, 400)], Ns=101)
    return basinhopping(Minimizer(ref, input), minimize_result, seed=1, minimizer_kwargs={'options': {'eps': 1, 'disp': 1}}).x
