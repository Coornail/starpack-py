from scipy.ndimage import shift
from scipy.optimize import basinhopping
import numexpr as ne


def alignment_score(xshift, yshift, angle, ref, input):
    # out = rotate(input, angle, reshape=False)
    out = shift(input, (xshift, yshift))
    # diff = (abs(ref-out)).sum()
    diff = ne.evaluate('abs(ref-out)').sum()
    # print(xshift, yshift, angle, diff)
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

    minimize_result = basinhopping(Minimizer(ref, input), [
                                   0.0, 0.0], niter=10, interval=10, seed=1, minimizer_kwargs={'options': {'eps': 0.5}})
    return minimize_result.x
