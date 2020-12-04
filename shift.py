import numpy as np
from scipy.ndimage import shift
from scipy.ndimage import rotate
from scipy.optimize import basinhopping
from os import walk
import numexpr as ne
from numpy import asarray, zeros_like
from PIL import Image
import sys

from color import to_grayscale


def alignment_score(xshift, yshift, angle, ref, input):
    # out = rotate(input, angle, reshape=False)
    out = shift(input, (xshift, yshift))
    # diff = (abs(ref-out)).sum()
    diff = ne.evaluate('abs(ref-out)').sum()
    # print(xshift, yshift, angle, diff)
    return diff


class Minimizer(object):
    def __init__(self, ref, input):
        self.ref = ref
        self.input = input

    def __call__(self, guess):
        return round(alignment_score(guess[0], guess[1], 0, self.ref, self.input))


def find_best_shift_minimize(ref, input):
    return [0.0, 0.0]
    minimize_result = basinhopping(Minimizer(ref, input), [
                                   0.0, 0.0], niter=10, interval=10, seed=1, minimizer_kwargs={'options': {'eps': 0.5}})
    return minimize_result.x


def starpack(image_paths):
    shifts = {image_paths[0]: [0.0, 0.0]}
    ref = asarray(Image.open(image_paths[0]))
    ref_bw = to_grayscale(ref)
    for i in range(1, len(image_paths)):
        img = asarray(Image.open(image_paths[i]))
        img_bw = to_grayscale(img)
        s = find_best_shift_minimize(ref_bw, img_bw)
        shifts[image_paths[i]] = s

    out = zeros_like(ref, dtype=float)
    for file_name in shifts:
        s = shifts[file_name]
        img = asarray(Image.open(file_name))
        shifted_img = shift(img, (0, s[0], s[1]))
        out += shifted_img

    out /= len(image_paths)

    return out.astype(np.uint8)


def collect_images(dir):
    files = []
    for (dirpath, dirnames, filenames) in walk(dir):
        files.extend([dirpath + "/" + filename for filename in filenames if filename.endswith(
            ".jpg") or filename.endswith(".png")])

    return files


def main():
    image_paths = collect_images(sys.argv[1])
    img = starpack(image_paths)
    print(img)
    Image.fromarray(img).save("out.tif")


if __name__ == "__main__":
    main()
