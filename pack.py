from eta import ETA
from PIL import Image
from numpy import asarray, full_like
from numpy.core.fromnumeric import shape
from scipy.ndimage import shift
import numpy as np

from shift import find_best_shift_minimize
from image import alignment_window


def starpack(image_paths, darkframe_paths=[]):
    shifts = {image_paths[0]: [0.0, 0.0]}
    ref = asarray(Image.open(image_paths[0]))
    ref_bw = alignment_window(ref, 512)

    darkframe_master = darkframepack(ref, darkframe_paths)

    eta = ETA(len(image_paths))
    for i in range(1, len(image_paths)):
        img = asarray(Image.open(image_paths[i]))
        img_bw = alignment_window(img, 512)
        s = find_best_shift_minimize(ref_bw, img_bw)
        shifts[image_paths[i]] = s
        eta.print_status()
    eta.done()

    out = zeros_like(ref, dtype=float)
    for file_name in shifts:
        s = shifts[file_name]
        img = asarray(Image.open(file_name))
        shifted_img = shift(img, (s[0], s[1], 0))
        out += shifted_img

    out /= len(image_paths)
    out -= darkframe_master

    return out.astype(np.uint8)


def darkframepack(ref, image_paths):
    out = full_like(ref, 255.0, dtype=float)

    for i in range(len(image_paths)):
        out += asarray(Image.open(image_paths[i]))

    out /= (len(image_paths) + 1)
    out = 255 - out

    return out
