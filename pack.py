from eta import ETA
from PIL import Image
from numpy import asarray, zeros_like
from scipy.ndimage import shift
import numpy as np
from multiprocessing import Pool, cpu_count

from shift import find_best_shift_minimize
from image import alignment_window


def minimize(input):
    ref_bw = input[0]
    image_path = input[1]

    img = asarray(Image.open(image_path))
    img_bw = alignment_window(img, 512)
    return (image_path, find_best_shift_minimize(ref_bw, img_bw))


def starpack(image_paths, darkframe_paths=[]):
    ref = asarray(Image.open(image_paths[0]))
    ref_bw = alignment_window(ref, 512)

    # Prepare work
    work = []
    for image_path in image_paths[1:]:
        work.append((ref_bw, image_path))

    # Worker threads
    p = Pool(cpu_count())
    s = p.map(minimize, work)

    shifts = {image_paths[0]: [0.0, 0.0]}
    for shift_result in s:
        shifts[shift_result[0]] = shift_result[1]

    # Compile output file
    out = zeros_like(ref, dtype=float)
    for file_name in shifts:
        s = shifts[file_name]
        img = asarray(Image.open(file_name))
        shifted_img = shift(img, (s[0], s[1], 0))
        out += shifted_img

    out /= len(image_paths)

    if len(darkframe_paths):
        darkframe_master = starpack_unaligned(ref, darkframe_paths)
        out -= darkframe_master

    # We are writing a 16 bit image
    out *= 2 << 7

    return out.astype(np.uint16)


def starpack_unaligned(ref, image_paths):
    out = zeros_like(ref, dtype=float)

    for i in range(len(image_paths)):
        out += asarray(Image.open(image_paths[i]))

    out /= (len(image_paths))

    return out