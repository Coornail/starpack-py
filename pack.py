from PIL import Image
from numpy import asarray, zeros_like
import numpy as np
from multiprocessing import Pool, cpu_count
import cv2

from shift import find_best_shift_minimize


def treshold(img, threshold):
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = img_bw.copy()
    img2[img2 < threshold] = 0
    # imwrite('bw.tif', img2)
    return img2


def find_threshold(img):
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bw[img_bw < 200] = 0
    return cv2.adaptiveThreshold(
        img_bw, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


def minimize(inp):
    ref = inp[0]
    image_path = inp[1]

    img = asarray(Image.open(image_path))
    threshold = find_threshold(ref)
    return(image_path, find_best_shift_minimize(treshold(ref, threshold), treshold(img, threshold)))


def starpack(image_paths, darkframe_paths=[], biasframe_paths=[]):
    ref = asarray(Image.open(image_paths[0]))

    # Prepare work
    work = []
    for image_path in image_paths[1:]:
        work.append((ref, image_path))

    # Worker threads
    p = Pool(cpu_count())
    s = p.map(minimize, work)

    shifts = {image_paths[0]: np.eye(3, 3, dtype=np.float32)}
    for shift_result in s:
        shifts[shift_result[0]] = shift_result[1]

    # Compile output file
    out = zeros_like(ref, dtype=float)
    darkframe_master = zeros_like(ref, dtype=float)
    biasframe_master = zeros_like(ref, dtype=float)
    if len(darkframe_paths):
        darkframe_master = starpack_unaligned(ref, darkframe_paths)

    if len(biasframe_paths):
        biasframe_master = starpack_unaligned(ref, biasframe_paths)

    shape = ref.shape
    for file_name in shifts:
        img = asarray(Image.open(file_name))
        corrected_img = img - darkframe_master
        corrected_img -= biasframe_master

        shifted_img = cv2.warpPerspective(corrected_img, shifts[file_name], (
            shape[1], shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        out += shifted_img

    out /= len(image_paths)

    # We are writing a 16 bit image
    out *= 2 << 7

    return out.astype(np.uint16)


def starpack_unaligned(ref, image_paths):
    out = zeros_like(ref, dtype=float)

    for i in range(len(image_paths)):
        out += asarray(Image.open(image_paths[i]))

    out /= (len(image_paths))

    return out
