from numpy import dot
import cv2


def to_grayscale(img):
    """
    Converts an image to grayscale

    Based on https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def alignment_window(img, window_size):
    """
    Get a small part of the image from the center to perform alignment on.
    """
    img_bw = to_grayscale(img)

    # Treshold image
    img_bw[img_bw < 128] = 0

    w, h = img_bw.shape
    return img_bw[round((w-window_size)/2):round((w+window_size)/2), round((h-window_size)/2):round((h+window_size)/2)]
