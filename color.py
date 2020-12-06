from numpy import empty
from numpy import dot


def to_grayscale(img):
    return dot(img[..., :3], [0.2989, 0.5870, 0.1140])


def alignment_window(img, window_size):
    img_bw = to_grayscale(img)

    # Treshold image
    img_bw[img_bw < 128] = 0

    w, h = img_bw.shape
    return img_bw[round((w-window_size)/2):round((w+window_size)/2), round((h-window_size)/2):round((h+window_size)/2)]
