from numpy import empty


def to_grayscale(img):
    bw = empty((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            bw[i, j] = img[i, j, 0]*.2126 + \
                img[i, j, 1] * .7152 + img[i, j, 2] * .072

    return bw


def alignment_window(img, window_size):
    img_bw = to_grayscale(img)

    # Treshold image
    img_bw[img_bw < 128] = 0

    w, h = img_bw.shape
    return img_bw[round((w-window_size)/2):round((w+window_size)/2), round((h-window_size)/2):round((h+window_size)/2)]
