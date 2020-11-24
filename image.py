from PIL import Image
from numpy import asarray


def load(file_name):
    return asarray(Image.load(file_name))
