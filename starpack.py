from os import walk
from tifffile import imwrite
import argparse

from pack import starpack


def collect_images(dir):
    files = []
    for (dirpath, dirnames, filenames) in walk(dir):
        files.extend([dirpath + "/" + filename for filename in filenames if filename.endswith(
            ".jpg") or filename.endswith(".png") or filename.endswith(".tif")])

    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Stack deep-sky/planet images')
    parser.add_argument('lights', help='light directory')
    parser.add_argument('--darks', nargs='?', help='dark directory')
    parser.add_argument('--bias', nargs='?', help='bias directory')

    args = parser.parse_args()

    light_paths = collect_images(args.lights)
    dark_paths = []
    bias_paths = []
    if (args.darks):
        dark_paths = collect_images(args.darks)
    bias_paths = []
    if (args.bias):
        bias_paths = collect_images(args.bias)

    img = starpack(light_paths, darkframe_paths=dark_paths)
    imwrite('out.tif', img, photometric='rgb')
