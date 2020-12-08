import sys
from os import walk
from tifffile import imwrite

from pack import starpack


def collect_images(dir):
    files = []
    for (dirpath, dirnames, filenames) in walk(dir):
        files.extend([dirpath + "/" + filename for filename in filenames if filename.endswith(
            ".jpg") or filename.endswith(".png") or filename.endswith(".tif")])

    return files


if __name__ == "__main__":
    image_paths = collect_images(sys.argv[1])
    img = starpack(image_paths)
    imwrite('out.tif', img, photometric='rgb')
