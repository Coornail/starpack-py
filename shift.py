from os import terminal_size
from scipy.ndimage import shift
from scipy.optimize import basinhopping, brute
import numexpr as ne
import cv2
import numpy as np
from skimage.filters import threshold_otsu


def alignment_score(xshift, yshift, angle, ref, input):
    # out = rotate(input, angle, reshape=False)
    out = shift(input, (xshift, yshift), cval=-1)
    # diff = (abs(ref-out)).sum()
    diff = ne.evaluate('abs(ref-out)').sum()
    print(xshift, yshift, angle, diff)
    return diff


def find_best_shift_minimize(ref, inp):
    """
    Find the best alignment for an image based on a reference image.
    """

    # warp_matrix = np.eye(3, 3, dtype=np.float32)
    # criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10000, 0.1)
    # (cc, warp_matrix) = cv2.findTransformECC(ref, inp, warp_matrix,
    #                                          cv2.MOTION_HOMOGRAPHY, criteria, inputMask=None, gaussFiltSize=3)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(2 << 14)
    keypoints1, descriptors1 = orb.detectAndCompute(ref, None)
    keypoints2, descriptors2 = orb.detectAndCompute(inp, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(
        cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMINGLUT)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * 0.005)
    matches = matches[:numGoodMatches]
    print(matches)

    # Draw top matches
    imMatches = cv2.drawMatches(
        ref, keypoints1, inp, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    return h

    return warp_matrix
