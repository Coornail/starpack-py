import cv2
import numpy as np


def find_best_shift_minimize(ref, inp):
    """
    Find the best alignment for an image based on a reference image.
    """

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
