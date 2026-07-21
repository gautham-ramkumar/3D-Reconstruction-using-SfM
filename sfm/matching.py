"""SIFT feature extraction and geometrically verified pairwise matching."""

import cv2
import numpy as np

from .geometry import ransac_fundamental_matrix
from .rng import set_random_seed


def extract_features_and_matches(
    input_images,
    nfeatures=8000,
    window_size=8,
    ratio_thresh=0.8,
    min_matches=15,
    min_inliers=15,
    seed=0,
    ransac_iterations=2000,
    contrast_threshold=0.02,
    edge_threshold=10,
):
    """
    Extract SIFT features and match image pairs.

    Parameters
    ----------
    nfeatures : int
        Max SIFT features per image. 0 means unlimited.
    contrast_threshold : float
        Lower than OpenCV default (0.04) to detect more features on low-texture
        surfaces (Buddha). COLMAP-like denser keypoints.
    window_size : int or None
        If None, match all pairs (exhaustive — can corrupt tracks).
        Otherwise only pairs with index difference <= window_size.
    """
    if seed is not None:
        set_random_seed(seed)

    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
    )
    bf = cv2.BFMatcher()
    keypoints_list = []
    descriptors_list = []
    matches_list = {}

    for img in input_images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    num_images = len(input_images)
    if window_size is None:
        window_size = num_images

    for i in range(num_images):
        for j in range(i + 1, min(i + 1 + window_size, num_images)):
            if descriptors_list[i] is None or descriptors_list[j] is None:
                continue

            knn_matches = bf.knnMatch(descriptors_list[i], descriptors_list[j], k=2)
            # knnMatch can return lists with a single match; skip those
            good_matches = []
            for pair in knn_matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)

            if len(good_matches) >= min_matches:
                pts1 = np.array(
                    [keypoints_list[i][m.queryIdx].pt for m in good_matches]
                )
                pts2 = np.array(
                    [keypoints_list[j][m.trainIdx].pt for m in good_matches]
                )

                F, inliers = ransac_fundamental_matrix(
                    pts1, pts2, num_iterations=ransac_iterations
                )
                if F is not None and len(inliers) > min_inliers:
                    matches_list[(i, j)] = [good_matches[idx] for idx in inliers]
                    print(f"Verified pair ({i}, {j}) with {len(inliers)} inliers.")

    print(f"{len(matches_list)} matches found.")
    return keypoints_list, descriptors_list, matches_list
