"""SuperPoint + LightGlue feature matching (CUDA), with F-RANSAC verification."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .geometry import ransac_fundamental_matrix
from .rng import set_random_seed


def _to_cv_keypoints(xy: np.ndarray):
    """Convert Nx2 float keypoints to a list of cv2.KeyPoint."""
    return [cv2.KeyPoint(float(x), float(y), 1) for x, y in xy]


def extract_features_and_matches_superpoint(
    input_images,
    max_num_keypoints=4096,
    window_size=8,
    min_matches=15,
    min_inliers=15,
    match_threshold=0.1,
    seed=0,
    ransac_iterations=2000,
    device=None,
):
    """
    Detect SuperPoint features and match with LightGlue, then verify with F-RANSAC.

    Returns the same structure as SIFT matching:
    keypoints_list, descriptors_list, matches_list
    where matches_list[(i, j)] is a list of cv2.DMatch inliers.
    """
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import numpy_image_to_torch

    if seed is not None:
        set_random_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"SuperPoint/LightGlue device: {device}")

    extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
    matcher = (
        LightGlue(
            features="superpoint",
            depth_confidence=-1,
            width_confidence=-1,
            filter_threshold=match_threshold,
        )
        .eval()
        .to(device)
    )

    keypoints_list = []
    descriptors_list = []
    torch_feats = []

    with torch.no_grad():
        for idx, img in enumerate(input_images):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t_img = numpy_image_to_torch(rgb).to(device)
            feats = extractor.extract(t_img)
            torch_feats.append(feats)

            kps = feats["keypoints"][0].detach().cpu().numpy()
            desc = feats["descriptors"][0].detach().cpu().numpy()
            keypoints_list.append(_to_cv_keypoints(kps))
            descriptors_list.append(desc.astype(np.float32))
            print(f"  Image {idx}: {len(kps)} SuperPoint features")

    num_images = len(input_images)
    if window_size is None:
        window_size = num_images

    matches_list = {}

    with torch.no_grad():
        for i in range(num_images):
            for j in range(i + 1, min(i + 1 + window_size, num_images)):
                out = matcher({"image0": torch_feats[i], "image1": torch_feats[j]})
                # Keep batch dim for indexing, then strip
                matches = out["matches"][0].detach().cpu().numpy()  # [K, 2]
                if matches.ndim != 2 or matches.shape[0] < min_matches:
                    continue

                idx0 = matches[:, 0].astype(int)
                idx1 = matches[:, 1].astype(int)

                pts1 = np.array(
                    [keypoints_list[i][a].pt for a in idx0], dtype=float
                )
                pts2 = np.array(
                    [keypoints_list[j][b].pt for b in idx1], dtype=float
                )

                F, inliers = ransac_fundamental_matrix(
                    pts1, pts2, num_iterations=ransac_iterations
                )
                if F is None or len(inliers) <= min_inliers:
                    continue

                good = []
                for k in inliers:
                    m = cv2.DMatch(int(idx0[k]), int(idx1[k]), 0.0)
                    good.append(m)

                matches_list[(i, j)] = good
                print(f"Verified pair ({i}, {j}) with {len(good)} inliers.")

    print(f"{len(matches_list)} matches found.")
    return keypoints_list, descriptors_list, matches_list
