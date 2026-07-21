"""Two-view geometry: F/E estimation, pose recovery, triangulation."""

import cv2
import numpy as np


def normalize_points(pts):
    if pts is None or len(pts) == 0:
        return None, None

    centroid = np.mean(pts, axis=0)
    pts_centered = pts - centroid
    avg_dist = np.mean(np.sqrt(np.sum(pts_centered**2, axis=1)))

    if avg_dist < 1e-6:
        return None, None

    scale = np.sqrt(2) / avg_dist
    T = np.array(
        [
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1],
        ]
    )

    pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
    pts_normalized_h = (T @ pts_h.T).T

    return pts_normalized_h[:, :2], T


def estimate_fundamental_matrix(pts1, pts2):
    """Normalized 8-point algorithm for the fundamental matrix."""
    pts1_normalized, T1 = normalize_points(pts1)
    if pts1_normalized is None:
        return None, None, None
    pts2_normalized, T2 = normalize_points(pts2)
    if pts2_normalized is None:
        return None, None, None

    A = np.zeros((pts1_normalized.shape[0], 9))
    for i in range(pts1_normalized.shape[0]):
        x1, y1 = pts1_normalized[i]
        x2, y2 = pts2_normalized[i]
        A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]

    _, _, Vt = np.linalg.svd(A)
    F_normalized = Vt[-1].reshape(3, 3)

    U, S, Vt_f = np.linalg.svd(F_normalized)
    S[2] = 0
    F_normalized_rank2 = U @ np.diag(S) @ Vt_f
    F = T2.T @ F_normalized_rank2 @ T1
    return F / F[2, 2], T1, T2


def ransac_fundamental_matrix(pts1, pts2, num_iterations=5000, threshold=1.0):
    """Estimate F with RANSAC using Sampson distance."""
    max_inliers = []
    best_F = None

    for _ in range(num_iterations):
        idx = np.random.choice(len(pts1), 8, replace=False)
        F_candidate, _, _ = estimate_fundamental_matrix(pts1[idx], pts2[idx])
        if F_candidate is None:
            continue

        pts1_h = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
        pts2_h = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

        Fx1 = F_candidate @ pts1_h.T
        Ftx2 = F_candidate.T @ pts2_h.T

        denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Ftx2[0] ** 2 + Ftx2[1] ** 2
        err = (np.sum(pts2_h * (F_candidate @ pts1_h.T).T, axis=1)) ** 2 / denom

        inliers = np.where(err < threshold)[0]

        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            best_F = F_candidate

    return best_F, max_inliers


def compute_essential_matrix(F, K_i, K_j):
    """E = K_j^T F K_i with SVD rank cleanup."""
    E = K_j.T @ F @ K_i
    U, _, Vt = np.linalg.svd(E)
    E_corrected = U @ np.diag([1, 1, 0]) @ Vt
    return E_corrected


def estimate_camera_pose(E):
    """Decompose essential matrix into four (R, t) hypotheses."""
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]
    return (R1, t), (R1, -t), (R2, t), (R2, -t)


def triangulate_points(P1, P2, pts1, pts2):
    """Triangulate Nx2 correspondences into Nx3 Euclidean points."""
    pts1_input = pts1.T.astype(np.float32)
    pts2_input = pts2.T.astype(np.float32)
    points_4d = cv2.triangulatePoints(P1, P2, pts1_input, pts2_input)
    points_3d = (points_4d[:3] / points_4d[3]).T
    return points_3d


def cheirality_check(poses, K_i, K_j, pts1, pts2):
    """Pick the (R, t) hypothesis with the most points in front of both cameras."""
    P1 = K_i @ np.hstack((np.eye(3), np.zeros((3, 1))))

    max_positive_depth = -1
    best_pose = None

    for R, t in poses:
        P2 = K_j @ np.hstack((R, t.reshape(3, 1)))
        points_3d = triangulate_points(P1, P2, pts1, pts2)

        depth1 = points_3d[:, 2]
        points_cam2 = R @ points_3d.T + t.reshape(3, 1)
        depth2 = points_cam2[2, :]

        positive_depth_count = np.sum((depth1 > 0) & (depth2 > 0))

        if positive_depth_count > max_positive_depth:
            max_positive_depth = positive_depth_count
            best_pose = (R, t)

    print(
        f"Cheirality Check: Found valid pose with {max_positive_depth} "
        "points in front of cameras."
    )
    return best_pose


def calculate_median_parallax(pts1, pts2, K_i, K_j, R, t):
    """Median angle (degrees) between rays from two cameras to triangulated points."""
    P1 = K_i @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K_j @ np.hstack((R, t.reshape(3, 1)))

    pts3d = triangulate_points(P1, P2, pts1, pts2)

    C1 = np.zeros(3)
    C2 = -R.T @ t

    angles = []
    for pt in pts3d:
        v1 = pt - C1
        v2 = pt - C2
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        angles.append(angle)

    return np.median(angles)
