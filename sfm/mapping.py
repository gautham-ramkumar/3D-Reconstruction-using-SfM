"""Incremental SfM: seed selection, initialization, and PnP map expansion."""

from collections import defaultdict

import cv2
import numpy as np

from .geometry import (
    calculate_median_parallax,
    cheirality_check,
    compute_essential_matrix,
    estimate_camera_pose,
    ransac_fundamental_matrix,
    triangulate_points,
)


def find_best_seed(matches, all_intrinsics, keypoints, top_k=15, min_parallax_deg=2.0):
    """Find the best starting pair based on inlier count and median parallax."""
    best_pair = None
    max_inliers = 0
    best_pose_data = None

    sorted_candidates = sorted(
        matches.keys(), key=lambda k: len(matches[k]), reverse=True
    )

    for i, j in sorted_candidates[:top_k]:
        inlier_matches = matches[(i, j)]

        K_i = all_intrinsics[i + 1]["K"]
        K_j = all_intrinsics[j + 1]["K"]

        pts1 = np.array([keypoints[i][m.queryIdx].pt for m in inlier_matches])
        pts2 = np.array([keypoints[j][m.trainIdx].pt for m in inlier_matches])

        F, _ = ransac_fundamental_matrix(pts1, pts2)
        if F is None:
            continue

        E = compute_essential_matrix(F, K_i, K_j)
        poses = estimate_camera_pose(E)
        R_rel, t_rel = cheirality_check(poses, K_i, K_j, pts1, pts2)

        if R_rel is None:
            continue

        parallax = calculate_median_parallax(pts1, pts2, K_i, K_j, R_rel, t_rel)

        if parallax > min_parallax_deg:
            print(
                f"Candidate ({i}, {j}): {len(inlier_matches)} inliers, "
                f"{parallax:.2f}° parallax - VALID\n"
            )
            if len(inlier_matches) > max_inliers:
                max_inliers = len(inlier_matches)
                best_pair = (i, j)
                best_pose_data = (R_rel, t_rel)
        else:
            print(
                f"Candidate ({i}, {j}): {parallax:.2f}° parallax - "
                "TOO LOW (Narrow Baseline)\n"
            )

    return best_pair, best_pose_data


def initialize_seed_map(
    seed_i,
    seed_j,
    matches,
    keypoints,
    img_kp_to_track,
    all_intrinsics,
    min_depth=0.1,
):
    """
    Triangulate the seed pair and initialize the global map / poses.

    Uses dual-camera depth checks (both views must see the point in front).
    """
    map_3d = {}
    camera_poses = {}

    matches_candidates = matches[(seed_i, seed_j)]
    pts1 = np.array([keypoints[seed_i][m.queryIdx].pt for m in matches_candidates])
    pts2 = np.array([keypoints[seed_j][m.trainIdx].pt for m in matches_candidates])

    F_seed, inlier_mask = ransac_fundamental_matrix(pts1, pts2)
    pts1_inliers = pts1[inlier_mask]
    pts2_inliers = pts2[inlier_mask]

    Ki = all_intrinsics[seed_i + 1]["K"]
    Kj = all_intrinsics[seed_j + 1]["K"]

    E = compute_essential_matrix(F_seed, Ki, Kj)
    poses = estimate_camera_pose(E)
    R_seed, t_seed = cheirality_check(poses, Ki, Kj, pts1_inliers, pts2_inliers)

    P1 = Ki @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = Kj @ np.hstack((R_seed, t_seed.reshape(3, 1)))
    points_3d_raw = triangulate_points(P1, P2, pts1_inliers, pts2_inliers)

    camera_poses[seed_i] = (np.eye(3), np.zeros(3))
    camera_poses[seed_j] = (R_seed, t_seed)

    for idx, pt3d in enumerate(points_3d_raw):
        depth_cam1 = pt3d[2]
        depth_cam2 = (R_seed @ pt3d + t_seed)[2]

        if depth_cam1 > min_depth and depth_cam2 > min_depth:
            match_obj = matches_candidates[inlier_mask[idx]]
            track_id = img_kp_to_track[seed_i][match_obj.queryIdx]
            map_3d[track_id] = pt3d

    print(f"Initialized Map with {len(map_3d)} 3D points.")
    print(f"Seed Pair ({seed_i}, {seed_j}) registered successfully.")
    return map_3d, camera_poses, R_seed, t_seed


def run_pnp_mapping(
    indices,
    map_3d,
    camera_poses,
    img_kp_to_track,
    matches,
    keypoints,
    all_intrinsics,
    min_correspondences=12,
    reprojection_error=8.0,
):
    """Register new views via PnP and expand the map by triangulating neighbors."""
    for i in indices:
        if i in camera_poses:
            continue

        pts_3d, pts_2d = [], []
        for kp_idx, track_id in img_kp_to_track[i].items():
            if track_id in map_3d:
                pts_3d.append(map_3d[track_id])
                pts_2d.append(keypoints[i][kp_idx].pt)

        if len(pts_3d) < min_correspondences:
            print(f"Skipping image {i}: Only {len(pts_3d)} correspondences.")
            continue

        K_curr = all_intrinsics[i + 1]["K"]
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.array(pts_3d, dtype=np.float32),
            np.array(pts_2d, dtype=np.float32),
            K_curr,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=reprojection_error,
        )

        if not success:
            continue

        R_curr, _ = cv2.Rodrigues(rvec)
        t_curr = tvec.flatten()
        camera_poses[i] = (R_curr, t_curr)

        P2 = K_curr @ np.hstack((R_curr, t_curr.reshape(3, 1)))

        new_points_count = 0
        # Triangulate against all already-registered neighbors (not just ±3)
        for neighbor_idx in list(camera_poses.keys()):
            if neighbor_idx == i:
                continue

            pair = tuple(sorted((neighbor_idx, i)))
            if pair not in matches:
                continue

            K_p = all_intrinsics[neighbor_idx + 1]["K"]
            R_p, t_p = camera_poses[neighbor_idx]
            P1 = K_p @ np.hstack((R_p, t_p.reshape(3, 1)))

            pts1_tri, pts2_tri, tids_tri = [], [], []
            for m in matches[pair]:
                idx_i = m.queryIdx if pair[0] == i else m.trainIdx
                idx_p = m.trainIdx if pair[0] == i else m.queryIdx

                tid = img_kp_to_track[i][idx_i]
                if tid not in map_3d:
                    pts1_tri.append(keypoints[neighbor_idx][idx_p].pt)
                    pts2_tri.append(keypoints[i][idx_i].pt)
                    tids_tri.append(tid)

            if len(pts1_tri) > 0:
                new_3d = triangulate_points(
                    P1, P2, np.array(pts1_tri), np.array(pts2_tri)
                )
                for j, pt3d in enumerate(new_3d):
                    if pt3d[2] > 0:
                        map_3d[tids_tri[j]] = pt3d
                        new_points_count += 1

        print(
            f"Registered {i}: {len(inliers)} inliers, added {new_points_count} points."
        )


def retriangulate_tracks(
    map_3d,
    camera_poses,
    tracks,
    keypoints,
    all_intrinsics,
    min_depth=0.1,
    max_reproj_error=4.0,
):
    """
    Triangulate tracks not yet in the map using any two registered observations.

    This is the main way to grow the point cloud after denser matching: many
    tracks exist but were never triangulated during the ±3 PnP expansion window.
    """
    added = 0
    rejected = 0

    for tid, observations in tracks.items():
        if tid in map_3d:
            continue

        registered_obs = [
            (img_id, kp_idx)
            for img_id, kp_idx in observations
            if img_id in camera_poses
        ]
        if len(registered_obs) < 2:
            continue

        # Prefer the pair with largest baseline (camera-center distance)
        best = None
        best_baseline = -1.0
        centers = {}
        for img_id, _ in registered_obs:
            R, t = camera_poses[img_id]
            centers[img_id] = -R.T @ t

        for a in range(len(registered_obs)):
            for b in range(a + 1, len(registered_obs)):
                i, kpi = registered_obs[a]
                j, kpj = registered_obs[b]
                baseline = np.linalg.norm(centers[i] - centers[j])
                if baseline > best_baseline:
                    best_baseline = baseline
                    best = (i, kpi, j, kpj)

        i, kpi, j, kpj = best
        Ri, ti = camera_poses[i]
        Rj, tj = camera_poses[j]
        Ki = all_intrinsics[i + 1]["K"]
        Kj = all_intrinsics[j + 1]["K"]
        Pi = Ki @ np.hstack((Ri, ti.reshape(3, 1)))
        Pj = Kj @ np.hstack((Rj, tj.reshape(3, 1)))

        pts_i = np.array([keypoints[i][kpi].pt], dtype=float)
        pts_j = np.array([keypoints[j][kpj].pt], dtype=float)
        pt3d = triangulate_points(Pi, Pj, pts_i, pts_j)[0]

        depth_i = (Ri @ pt3d + ti)[2]
        depth_j = (Rj @ pt3d + tj)[2]
        if depth_i <= min_depth or depth_j <= min_depth:
            rejected += 1
            continue

        # Reprojection check in both views
        ok = True
        for img_id, uv in ((i, pts_i[0]), (j, pts_j[0])):
            R, t = camera_poses[img_id]
            K = all_intrinsics[img_id + 1]["K"]
            Xc = R @ pt3d + t
            if Xc[2] <= 1e-8:
                ok = False
                break
            proj = K @ Xc
            proj = proj[:2] / proj[2]
            if np.linalg.norm(proj - uv) > max_reproj_error:
                ok = False
                break

        if not ok:
            rejected += 1
            continue

        map_3d[tid] = pt3d
        added += 1

    print(
        f"Retriangulation: added {added} points "
        f"(rejected {rejected}, map now {len(map_3d)} points)."
    )
    return added


def diagnose_map_quality(map_3d, camera_poses, img_kp_to_track, keypoints):
    """Report tracks that have negative depth in at least one observing camera."""
    obs_by_track = defaultdict(list)

    for i, kp_map in enumerate(img_kp_to_track):
        if i >= len(keypoints) or keypoints[i] is None:
            continue
        for kp_idx, tid in kp_map.items():
            u, v = keypoints[i][kp_idx].pt
            obs_by_track[tid].append((i, (float(u), float(v))))

    print("\n=== Diagnosing Map Quality ===")
    bad_tracks = []
    for tid, pt3d in map_3d.items():
        observing_cams = [
            cam_idx for cam_idx, _ in obs_by_track[tid] if cam_idx in camera_poses
        ]

        for cam_idx in observing_cams:
            R_w2c, t_w2c = camera_poses[cam_idx]
            depth = (R_w2c @ pt3d + t_w2c)[2]
            if depth <= 0:
                bad_tracks.append((tid, cam_idx, depth))
                break

    print(f"Total tracks in map: {len(map_3d)}")
    print(
        f"Tracks with negative depth in at least one observer: {len(bad_tracks)}"
    )

    if bad_tracks[:10]:
        print("\nFirst 10 bad tracks (tid, cam_idx, depth):")
        for t in bad_tracks[:10]:
            print(f"  Track {t[0]}: depth {t[2]:.3f} in camera {t[1]}")

    return bad_tracks, obs_by_track
