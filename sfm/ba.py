"""GTSAM bundle adjustment."""

from collections import defaultdict

import gtsam
import numpy as np
from gtsam.symbol_shorthand import L, X


def build_observations(img_kp_to_track, keypoints):
    """Build track_id -> [(cam_idx, (u, v)), ...] from keypoint lookups."""
    obs_by_track = defaultdict(list)

    for i, kp_map in enumerate(img_kp_to_track):
        if i >= len(keypoints) or keypoints[i] is None:
            continue
        for kp_idx, tid in kp_map.items():
            u, v = keypoints[i][kp_idx].pt
            obs_by_track[tid].append((i, (float(u), float(v))))

    return obs_by_track


def run_bundle_adjustment(
    map_3d,
    camera_poses,
    img_kp_to_track,
    keypoints,
    all_intrinsics,
    anchor_pose_idx=10,
    min_depth=0.1,
    max_iterations=40,
    add_pose_jitter=False,
):
    """
    Run GTSAM Levenberg-Marquardt BA on the incremental reconstruction.

    Returns
    -------
    optimized_poses : dict
    optimized_map_3d : dict
    stats : dict
    """
    obs_by_track = build_observations(img_kp_to_track, keypoints)

    valid_obs_by_track = defaultdict(list)
    filtered_obs_count = 0

    for tid, obs in obs_by_track.items():
        if tid not in map_3d:
            continue
        pt3d = np.array(map_3d[tid])

        for cam_idx, uv in obs:
            if cam_idx not in camera_poses:
                continue
            R_w2c, t_w2c = camera_poses[cam_idx]
            depth = (R_w2c @ pt3d + t_w2c)[2]

            if depth > min_depth:
                valid_obs_by_track[tid].append((cam_idx, uv))
            else:
                filtered_obs_count += 1

    valid_tracks = {
        tid for tid, obs in valid_obs_by_track.items() if len(obs) >= 2
    }
    print(f"Filtered {filtered_obs_count} bad observations.")
    print(f"Using {len(valid_tracks)}/{len(map_3d)} tracks for BA.")

    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()

    measurement_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
        gtsam.noiseModel.Isotropic.Sigma(2, 1.0),
    )

    # Cache calibrations once per camera (not once per factor)
    gtsam_K_by_cam = {}
    for cam_idx in camera_poses:
        K_vals = all_intrinsics[cam_idx + 1]["K"]
        gtsam_K_by_cam[cam_idx] = gtsam.Cal3_S2(
            float(K_vals[0, 0]),
            float(K_vals[1, 1]),
            float(K_vals[0, 1]),
            float(K_vals[0, 2]),
            float(K_vals[1, 2]),
        )

    for i, (R_w2c, t_w2c) in camera_poses.items():
        R_c2w = R_w2c.T
        t_c2w = -R_w2c.T @ t_w2c.reshape(3, 1)
        Twc = gtsam.Pose3(gtsam.Rot3(R_c2w), gtsam.Point3(t_c2w.flatten()))

        if add_pose_jitter:
            initial_values.insert(X(i), Twc.retract(1e-4 * np.random.randn(6, 1)))
        else:
            initial_values.insert(X(i), Twc)

        if i == anchor_pose_idx:
            pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([1e-4] * 3 + [1e-6] * 3)
            )
            graph.add(gtsam.PriorFactorPose3(X(i), Twc, pose_prior_noise))

    # Remap track IDs -> compact landmark indices (faster Keys, less sparse overhead)
    tid_list = sorted(valid_tracks)
    tid_to_lid = {tid: k for k, tid in enumerate(tid_list)}

    num_projections = 0
    landmarks_added = set()

    for tid in tid_list:
        lid = tid_to_lid[tid]
        pt3d = np.array(map_3d[tid], dtype=float)
        initial_values.insert(L(lid), gtsam.Point3(pt3d[0], pt3d[1], pt3d[2]))
        landmarks_added.add(tid)

        for cam_idx, (u, v) in valid_obs_by_track[tid]:
            # throwCheirality=False, verboseCheirality=False
            # (verbose=True prints every cheirality hit and can take many minutes)
            graph.add(
                gtsam.GenericProjectionFactorCal3_S2(
                    gtsam.Point2(u, v),
                    measurement_noise,
                    X(cam_idx),
                    L(lid),
                    gtsam_K_by_cam[cam_idx],
                    False,
                    False,
                )
            )
            num_projections += 1

    print(
        f"Added {len(landmarks_added)} landmarks with {num_projections} projections."
    )
    print("Running Levenberg-Marquardt (this should take seconds, not minutes)...")

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(max_iterations)
    params.setRelativeErrorTol(1e-4)
    params.setAbsoluteErrorTol(1e-4)
    # Quiet progress that doesn't spam cheirality lines
    try:
        params.setVerbosityLM("SUMMARY")
    except Exception:
        pass

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)
    result = optimizer.optimize()

    initial_error = graph.error(initial_values)
    final_error = graph.error(result)
    final_rmse = np.sqrt(final_error / max(2 * num_projections, 1))

    print("\nOptimization Complete.")
    print(f"Initial error: {initial_error:.2f}")
    print(f"Final error: {final_error:.2f}")
    if initial_error > 0:
        print(f"Error reduction: {100 * (1 - final_error / initial_error):.1f}%")
    print(f"Final RMSE: {final_rmse:.4f} px")

    optimized_poses = {}
    for i in camera_poses.keys():
        if result.exists(X(i)):
            p = result.atPose3(X(i))
            R_opt = p.rotation().matrix().T
            t_opt = -p.rotation().matrix().T @ p.translation().reshape(3, 1)
            optimized_poses[i] = (R_opt, t_opt.flatten())

    optimized_map_3d = {}
    for tid, lid in tid_to_lid.items():
        if result.exists(L(lid)):
            optimized_map_3d[tid] = np.array(result.atPoint3(L(lid)))

    stats = {
        "initial_error": initial_error,
        "final_error": final_error,
        "final_rmse": final_rmse,
        "num_projections": num_projections,
        "num_landmarks": len(landmarks_added),
    }
    return optimized_poses, optimized_map_3d, stats
