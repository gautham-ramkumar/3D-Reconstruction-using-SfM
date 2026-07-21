"""Open3D visualization of sparse reconstructions."""

import numpy as np
import open3d as o3d


def plot_3d_map(
    map_3d_dict,
    camera_poses_dict,
    all_intrinsics,
    images,
    title="Buddha Reconstruction",
    point_size=2,
    img_kp_to_track=None,
    keypoints=None,
    show=True,
):
    """
    Plot the 3D point cloud and camera frustums with Open3D.

    Extra kwargs (title, point_size, img_kp_to_track, keypoints, show) are
    accepted for notebook compatibility; point colors are sampled from images
    when track/keypoint data is provided.
    """
    geometry_list = []
    points = (
        np.array(list(map_3d_dict.values()), dtype=float)
        if map_3d_dict
        else np.zeros((0, 3))
    )

    if points.size > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        colors = _sample_point_colors(
            map_3d_dict, camera_poses_dict, img_kp_to_track, keypoints, images
        )
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            pcd.paint_uniform_color([0.1, 0.1, 0.1])
        geometry_list.append(pcd)
    else:
        print("No 3D points in the map to display.")

    scale = 1.0
    if points.size > 0:
        scale = float(np.max(np.linalg.norm(points, axis=1)) / 300.0)

    all_frustum_points = []
    all_lines = []
    all_colors = []

    h, w = images[0].shape[:2]
    corners_2d = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=float).T

    for i, (img_idx, (R, t)) in enumerate(camera_poses_dict.items()):
        K = all_intrinsics[img_idx + 1]["K"]
        K_inv = np.linalg.inv(K)
        corners_3d = K_inv @ corners_2d

        T_world = np.eye(4)
        T_world[:3, :3] = R.T
        T_world[:3, 3] = (-R.T @ t.reshape(3, 1)).flatten()
        C_world = T_world[:3, 3]

        p1 = (T_world @ np.append(corners_3d[:, 0] * scale, 1))[:3]
        p2 = (T_world @ np.append(corners_3d[:, 1] * scale, 1))[:3]
        p3 = (T_world @ np.append(corners_3d[:, 2] * scale, 1))[:3]
        p4 = (T_world @ np.append(corners_3d[:, 3] * scale, 1))[:3]

        base = len(all_frustum_points)
        all_frustum_points.extend([C_world, p1, p2, p3, p4])
        lines = [
            [base, base + 1],
            [base, base + 2],
            [base, base + 3],
            [base, base + 4],
            [base + 1, base + 2],
            [base + 2, base + 3],
            [base + 3, base + 4],
            [base + 4, base + 1],
        ]
        all_lines.extend(lines)
        color = [0, 1, 0] if i == 0 else [1, 0, 0]
        all_colors.extend([color] * len(lines))

    if all_frustum_points:
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.array(all_frustum_points)),
            lines=o3d.utility.Vector2iVector(np.array(all_lines)),
        )
        line_set.colors = o3d.utility.Vector3dVector(np.array(all_colors))
        geometry_list.append(line_set)

    if show:
        o3d.visualization.draw_geometries(geometry_list, window_name=title)
    return geometry_list


def _sample_point_colors(
    map_3d_dict, camera_poses_dict, img_kp_to_track, keypoints, images
):
    """RGB colors in [0,1] from the first observing image, or None."""
    if not map_3d_dict or img_kp_to_track is None or keypoints is None:
        return None

    first_obs = {}
    for img_idx, kp_map in enumerate(img_kp_to_track):
        if img_idx not in camera_poses_dict:
            continue
        for kp_idx, tid in kp_map.items():
            if tid in map_3d_dict and tid not in first_obs:
                u, v = keypoints[img_idx][kp_idx].pt
                first_obs[tid] = (img_idx, u, v)

    colors = []
    for tid in map_3d_dict.keys():
        if tid not in first_obs:
            colors.append([0.15, 0.15, 0.15])
            continue
        img_idx, u, v = first_obs[tid]
        h, w = images[img_idx].shape[:2]
        x = int(np.clip(round(u), 0, w - 1))
        y = int(np.clip(round(v), 0, h - 1))
        b, g, r = images[img_idx][y, x]
        colors.append([r / 255.0, g / 255.0, b / 255.0])
    return np.array(colors, dtype=float)
