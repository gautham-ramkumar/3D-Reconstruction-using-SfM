"""Export reconstructions to COLMAP sparse model format for GUI viewing."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to COLMAP quaternion (qw, qx, qy, qz)."""
    R = np.asarray(R, dtype=float)
    trace = np.trace(R)

    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=float)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


def _image_names(image_dir):
    return sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith(".png")
    )


def _sample_color(images, img_idx, u, v):
    h, w = images[img_idx].shape[:2]
    x = int(np.clip(round(u), 0, w - 1))
    y = int(np.clip(round(v), 0, h - 1))
    bgr = images[img_idx][y, x]
    return int(bgr[2]), int(bgr[1]), int(bgr[0])


def _mean_reproj_error(pt3d, observations, camera_poses, all_intrinsics):
    errors = []
    for cam_idx, (u, v) in observations:
        if cam_idx not in camera_poses:
            continue
        R, t = camera_poses[cam_idx]
        K = all_intrinsics[cam_idx + 1]["K"]
        Xc = R @ pt3d + t
        if Xc[2] <= 1e-8:
            continue
        proj = K @ Xc
        proj = proj[:2] / proj[2]
        errors.append(np.linalg.norm(proj - np.array([u, v])))
    if not errors:
        return 0.0
    return float(np.mean(errors))


def export_colmap_model(
    output_dir,
    map_3d,
    camera_poses,
    img_kp_to_track,
    keypoints,
    all_intrinsics,
    images,
    image_dir="buddha_images",
    also_binary=True,
):
    """
    Write a COLMAP sparse model (cameras/images/points3D .txt, optional .bin).

    Returns the absolute path to the model directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = _image_names(image_dir)
    if len(names) != len(images):
        raise ValueError(
            f"Found {len(names)} PNGs in {image_dir} but {len(images)} loaded images"
        )

    point_ids = sorted(map_3d.keys())
    tid_to_pid = {tid: i + 1 for i, tid in enumerate(point_ids)}

    image_obs = {i: [] for i in camera_poses}
    tracks_for_points = defaultdict(list)

    for img_idx in sorted(camera_poses.keys()):
        kp_map = img_kp_to_track[img_idx]
        for kp_idx in sorted(kp_map.keys()):
            tid = kp_map[kp_idx]
            if tid not in tid_to_pid:
                continue
            u, v = keypoints[img_idx][kp_idx].pt
            pid = tid_to_pid[tid]
            point2d_idx = len(image_obs[img_idx])
            image_obs[img_idx].append((float(u), float(v), pid))
            tracks_for_points[pid].append((img_idx + 1, point2d_idx))

    cam_lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(camera_poses)}",
    ]
    used_camera_ids = sorted(i + 1 for i in camera_poses.keys())
    for cam_id in used_camera_ids:
        info = all_intrinsics[cam_id]
        K = info["K"]
        k1 = info.get("k1", 0.0)
        f_val = float(K[0, 0])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        img_idx = cam_id - 1
        height, width = images[img_idx].shape[:2]
        cam_lines.append(
            f"{cam_id} SIMPLE_RADIAL {width} {height} "
            f"{f_val} {cx} {cy} {k1}"
        )
    (output_dir / "cameras.txt").write_text("\n".join(cam_lines) + "\n")

    mean_obs = (
        np.mean([len(image_obs[i]) for i in camera_poses]) if camera_poses else 0.0
    )
    img_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(camera_poses)}, "
        f"mean observations per image: {mean_obs}",
    ]
    for img_idx in sorted(camera_poses.keys()):
        R, t = camera_poses[img_idx]
        qw, qx, qy, qz = rotation_matrix_to_quaternion(R)
        tx, ty, tz = np.asarray(t, dtype=float).reshape(3)
        image_id = img_idx + 1
        img_lines.append(
            f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} "
            f"{image_id} {names[img_idx]}"
        )
        pts2d = image_obs[img_idx]
        img_lines.append(
            " ".join(f"{u} {v} {pid}" for u, v, pid in pts2d) if pts2d else ""
        )
    (output_dir / "images.txt").write_text("\n".join(img_lines) + "\n")

    track_lengths = [len(tracks_for_points[tid_to_pid[tid]]) for tid in point_ids]
    mean_track = float(np.mean(track_lengths)) if track_lengths else 0.0
    pts_lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
        f"# Number of points: {len(point_ids)}, mean track length: {mean_track}",
    ]

    obs_by_tid = defaultdict(list)
    for img_idx, kp_map in enumerate(img_kp_to_track):
        if img_idx not in camera_poses:
            continue
        for kp_idx, tid in kp_map.items():
            if tid not in tid_to_pid:
                continue
            u, v = keypoints[img_idx][kp_idx].pt
            obs_by_tid[tid].append((img_idx, (float(u), float(v))))

    for tid in point_ids:
        pid = tid_to_pid[tid]
        pt = np.asarray(map_3d[tid], dtype=float).reshape(3)
        obs = obs_by_tid[tid]
        err = _mean_reproj_error(pt, obs, camera_poses, all_intrinsics)
        if obs:
            cam_idx, (u, v) = obs[0]
            r, g, b = _sample_color(images, cam_idx, u, v)
        else:
            r, g, b = 0, 0, 0
        track = tracks_for_points[pid]
        track_str = " ".join(f"{iid} {p2d}" for iid, p2d in track)
        pts_lines.append(
            f"{pid} {pt[0]} {pt[1]} {pt[2]} {r} {g} {b} {err} {track_str}"
        )
    (output_dir / "points3D.txt").write_text("\n".join(pts_lines) + "\n")

    print(
        f"Wrote COLMAP text model to {output_dir.resolve()} "
        f"({len(camera_poses)} images, {len(point_ids)} points)"
    )

    if also_binary:
        _convert_to_binary(output_dir)

    return output_dir.resolve()


def _convert_to_binary(model_dir):
    """Convert text model to .bin using COLMAP CLI if available."""
    colmap_bin = shutil.which("colmap")
    if colmap_bin is None:
        print("colmap not found on PATH; skipped binary conversion.")
        return

    model_dir = Path(model_dir)
    # COLMAP prefers .bin over .txt — delete stale binaries so converter reads text.
    for name in (
        "cameras.bin",
        "images.bin",
        "points3D.bin",
        "frames.bin",
        "rigs.bin",
    ):
        bin_path = model_dir / name
        if bin_path.exists():
            bin_path.unlink()

    try:
        subprocess.run(
            [
                colmap_bin,
                "model_converter",
                "--input_path",
                str(model_dir),
                "--output_path",
                str(model_dir),
                "--output_type",
                "BIN",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Also wrote binary model (.bin) in {model_dir.resolve()}")
    except subprocess.CalledProcessError as e:
        print(f"Binary conversion failed: {e.stderr or e}")


def open_in_colmap_gui(
    model_dir,
    image_dir="buddha_images",
    database_path="buddha_images/database.db",
):
    """Launch COLMAP GUI importing the given sparse model."""
    colmap_bin = shutil.which("colmap")
    if colmap_bin is None:
        print("colmap not found on PATH.")
        return None

    model_dir = Path(model_dir).resolve()
    image_dir = Path(image_dir).resolve()
    cmd = [
        colmap_bin,
        "gui",
        "--database_path",
        str(Path(database_path).resolve()),
        "--image_path",
        str(image_dir),
        "--import_path",
        str(model_dir),
    ]
    print("Launching:", " ".join(cmd))
    return subprocess.Popen(cmd)
