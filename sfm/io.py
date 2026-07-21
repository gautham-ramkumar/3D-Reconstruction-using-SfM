"""Data loading helpers for images and COLMAP intrinsics."""

import os

import cv2
import numpy as np


def load_data(image_dir):
    """Load PNG images from a directory in sorted filename order."""
    images = []
    file_list = sorted(os.listdir(image_dir))
    for file_name in file_list:
        if file_name.lower().endswith(".png"):
            img_path = os.path.join(image_dir, file_name)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
            else:
                print(f"Warning: Failed to read {img_path}")
    return images


def load_all_intrinsics(file_path, verbose=True):
    """
    Parse COLMAP cameras.txt (SIMPLE_RADIAL model).

    Format: CAMERA_ID, MODEL, WIDTH, HEIGHT, f, cx, cy, k1
    """
    intrinsics_dict = {}

    with open(file_path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue

            params = line.split()
            cam_id = int(params[0])
            f_val = float(params[4])
            cx = float(params[5])
            cy = float(params[6])
            k1 = float(params[7])

            K = np.array(
                [[f_val, 0, cx], [0, f_val, cy], [0, 0, 1.0]], dtype=float
            )

            if verbose:
                print(f"Camera ID: {cam_id}")
                print(f"Focal Length: {f_val}")
                print(f"Principal Point: ({cx}, {cy})")
                print(f"Radial Distortion (k1): {k1}")
                print(f"Intrinsic Matrix K:\n{K}\n")

            intrinsics_dict[cam_id] = {"K": K, "k1": k1}

    return intrinsics_dict
