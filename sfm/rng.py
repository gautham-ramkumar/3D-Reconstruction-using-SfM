"""Reproducibility helpers."""

import random

import cv2
import numpy as np


def set_random_seed(seed: int = 0) -> None:
    """Seed Python, NumPy, and OpenCV RNGs for reproducible SfM runs."""
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    # SIFT / matchers can be nondeterministic with multi-threading on some builds
    cv2.setNumThreads(1)
    print(f"Random seed set to {seed} (NumPy, OpenCV; OpenCV threads=1)")
