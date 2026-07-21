"""Custom Structure-from-Motion pipeline for AFR HW4."""

from .ba import run_bundle_adjustment
from .colmap_export import export_colmap_model, open_in_colmap_gui
from .rng import set_random_seed
from .geometry import (
    calculate_median_parallax,
    cheirality_check,
    compute_essential_matrix,
    estimate_camera_pose,
    estimate_fundamental_matrix,
    normalize_points,
    ransac_fundamental_matrix,
    triangulate_points,
)
from .io import load_all_intrinsics, load_data
from .mapping import (
    diagnose_map_quality,
    find_best_seed,
    initialize_seed_map,
    retriangulate_tracks,
    run_pnp_mapping,
)
from .matching import extract_features_and_matches
from .matching_superpoint import extract_features_and_matches_superpoint
from .tracks import build_global_tracks
from .viz import plot_3d_map

__all__ = [
    "load_data",
    "load_all_intrinsics",
    "normalize_points",
    "estimate_fundamental_matrix",
    "ransac_fundamental_matrix",
    "compute_essential_matrix",
    "estimate_camera_pose",
    "triangulate_points",
    "cheirality_check",
    "calculate_median_parallax",
    "extract_features_and_matches",
    "extract_features_and_matches_superpoint",
    "build_global_tracks",
    "find_best_seed",
    "initialize_seed_map",
    "run_pnp_mapping",
    "retriangulate_tracks",
    "diagnose_map_quality",
    "run_bundle_adjustment",
    "plot_3d_map",
    "export_colmap_model",
    "open_in_colmap_gui",
    "set_random_seed",
]
