# Incremental Structure-from-Motion

A from-scratch **incremental sparse SfM** pipeline that reconstructs camera poses and a 3D landmark cloud from a monocular image sequence. The geometry stack (fundamental / essential matrices, triangulation, PnP, tracks) is implemented in Python; bundle adjustment uses **GTSAM**.

## Pipeline

```text
Images + intrinsics
  → SIFT features & pairwise matching (RANSAC F)
  → Global tracks (Union-Find)
  → Seed pair (E-matrix, cheirality, triangulation)
  → Bidirectional PnP registration + map growth
  → Retriangulation of leftover tracks
  → GTSAM bundle adjustment
  → Open3D visualization / COLMAP export
```

**Core stages**

1. **Feature matching** — SIFT + Lowe ratio test + RANSAC fundamental matrix (Sampson error). Optional SuperPoint + LightGlue front-end.
2. **Two-view geometry** — normalized 8-point `F`, essential matrix `E = Kⱼᵀ F Kᵢ`, pose hypotheses, cheirality selection.
3. **Incremental mapping** — seed triangulation, `solvePnPRansac` localization, neighbor triangulation, global retriangulation.
4. **Bundle adjustment** — robust (Huber) projection factors in GTSAM; joint pose + landmark LM optimization.
5. **Visualization / export** — Open3D viewer; COLMAP-compatible sparse models for the COLMAP GUI.

## Repository layout

```text
.
├── SFM.ipynb              # End-to-end runner notebook
├── sfm/                   # Pipeline package
│   ├── io.py              # Image + COLMAP cameras.txt loading
│   ├── geometry.py        # F / E / pose / triangulation
│   ├── matching.py        # SIFT matching
│   ├── matching_superpoint.py  # Optional SuperPoint + LightGlue
│   ├── tracks.py          # Multi-view tracks
│   ├── mapping.py         # Seed, PnP expansion, retriangulation
│   ├── ba.py              # GTSAM bundle adjustment
│   ├── viz.py             # Open3D visualization
│   ├── colmap_export.py   # Export for COLMAP GUI
│   └── rng.py             # Reproducible seeding
├── buddha_images/         # Input RGB frames
├── cameras.txt            # Per-view intrinsics (SIMPLE_RADIAL)
├── colmap_exports/        # Exported before/after-BA models
└── requirements.txt
```

## Requirements

- Python 3.10+
- OpenCV, NumPy, Open3D, GTSAM, Jupyter
- Optional (learned features): PyTorch (CUDA recommended), [LightGlue](https://github.com/cvg/LightGlue)

```bash
pip install -r requirements.txt
# Optional SuperPoint / LightGlue:
pip install git+https://github.com/cvg/LightGlue.git
```

COLMAP is only needed if you want to inspect exported models in the COLMAP GUI (`colmap` on `PATH`).

## Quick start

1. Place images under `buddha_images/` and provide matching `cameras.txt` (COLMAP text format, `SIMPLE_RADIAL`).
2. Open and run [`SFM.ipynb`](SFM.ipynb) from the project root (kernel working directory = repo root).

```python
from sfm import (
    load_data,
    load_all_intrinsics,
    extract_features_and_matches,
    build_global_tracks,
    initialize_seed_map,
    run_pnp_mapping,
    retriangulate_tracks,
    run_bundle_adjustment,
    plot_3d_map,
    export_colmap_model,
    set_random_seed,
)

set_random_seed(0)
images = load_data("buddha_images/")
intrinsics = load_all_intrinsics("cameras.txt", verbose=False)
# … matching → tracks → seed → PnP → BA (see notebook)
```

### Feature front-end

**Default (SIFT)** — typically cleaner structure on this dataset:

```python
keypoints, descriptors, matches = extract_features_and_matches(
    images,
    nfeatures=8000,
    window_size=8,
    contrast_threshold=0.02,
)
```

**Optional (SuperPoint + LightGlue)** — denser matches; structure quality can vary:

```python
from sfm import extract_features_and_matches_superpoint
keypoints, descriptors, matches = extract_features_and_matches_superpoint(
    images, max_num_keypoints=4096, window_size=8
)
```

### COLMAP GUI

After a run, models are written to:

- `colmap_exports/before_ba`
- `colmap_exports/after_ba`

```bash
colmap gui \
  --database_path buddha_images/database.db \
  --image_path buddha_images \
  --import_path colmap_exports/after_ba
```

Exports regenerate `.bin` files from text so the GUI does not load stale binaries.

## Example results

On a 24-view sequence with denser SIFT settings (illustrative; exact counts depend on parameters and seed):

| Stage | Typical scale |
|--------|----------------|
| Images registered | 24 / 24 |
| Sparse landmarks (pre-BA) | ~5.5k |
| BA cost reduction | ~60–70% |
| Final reprojection RMSE | ~3–4 px |

Pre/post-BA Open3D screenshots in the repo (`Pre_BA_*.png`, `Post_BA_*.png`) show the recovered geometry and camera frustums.

## Design notes

- Intrinsics are treated as known (from `cameras.txt`); radial `k1` is stored but the core solver uses a pinhole `Cal3_S2` model in BA.
- Matching uses a sliding image window to limit false long-range associations that can corrupt tracks.
- `set_random_seed(0)` and OpenCV single-threading improve run-to-run reproducibility for RANSAC stages.
- Bundle adjustment disables verbose cheirality logging for speed on larger landmark sets.

## License

Add a license of your choice if you distribute this repository.
