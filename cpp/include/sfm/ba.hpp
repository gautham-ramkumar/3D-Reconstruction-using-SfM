// Ceres bundle adjustment over camera poses and 3D points (intrinsics fixed, pinhole).
#pragma once

#include <vector>

#include "sfm/mapping.hpp"
#include "sfm/types.hpp"

namespace sfm {

struct BAParams {
    int anchor_image = -1;          // held constant to fix the world frame; -1 = first registered
    // Fix the overall scale by holding the largest translation component of a second camera
    // constant. Without it the problem has a free scale and Ceres's linear solves are singular.
    bool fix_scale = true;
    int scale_image = -1;           // -1 = second registered image (the other seed camera)
    double min_depth = 0.1;         // observations closer than this are dropped (as in Python)
    double max_initial_error = 10.0;  // px; gross outliers removed before round 1
    double max_final_error = 4.0;   // px; removed between rounds, measured on the new solution
    int rounds = 2;
    double huber_delta = 1.345;     // px; same cost as GTSAM's Huber(1.345) with sigma = 1
    int max_iterations = 100;
    double function_tolerance = 1e-6;  // relative cost change to stop at
    bool verbose = true;
};

struct BARoundStats {
    int num_observations = 0;
    int num_tracks = 0;
    int iterations = 0;
    double initial_cost = 0.0;   // 0.5 * sum of Huber losses; equals GTSAM's graph.error()
    double final_cost = 0.0;
    double python_rmse = 0.0;    // sqrt(final_cost / (2 * N)), the Python notebook's "Final RMSE"
    double plain_rms = 0.0;      // sqrt(mean squared 2D error) over this round's observations, px
    int num_behind = 0;          // observations behind their camera after solving (not in plain_rms)
    double seconds = 0.0;
};

struct BAResult {
    Reconstruction rec;          // optimized poses; only tracks kept by BA remain in the map
    std::vector<BARoundStats> rounds;
    int dropped_depth = 0;       // observations removed for depth <= min_depth
    int dropped_initial = 0;     // removed for error > max_initial_error
    int dropped_between = 0;     // removed between rounds for error > max_final_error
};

BAResult run_bundle_adjustment(const Reconstruction& rec, const KpToTrack& kp_to_track,
                               const Keypoints& keypoints, const IntrinsicsMap& intrinsics,
                               const BAParams& params = {});

}  // namespace sfm
