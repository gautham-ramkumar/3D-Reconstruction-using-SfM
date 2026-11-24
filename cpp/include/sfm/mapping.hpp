// Incremental SfM: seed selection, initialization, PnP registration and map growth.
#pragma once

#include <optional>
#include <tuple>
#include <vector>

#include "sfm/geometry.hpp"
#include "sfm/types.hpp"

namespace sfm {

struct Reconstruction {
    Map3D points;                        // track_id -> world point
    CameraPoses poses;                   // image -> world-to-camera pose
    std::vector<int> registration_order; // images in the order they got a pose

    void add_pose(int image, const Pose& pose);
};

struct SeedCandidate {
    ImagePair pair;
    RelativePose pose;  // pose of the second image relative to the first
};

// Among the top_k pairs by match count, pick the one with the most matches whose median
// parallax exceeds min_parallax_deg. nullopt if no candidate qualifies.
std::optional<SeedCandidate> find_best_seed(const PairMatches& matches,
                                            const IntrinsicsMap& intrinsics,
                                            const Keypoints& keypoints, int top_k = 15,
                                            double min_parallax_deg = 2.0,
                                            int ransac_iterations = 5000, bool refine_f = true);

// Points whose two triangulating rays meet at less than this are rejected: their depth along
// the ray is barely constrained. 1.5 deg is COLMAP's default; 0 disables the check.
constexpr double kDefaultMinTriAngleDeg = 1.5;

// Triangulate the seed pair. Image seed_i defines the world frame (R = I, t = 0).
Reconstruction initialize_seed_map(int seed_i, int seed_j, const PairMatches& matches,
                                   const Keypoints& keypoints, const KpToTrack& kp_to_track,
                                   const IntrinsicsMap& intrinsics, double min_depth = 0.1,
                                   int ransac_iterations = 5000, bool refine_f = true,
                                   double min_tri_angle_deg = kDefaultMinTriAngleDeg);

struct PnPParams {
    int min_correspondences = 12;   // 2D-3D matches needed to attempt PnP
    int min_inliers = 12;           // PnP inliers needed to accept the pose (new vs Python)
    double reprojection_error = 8.0;  // PnP-RANSAC inlier threshold, px
    double min_depth = 0.1;         // new points must be this far in front of both cameras
    double max_point_error = 8.0;   // new points must reproject within this, px (new vs Python)
    double min_tri_angle_deg = kDefaultMinTriAngleDeg;  // new vs Python
};

// Register each image in `indices` (in order) via PnP-RANSAC, then triangulate its
// still-unmapped tracks against every registered neighbour it shares matches with.
void run_pnp_mapping(const std::vector<int>& indices, Reconstruction& rec,
                     const KpToTrack& kp_to_track, const PairMatches& matches,
                     const Keypoints& keypoints, const IntrinsicsMap& intrinsics,
                     const PnPParams& params = {});

// Triangulate tracks not yet in the map from their widest-baseline pair of registered views.
// Returns the number of points added.
int retriangulate_tracks(Reconstruction& rec, const Tracks& tracks, const Keypoints& keypoints,
                         const IntrinsicsMap& intrinsics, double min_depth = 0.1,
                         double max_reproj_error = 4.0,
                         double min_tri_angle_deg = kDefaultMinTriAngleDeg);

struct BadTrack {
    int track_id;
    int image;
    double depth;
};

// Tracks with depth <= 0 in at least one registered camera that observes them.
std::vector<BadTrack> diagnose_map_quality(const Reconstruction& rec, const KpToTrack& kp_to_track,
                                           const Keypoints& keypoints);

struct ReprojectionStats {
    int num_observations = 0;  // observations of mapped tracks in registered images
    int num_behind = 0;        // of those, how many are behind the camera (not in the stats)
    double mean = 0.0;         // mean 2D error, px
    double rms = 0.0;          // sqrt(mean squared 2D error), px
    double median = 0.0;
    double p90 = 0.0;
    double p99 = 0.0;
    double max = 0.0;
    int num_over_10px = 0;
};

ReprojectionStats compute_reprojection_stats(const Reconstruction& rec,
                                             const KpToTrack& kp_to_track,
                                             const Keypoints& keypoints,
                                             const IntrinsicsMap& intrinsics);

}  // namespace sfm
