// Module 5 check (plus the undistortion added to module 3):
//   1. distortion round trip on a pixel grid
//   2. synthetic 6-camera scene with perfect correspondences: every pose and point recovered
//   3. optional: the notebook's mapping flow on the real dataset
// Usage: test_mapping [data_dir [--no-refine] [--no-undistort] [--keep-conflicting]]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <string>

#include <Eigen/Geometry>

#include "sfm/camera.hpp"
#include "sfm/io.hpp"
#include "sfm/mapping.hpp"
#include "sfm/matching.hpp"
#include "sfm/rng.hpp"
#include "sfm/tracks.hpp"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (!ok) ++failures;
}

sfm::Intrinsics buddha_camera() {
    sfm::Intrinsics intr;
    intr.K << 1700.596, 0, 540,
              0, 1700.596, 960,
              0, 0, 1;
    intr.k1 = 0.044607;
    intr.width = 1080;
    intr.height = 1920;
    return intr;
}

void test_distortion() {
    const sfm::Intrinsics intr = buddha_camera();
    double max_round_trip = 0;
    for (double u = 0; u <= 1080; u += 54) {
        for (double v = 0; v <= 1920; v += 96) {
            const Eigen::Vector2d p(u, v);
            max_round_trip = std::max(max_round_trip, (sfm::distort_pixel(sfm::undistort_pixel(p, intr), intr) - p).norm());
            max_round_trip = std::max(max_round_trip, (sfm::undistort_pixel(sfm::distort_pixel(p, intr), intr) - p).norm());
        }
    }
    const double corner_shift = (sfm::undistort_pixel({0, 0}, intr) - Eigen::Vector2d(0, 0)).norm();
    const double center_shift = (sfm::undistort_pixel({540, 960}, intr) - Eigen::Vector2d(540, 960)).norm();
    std::cout << "  max round-trip error " << max_round_trip << " px, corner shift " << corner_shift
              << " px, center shift " << center_shift << " px\n";
    check(max_round_trip < 1e-6, "undistort and distort invert each other (< 1e-6 px)");
    check(corner_shift > 15 && corner_shift < 25 && center_shift == 0.0, "corner moves ~20 px, center does not move");
}

// Six cameras on an arc looking at a cloud of points; every pair within 3 frames is matched.
void test_synthetic_mapping() {
    sfm::set_random_seed(0);
    std::mt19937 gen(7);
    std::uniform_real_distribution<double> uxy(-1.0, 1.0), uz(-1.0, 1.0);
    const sfm::Intrinsics cam = buddha_camera();
    const int num_cams = 6, num_points = 600;

    sfm::IntrinsicsMap intrinsics;
    std::vector<sfm::Pose> truth(num_cams);
    for (int c = 0; c < num_cams; ++c) {
        intrinsics[c + 1] = cam;
        const double a = (-25.0 + 10.0 * c) * M_PI / 180.0;  // cameras from -25 to +25 deg
        const Eigen::Vector3d center(6.0 * std::sin(a), 0.0, -6.0 * std::cos(a));
        // Look at the origin: camera z axis points from the center toward (0, 0, 0).
        const Eigen::Vector3d z = (-center).normalized();
        const Eigen::Vector3d x = Eigen::Vector3d::UnitY().cross(z).normalized();
        const Eigen::Vector3d y = z.cross(x);
        Eigen::Matrix3d R;
        R << x.transpose(), y.transpose(), z.transpose();
        truth[c] = {R, -R * center};
    }

    std::vector<Eigen::Vector3d> X(num_points);
    for (auto& p : X) p = Eigen::Vector3d(uxy(gen), 1.5 * uxy(gen), uz(gen));

    // Keypoint k of image c is point point_of[c][k].
    sfm::Keypoints kps(num_cams);
    std::vector<std::vector<int>> point_of(num_cams);
    std::vector<std::vector<int>> kp_of(num_cams, std::vector<int>(num_points, -1));
    for (int c = 0; c < num_cams; ++c) {
        for (int p = 0; p < num_points; ++p) {
            const Eigen::Vector2d uv = (cam.K * (truth[c].R * X[p] + truth[c].t)).hnormalized();
            if (uv.x() < 0 || uv.x() >= 1080 || uv.y() < 0 || uv.y() >= 1920) continue;
            kp_of[c][p] = static_cast<int>(kps[c].size());
            kps[c].emplace_back(cv::Point2f(float(uv.x()), float(uv.y())), 1.f);
            point_of[c].push_back(p);
        }
    }
    sfm::PairMatches matches;
    for (int i = 0; i < num_cams; ++i) {
        for (int j = i + 1; j < std::min(num_cams, i + 4); ++j) {
            for (int p = 0; p < num_points; ++p) {
                if (kp_of[i][p] >= 0 && kp_of[j][p] >= 0) matches[{i, j}].emplace_back(kp_of[i][p], kp_of[j][p], 0.f);
            }
        }
    }

    const sfm::TrackSet ts = sfm::build_global_tracks(kps, matches);
    sfm::Reconstruction rec = sfm::initialize_seed_map(2, 3, matches, kps, ts.kp_to_track, intrinsics);
    sfm::run_pnp_mapping({4, 5}, rec, ts.kp_to_track, matches, kps, intrinsics);
    sfm::run_pnp_mapping({1, 0}, rec, ts.kp_to_track, matches, kps, intrinsics);
    sfm::retriangulate_tracks(rec, ts.tracks, kps, intrinsics);

    // World frame = camera 2, so estimated R_c should equal R_c_true * R_2_true^T.
    double max_rot_err = 0;
    for (const auto& [c, pose] : rec.poses) {
        const Eigen::Matrix3d expected = truth[c].R * truth[2].R.transpose();
        max_rot_err = std::max(max_rot_err, Eigen::AngleAxisd(pose.R.transpose() * expected).angle() * 180 / M_PI);
    }
    const auto stats = sfm::compute_reprojection_stats(rec, ts.kp_to_track, kps, intrinsics);
    const auto bad = sfm::diagnose_map_quality(rec, ts.kp_to_track, kps);
    std::cout << "  " << rec.poses.size() << " cameras, " << rec.points.size() << "/" << ts.tracks.size()
              << " tracks mapped, max rotation error " << max_rot_err << " deg, reprojection RMS "
              << stats.rms << " px\n";
    check(rec.poses.size() == num_cams, "synthetic: all 6 cameras registered");
    check(rec.points.size() > 0.95 * ts.tracks.size(), "synthetic: > 95% of tracks mapped");
    check(max_rot_err < 0.05, "synthetic: rotations match ground truth (< 0.05 deg)");
    check(stats.rms < 0.05 && stats.num_behind == 0 && bad.empty(), "synthetic: reprojection RMS < 0.05 px, nothing behind a camera");
    const std::vector<int> expected_order{2, 3, 4, 5, 1, 0};
    check(rec.registration_order == expected_order, "synthetic: registration order is 2 3 4 5 1 0");
}

void run_real_data(const std::string& data_dir, int argc, char** argv) {
    sfm::MatchingParams params;
    bool drop = true;
    for (int a = 2; a < argc; ++a) {
        if (std::string(argv[a]) == "--no-refine") params.refine_f = false;
        if (std::string(argv[a]) == "--no-undistort") params.undistort = false;
        if (std::string(argv[a]) == "--keep-conflicting") drop = false;
    }
    std::cout << "\n=== Real data: " << std::boolalpha << "refine_f = " << params.refine_f
              << ", undistort = " << params.undistort << ", drop_conflicting = " << drop << " ===\n";

    const auto t0 = std::chrono::steady_clock::now();
    const sfm::ImageSet set = sfm::load_images(data_dir + "/buddha_images");
    const sfm::IntrinsicsMap intr = sfm::load_intrinsics(data_dir + "/cameras.txt", false);
    const sfm::FeatureSet fs = sfm::extract_features_and_matches(set.images, intr, params);
    const sfm::TrackSet ts = sfm::build_global_tracks(fs.keypoints, fs.matches, drop);

    const auto best = sfm::find_best_seed(fs.matches, intr, fs.keypoints, 15, 2.0, 5000, params.refine_f);
    if (best) std::cout << "find_best_seed suggests: (" << best->pair.first << ", " << best->pair.second << ")\n";

    // Same flow as the notebook: fixed seed (15, 16), forward pass, reverse pass, retriangulate.
    const int seed_i = 15, seed_j = 16, n = static_cast<int>(set.images.size());
    sfm::Reconstruction rec = sfm::initialize_seed_map(seed_i, seed_j, fs.matches, fs.keypoints,
                                                       ts.kp_to_track, intr, 0.1, 5000, params.refine_f);
    std::vector<int> forward(n - seed_j - 1), reverse(seed_i);
    std::iota(forward.begin(), forward.end(), seed_j + 1);
    for (int k = 0; k < seed_i; ++k) reverse[k] = seed_i - 1 - k;
    std::cout << "Starting Forward Pass\n";
    sfm::run_pnp_mapping(forward, rec, ts.kp_to_track, fs.matches, fs.keypoints, intr);
    std::cout << "\nStarting Reverse Pass\n";
    sfm::run_pnp_mapping(reverse, rec, ts.kp_to_track, fs.matches, fs.keypoints, intr);
    std::cout << "\nMap after PnP: " << rec.points.size() << " points\n";
    sfm::retriangulate_tracks(rec, ts.tracks, fs.keypoints, intr);
    const auto bad = sfm::diagnose_map_quality(rec, ts.kp_to_track, fs.keypoints);
    const auto stats = sfm::compute_reprojection_stats(rec, ts.kp_to_track, fs.keypoints, intr);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::cout << "\nSUMMARY cameras " << rec.poses.size() << "/" << n << " | points " << rec.points.size()
              << " | negative-depth tracks " << bad.size() << " | observations " << stats.num_observations
              << " (" << stats.num_behind << " behind) | reprojection mean " << stats.mean << " px, median "
              << stats.median << ", p90 " << stats.p90 << ", p99 " << stats.p99 << ", max " << stats.max
              << ", RMS " << stats.rms << " px, " << stats.num_over_10px << " obs > 10 px | " << secs << " s\n";
    check(static_cast<int>(rec.poses.size()) == n, "real: all cameras registered");
    check(!rec.points.empty(), "real: map is not empty");
}

}  // namespace

int main(int argc, char** argv) {
    test_distortion();
    test_synthetic_mapping();
    if (argc > 1) run_real_data(argv[1], argc, argv);
    std::cout << (failures == 0 ? "All checks passed.\n" : "Some checks FAILED.\n");
    return failures == 0 ? 0 : 1;
}
