// Module 6 check:
//   1. synthetic scene: perturbed poses/points, 0.5 px noise and injected outliers; BA must
//      recover the truth, remove the outliers and leave the anchor untouched
//   2. optional: the full notebook flow on the real dataset, then BA
// Usage: test_ba [data_dir [--no-refine] [--no-undistort] [--keep-conflicting]]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <string>

#include <Eigen/Geometry>

#include "sfm/ba.hpp"
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

double rot_err_deg(const Eigen::Matrix3d& A, const Eigen::Matrix3d& B) {
    return Eigen::AngleAxisd(A.transpose() * B).angle() * 180.0 / M_PI;
}

// noise_px = pixel noise per axis; with_outliers = inject 4.5% bad observations.
void test_synthetic(double noise_px, bool with_outliers) {
    std::cout << "\n--- synthetic: noise " << noise_px << " px, outliers " << std::boolalpha << with_outliers << " ---\n";
    std::mt19937 gen(11);
    std::normal_distribution<double> px_noise(0.0, 1.0), unit(0.0, 1.0);
    std::uniform_real_distribution<double> u(-1.0, 1.0), offset(5.0, 30.0), coin(0.0, 1.0);

    Eigen::Matrix3d K;
    K << 1700, 0, 540,
         0, 1700, 960,
         0, 0, 1;
    const int num_cams = 6, num_points = 600;
    sfm::IntrinsicsMap intrinsics;
    for (int c = 0; c < num_cams; ++c) intrinsics[c + 1].K = K;

    // Cameras on an arc looking at the origin, then re-expressed so camera 0 is the world frame.
    std::vector<sfm::Pose> truth(num_cams);
    for (int c = 0; c < num_cams; ++c) {
        const double a = (-25.0 + 10.0 * c) * M_PI / 180.0;
        const Eigen::Vector3d center(6.0 * std::sin(a), 0.0, -6.0 * std::cos(a));
        const Eigen::Vector3d z = (-center).normalized();
        const Eigen::Vector3d x = Eigen::Vector3d::UnitY().cross(z).normalized();
        Eigen::Matrix3d R;
        R << x.transpose(), z.cross(x).transpose(), z.transpose();
        truth[c] = {R, -R * center};
    }
    const sfm::Pose T0 = truth[0];
    std::vector<Eigen::Vector3d> X_true(num_points);
    for (auto& X : X_true) X = T0.R * Eigen::Vector3d(u(gen), 1.5 * u(gen), u(gen)) + T0.t;
    for (auto& T : truth) {
        const Eigen::Matrix3d R = T.R * T0.R.transpose();
        T = {R, T.t - R * T0.t};
    }

    // Observations: noisy projections, 4% of them shifted 5-30 px, 0.5% shifted 500 px.
    sfm::Keypoints kps(num_cams);
    sfm::KpToTrack kp_to_track(num_cams);
    int num_outliers = 0;
    std::set<std::pair<int, int>> outlier_obs;  // (image, kp)
    for (int c = 0; c < num_cams; ++c) {
        for (int p = 0; p < num_points; ++p) {
            Eigen::Vector2d uv = (K * (truth[c].R * X_true[p] + truth[c].t)).hnormalized();
            if (uv.x() < 0 || uv.x() >= 1080 || uv.y() < 0 || uv.y() >= 1920) continue;
            uv += noise_px * Eigen::Vector2d(px_noise(gen), px_noise(gen));
            const double r = coin(gen);
            if (with_outliers && r < 0.045) {
                const double mag = r < 0.005 ? 500.0 : offset(gen);
                const double ang = 2 * M_PI * coin(gen);
                uv += mag * Eigen::Vector2d(std::cos(ang), std::sin(ang));
                outlier_obs.insert({c, static_cast<int>(kps[c].size())});
                ++num_outliers;
            }
            kp_to_track[c][static_cast<int>(kps[c].size())] = p;
            kps[c].emplace_back(cv::Point2f(float(uv.x()), float(uv.y())), 1.f);
        }
    }

    // Start from a perturbed solution sized like the real pre-BA map (errors of a few px):
    // 0.05 deg / 5 mm on poses (not the anchor), 3 mm on points. At f = 1700 and ~6 m that is
    // roughly 1.5 px per term; much larger and the 10 px pre-filter rightly discards most data.
    sfm::Reconstruction start;
    std::vector<double> rot_before(num_cams, 0.0);
    for (int c = 0; c < num_cams; ++c) {
        sfm::Pose P = truth[c];
        if (c != 0) {
            const Eigen::Vector3d axis = Eigen::Vector3d(unit(gen), unit(gen), unit(gen)).normalized();
            P.R = Eigen::AngleAxisd(0.05 * M_PI / 180.0, axis).toRotationMatrix() * P.R;
            P.t += 0.005 * Eigen::Vector3d(unit(gen), unit(gen), unit(gen));
            rot_before[c] = rot_err_deg(P.R, truth[c].R);
        }
        start.add_pose(c, P);
    }
    for (int p = 0; p < num_points; ++p) {
        start.points[p] = X_true[p] + 0.003 * Eigen::Vector3d(unit(gen), unit(gen), unit(gen));
    }

    sfm::BAParams params;
    params.anchor_image = 0;
    const sfm::BAResult res = sfm::run_bundle_adjustment(start, kp_to_track, kps, intrinsics, params);
    const sfm::Reconstruction& out = res.rec;

    // Rotations are unaffected by the free scale; camera centers are compared after fitting it.
    double max_rot = 0, num = 0, den = 0, max_center_err = 0;
    for (int c = 0; c < num_cams; ++c) {
        max_rot = std::max(max_rot, rot_err_deg(out.poses.at(c).R, truth[c].R));
        num += out.poses.at(c).center().dot(truth[c].center());
        den += truth[c].center().squaredNorm();
    }
    const double s = num / den;
    for (int c = 0; c < num_cams; ++c) {
        max_center_err = std::max(max_center_err, (out.poses.at(c).center() - s * truth[c].center()).norm());
    }
    const double baseline = (truth[1].center() - truth[0].center()).norm();

    // How many injected outliers survived into the final round?
    int outliers_left = 0;
    for (const auto& [c, kp] : outlier_obs) {
        const int tid = kp_to_track[c].at(kp);
        if (!out.points.count(tid)) continue;
        const Eigen::Vector3d Xc = out.poses.at(c).R * out.points.at(tid) + out.poses.at(c).t;
        const Eigen::Vector2d uv(kps[c][kp].pt.x, kps[c][kp].pt.y);
        outliers_left += ((K * Xc).hnormalized() - uv).norm() <= params.max_final_error;
    }
    const auto& last = res.rounds.back();
    std::cout << "  " << num_outliers << " outliers injected, " << res.dropped_initial << " dropped before BA, "
              << res.dropped_between << " between rounds, " << outliers_left << " still within 4 px\n"
              << "  max rotation error " << max_rot << " deg, max center error " << max_center_err / baseline * 100
              << "% of a baseline (scale " << s << "), final plain RMS " << last.plain_rms << " px\n";

    check(res.rounds.size() == 2, "synthetic: two BA rounds ran");
    check(out.poses.at(0).R.isApprox(truth[0].R, 1e-12) && out.poses.at(0).t.isApprox(truth[0].t, 1e-12),
          "synthetic: anchor camera unchanged");
    const double max_rot_before = *std::max_element(rot_before.begin(), rot_before.end());
    std::cout << "  max rotation error before BA " << max_rot_before << " deg\n";
    if (noise_px == 0.0) {
        check(max_rot < 1e-4, "synthetic: noise-free rotations recovered exactly (< 1e-4 deg)");
        check(max_center_err / baseline < 1e-5, "synthetic: noise-free centers recovered exactly");
        check(last.plain_rms < 1e-3, "synthetic: noise-free RMS ~ 0");
    } else {
        check(max_center_err / baseline < 0.01, "synthetic: camera centers within 1% of a baseline (after scale)");
        // Noise of sigma per axis gives an expected 2D error RMS of sigma * sqrt(2).
        check(std::abs(last.plain_rms - noise_px * std::sqrt(2.0)) < 0.25 * noise_px,
              "synthetic: final RMS matches the pixel noise level");
    }
    check(res.dropped_initial + res.dropped_between >= 0.9 * num_outliers,
          "synthetic: at least 90% of outliers removed");
    check(last.final_cost < res.rounds.front().initial_cost, "synthetic: cost decreased");
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

    const sfm::ImageSet set = sfm::load_images(data_dir + "/buddha_images");
    const sfm::IntrinsicsMap intr = sfm::load_intrinsics(data_dir + "/cameras.txt", false);
    const sfm::FeatureSet fs = sfm::extract_features_and_matches(set.images, intr, params);
    const sfm::TrackSet ts = sfm::build_global_tracks(fs.keypoints, fs.matches, drop);

    const int seed_i = 15, seed_j = 16, n = static_cast<int>(set.images.size());
    sfm::Reconstruction rec = sfm::initialize_seed_map(seed_i, seed_j, fs.matches, fs.keypoints,
                                                       ts.kp_to_track, intr, 0.1, 5000, params.refine_f);
    std::vector<int> forward(n - seed_j - 1), reverse(seed_i);
    std::iota(forward.begin(), forward.end(), seed_j + 1);
    for (int k = 0; k < seed_i; ++k) reverse[k] = seed_i - 1 - k;
    sfm::run_pnp_mapping(forward, rec, ts.kp_to_track, fs.matches, fs.keypoints, intr);
    sfm::run_pnp_mapping(reverse, rec, ts.kp_to_track, fs.matches, fs.keypoints, intr);
    sfm::retriangulate_tracks(rec, ts.tracks, fs.keypoints, intr);

    std::cout << "\n--- Bundle adjustment ---\n";
    const auto before = sfm::compute_reprojection_stats(rec, ts.kp_to_track, fs.keypoints, intr);
    const sfm::BAResult res = sfm::run_bundle_adjustment(rec, ts.kp_to_track, fs.keypoints, intr);
    const auto after = sfm::compute_reprojection_stats(res.rec, ts.kp_to_track, fs.keypoints, intr);
    const auto bad = sfm::diagnose_map_quality(res.rec, ts.kp_to_track, fs.keypoints);

    auto line = [](const char* name, const sfm::ReprojectionStats& s) {
        std::cout << "  " << name << ": " << s.num_observations << " obs, median " << s.median << ", p90 "
                  << s.p90 << ", p99 " << s.p99 << ", mean " << s.mean << " px\n";
    };
    std::cout << "\nAll observations of mapped tracks:\n";
    line("before BA", before);
    line("after BA ", after);
    const auto& last = res.rounds.back();
    std::cout << "SUMMARY points " << rec.points.size() << " -> " << res.rec.points.size()
              << " | BA observations " << last.num_observations << " | Python-style RMSE "
              << last.python_rmse << " px | plain RMS " << last.plain_rms << " px | negative-depth "
              << bad.size() << "\n";
    check(res.rounds.size() == 2 && last.final_cost < res.rounds.front().initial_cost, "real: BA ran and lowered the cost");
    check(res.rec.poses.at(seed_i).R.isApprox(Eigen::Matrix3d::Identity(), 1e-12), "real: anchor (seed) camera unchanged");
}

}  // namespace

int main(int argc, char** argv) {
    sfm::set_random_seed(0);
    test_synthetic(0.0, false);
    test_synthetic(0.5, true);
    if (argc > 1) run_real_data(argv[1], argc, argv);
    std::cout << (failures == 0 ? "All checks passed.\n" : "Some checks FAILED.\n");
    return failures == 0 ? 0 : 1;
}
