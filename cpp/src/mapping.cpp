#include "sfm/mapping.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>

#include <Eigen/Geometry>  // hnormalized()
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include "sfm/matching.hpp"

namespace sfm {

namespace {

const Eigen::Matrix3d& K_of(const IntrinsicsMap& intrinsics, int image) {
    return intrinsics.at(image + 1).K;  // image i uses camera i + 1
}

Eigen::Vector2d pixel(const Keypoints& keypoints, int image, int kp) {
    const cv::Point2f& pt = keypoints[image][kp].pt;
    return {pt.x, pt.y};
}

std::optional<int> track_of(const KpToTrack& kp_to_track, int image, int kp) {
    const auto it = kp_to_track[image].find(kp);
    if (it == kp_to_track[image].end()) return std::nullopt;  // e.g. a dropped conflicting track
    return it->second;
}

FundamentalResult fit_fundamental(const Points2D& p1, const Points2D& p2, int iterations,
                                  bool refine_f) {
    FundamentalResult res = ransac_fundamental_matrix(p1, p2, iterations);
    return refine_f ? refine_fundamental_matrix(res, p1, p2) : res;
}

// Point must be in front of the camera by min_depth and reproject within max_error.
// "!(z > min_depth)" rather than "z <= min_depth" so NaN points are rejected too.
bool accept_in_view(const Eigen::Vector3d& X, const Pose& pose, const Eigen::Matrix3d& K,
                    const Eigen::Vector2d& uv, double min_depth, double max_error) {
    const Eigen::Vector3d Xc = pose.R * X + pose.t;
    if (!(Xc.z() > min_depth)) return false;
    const Eigen::Vector2d proj = (K * Xc).hnormalized();
    return (proj - uv).norm() <= max_error;
}

// Angle between the rays from the two camera centers to X must be at least min_angle_deg.
bool wide_enough(const Eigen::Vector3d& X, const Pose& a, const Pose& b, double min_angle_deg) {
    if (min_angle_deg <= 0) return true;
    const Eigen::Vector3d ra = X - a.center(), rb = X - b.center();
    const double denom = ra.norm() * rb.norm();
    if (!(denom > 0)) return false;
    const double cos_angle = std::clamp(ra.dot(rb) / denom, -1.0, 1.0);
    return std::acos(cos_angle) * 180.0 / M_PI >= min_angle_deg;
}

}  // namespace

void Reconstruction::add_pose(int image, const Pose& pose) {
    if (!poses.count(image)) registration_order.push_back(image);
    poses[image] = pose;
}

std::optional<SeedCandidate> find_best_seed(const PairMatches& matches,
                                            const IntrinsicsMap& intrinsics,
                                            const Keypoints& keypoints, int top_k,
                                            double min_parallax_deg, int ransac_iterations,
                                            bool refine_f) {
    std::vector<ImagePair> candidates;
    for (const auto& kv : matches) candidates.push_back(kv.first);
    // Stable sort keeps (i, j) order among ties, like Python's sorted() over the dict.
    std::stable_sort(candidates.begin(), candidates.end(), [&](const auto& a, const auto& b) {
        return matches.at(a).size() > matches.at(b).size();
    });
    if (static_cast<int>(candidates.size()) > top_k) candidates.resize(top_k);

    std::optional<SeedCandidate> best;
    size_t max_inliers = 0;
    for (const auto& pair : candidates) {
        const auto [i, j] = pair;
        const auto& pair_matches = matches.at(pair);
        const Eigen::Matrix3d& Ki = K_of(intrinsics, i);
        const Eigen::Matrix3d& Kj = K_of(intrinsics, j);
        const auto [p1, p2] = match_points(pair_matches, keypoints[i], keypoints[j]);

        const FundamentalResult res = fit_fundamental(p1, p2, ransac_iterations, refine_f);
        if (!res.F) continue;

        const Eigen::Matrix3d E = compute_essential_matrix(*res.F, Ki, Kj);
        const auto [R, t] = cheirality_check(estimate_camera_pose(E), Ki, Kj, p1, p2);
        const double parallax = calculate_median_parallax(p1, p2, Ki, Kj, R, t);

        std::cout << std::fixed << std::setprecision(2);
        if (parallax > min_parallax_deg) {
            std::cout << "Candidate (" << i << ", " << j << "): " << pair_matches.size()
                      << " inliers, " << parallax << "° parallax - VALID\n\n";
            if (pair_matches.size() > max_inliers) {
                max_inliers = pair_matches.size();
                best = SeedCandidate{pair, {R, t}};
            }
        } else {
            std::cout << "Candidate (" << i << ", " << j << "): " << parallax
                      << "° parallax - TOO LOW (Narrow Baseline)\n\n";
        }
        std::cout << std::defaultfloat;
    }
    return best;
}

Reconstruction initialize_seed_map(int seed_i, int seed_j, const PairMatches& matches,
                                   const Keypoints& keypoints, const KpToTrack& kp_to_track,
                                   const IntrinsicsMap& intrinsics, double min_depth,
                                   int ransac_iterations, bool refine_f,
                                   double min_tri_angle_deg) {
    const auto& pair_matches = matches.at({seed_i, seed_j});
    const auto [p1, p2] = match_points(pair_matches, keypoints[seed_i], keypoints[seed_j]);
    const FundamentalResult res = fit_fundamental(p1, p2, ransac_iterations, refine_f);
    if (!res.F) throw std::runtime_error("Seed pair has no valid fundamental matrix");

    Points2D in1, in2;
    for (int k : res.inliers) {
        in1.push_back(p1[k]);
        in2.push_back(p2[k]);
    }

    const Eigen::Matrix3d& Ki = K_of(intrinsics, seed_i);
    const Eigen::Matrix3d& Kj = K_of(intrinsics, seed_j);
    const Eigen::Matrix3d E = compute_essential_matrix(*res.F, Ki, Kj);
    const auto [R, t] = cheirality_check(estimate_camera_pose(E), Ki, Kj, in1, in2);

    Reconstruction rec;
    rec.add_pose(seed_i, Pose{});
    rec.add_pose(seed_j, Pose{R, t});

    const Points3D X = triangulate_points(
        projection_matrix(Ki, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()),
        projection_matrix(Kj, R, t), in1, in2);

    int narrow = 0;
    for (size_t k = 0; k < X.size(); ++k) {
        // Both views must see the point in front of them (NaN fails these tests).
        if (!(X[k].z() > min_depth) || !((R * X[k] + t).z() > min_depth)) continue;
        if (!wide_enough(X[k], rec.poses.at(seed_i), rec.poses.at(seed_j), min_tri_angle_deg)) {
            ++narrow;
            continue;
        }
        const auto tid = track_of(kp_to_track, seed_i, pair_matches[res.inliers[k]].queryIdx);
        if (tid) rec.points[*tid] = X[k];
    }

    if (narrow > 0) {
        std::cout << "Seed: rejected " << narrow << " points with triangulation angle < "
                  << min_tri_angle_deg << " deg.\n";
    }
    std::cout << "Initialized Map with " << rec.points.size() << " 3D points.\n"
              << "Seed Pair (" << seed_i << ", " << seed_j << ") registered successfully.\n";
    return rec;
}

void run_pnp_mapping(const std::vector<int>& indices, Reconstruction& rec,
                     const KpToTrack& kp_to_track, const PairMatches& matches,
                     const Keypoints& keypoints, const IntrinsicsMap& intrinsics,
                     const PnPParams& params) {
    for (int i : indices) {
        if (rec.poses.count(i)) continue;

        // 2D-3D correspondences from tracks that are already mapped.
        std::vector<cv::Point3d> pts_3d;
        std::vector<cv::Point2d> pts_2d;
        for (const auto& [kp_idx, tid] : kp_to_track[i]) {
            const auto it = rec.points.find(tid);
            if (it == rec.points.end()) continue;
            pts_3d.emplace_back(it->second.x(), it->second.y(), it->second.z());
            pts_2d.emplace_back(keypoints[i][kp_idx].pt.x, keypoints[i][kp_idx].pt.y);
        }
        if (static_cast<int>(pts_3d.size()) < params.min_correspondences) {
            std::cout << "Skipping image " << i << ": Only " << pts_3d.size()
                      << " correspondences.\n";
            continue;
        }

        const Eigen::Matrix3d& K_curr = K_of(intrinsics, i);
        cv::Mat K_cv, rvec, tvec;
        cv::eigen2cv(K_curr, K_cv);
        std::vector<int> inliers;
        // Keypoints are already undistorted, so no distortion coefficients here.
        const bool ok = cv::solvePnPRansac(pts_3d, pts_2d, K_cv, cv::noArray(), rvec, tvec,
                                           false, 100, params.reprojection_error, 0.99, inliers,
                                           cv::SOLVEPNP_ITERATIVE);
        if (!ok) {
            std::cout << "Skipping image " << i << ": PnP failed.\n";
            continue;
        }
        if (static_cast<int>(inliers.size()) < params.min_inliers) {
            std::cout << "Skipping image " << i << ": Only " << inliers.size()
                      << " PnP inliers.\n";
            continue;
        }

        cv::Mat R_cv;
        cv::Rodrigues(rvec, R_cv);
        Pose pose;
        cv::cv2eigen(R_cv, pose.R);
        cv::cv2eigen(tvec, pose.t);
        rec.add_pose(i, pose);
        const Mat34 P_i = projection_matrix(K_curr, pose.R, pose.t);

        int new_points = 0;
        // Copy: add_pose above already appended i, and the loop must not see later changes.
        const std::vector<int> neighbours = rec.registration_order;
        for (int p : neighbours) {
            if (p == i) continue;
            const ImagePair pair{std::min(p, i), std::max(p, i)};
            const auto mit = matches.find(pair);
            if (mit == matches.end()) continue;

            const Pose& pose_p = rec.poses.at(p);
            const Eigen::Matrix3d& K_p = K_of(intrinsics, p);
            const Mat34 P_p = projection_matrix(K_p, pose_p.R, pose_p.t);

            Points2D pts_p, pts_i;
            std::vector<int> tids;
            for (const auto& m : mit->second) {
                const int idx_i = pair.first == i ? m.queryIdx : m.trainIdx;
                const int idx_p = pair.first == i ? m.trainIdx : m.queryIdx;
                const auto tid = track_of(kp_to_track, i, idx_i);
                if (!tid || rec.points.count(*tid)) continue;
                pts_p.push_back(pixel(keypoints, p, idx_p));
                pts_i.push_back(pixel(keypoints, i, idx_i));
                tids.push_back(*tid);
            }
            if (tids.empty()) continue;

            const Points3D X = triangulate_points(P_p, P_i, pts_p, pts_i);
            for (size_t k = 0; k < X.size(); ++k) {
                // Fix vs Python: check depth in both cameras (not world z) and reprojection.
                if (!accept_in_view(X[k], pose_p, K_p, pts_p[k], params.min_depth,
                                    params.max_point_error) ||
                    !accept_in_view(X[k], pose, K_curr, pts_i[k], params.min_depth,
                                    params.max_point_error) ||
                    !wide_enough(X[k], pose_p, pose, params.min_tri_angle_deg)) {
                    continue;
                }
                rec.points[tids[k]] = X[k];
                ++new_points;
            }
        }
        std::cout << "Registered " << i << ": " << inliers.size() << " inliers, added "
                  << new_points << " points.\n";
    }
}

int retriangulate_tracks(Reconstruction& rec, const Tracks& tracks, const Keypoints& keypoints,
                         const IntrinsicsMap& intrinsics, double min_depth,
                         double max_reproj_error, double min_tri_angle_deg) {
    int added = 0, rejected = 0;
    for (const auto& [tid, observations] : tracks) {
        if (rec.points.count(tid)) continue;

        std::vector<Observation> registered;
        for (const auto& obs : observations) {
            if (rec.poses.count(obs.first)) registered.push_back(obs);
        }
        if (registered.size() < 2) continue;

        // Widest baseline (camera-center distance) gives the best-conditioned triangulation.
        size_t best_a = 0, best_b = 1;
        double best_baseline = -1.0;
        for (size_t a = 0; a < registered.size(); ++a) {
            for (size_t b = a + 1; b < registered.size(); ++b) {
                const double baseline = (rec.poses.at(registered[a].first).center() -
                                         rec.poses.at(registered[b].first).center()).norm();
                if (baseline > best_baseline) {
                    best_baseline = baseline;
                    best_a = a;
                    best_b = b;
                }
            }
        }

        const auto [i, kpi] = registered[best_a];
        const auto [j, kpj] = registered[best_b];
        const Pose& pi = rec.poses.at(i);
        const Pose& pj = rec.poses.at(j);
        const Eigen::Matrix3d& Ki = K_of(intrinsics, i);
        const Eigen::Matrix3d& Kj = K_of(intrinsics, j);
        const Eigen::Vector2d uv_i = pixel(keypoints, i, kpi);
        const Eigen::Vector2d uv_j = pixel(keypoints, j, kpj);

        const Eigen::Vector3d X = triangulate_points(projection_matrix(Ki, pi.R, pi.t),
                                                     projection_matrix(Kj, pj.R, pj.t),
                                                     {uv_i}, {uv_j})[0];
        if (!accept_in_view(X, pi, Ki, uv_i, min_depth, max_reproj_error) ||
            !accept_in_view(X, pj, Kj, uv_j, min_depth, max_reproj_error) ||
            !wide_enough(X, pi, pj, min_tri_angle_deg)) {
            ++rejected;
            continue;
        }
        rec.points[tid] = X;
        ++added;
    }
    std::cout << "Retriangulation: added " << added << " points (rejected " << rejected
              << ", map now " << rec.points.size() << " points).\n";
    return added;
}

std::vector<BadTrack> diagnose_map_quality(const Reconstruction& rec, const KpToTrack& kp_to_track,
                                           const Keypoints& keypoints) {
    // Observing registered cameras per track.
    std::map<int, std::vector<int>> cams_by_track;
    for (size_t img = 0; img < kp_to_track.size() && img < keypoints.size(); ++img) {
        if (!rec.poses.count(static_cast<int>(img))) continue;
        for (const auto& kv : kp_to_track[img]) cams_by_track[kv.second].push_back(static_cast<int>(img));
    }

    std::vector<BadTrack> bad;
    for (const auto& [tid, X] : rec.points) {
        for (int cam : cams_by_track[tid]) {
            const Pose& pose = rec.poses.at(cam);
            const double depth = (pose.R * X + pose.t).z();
            if (depth <= 0) {
                bad.push_back({tid, cam, depth});
                break;
            }
        }
    }

    std::cout << "\n=== Diagnosing Map Quality ===\n"
              << "Total tracks in map: " << rec.points.size() << "\n"
              << "Tracks with negative depth in at least one observer: " << bad.size() << "\n";
    if (!bad.empty()) {
        std::cout << "\nFirst 10 bad tracks (tid, cam_idx, depth):\n" << std::fixed << std::setprecision(3);
        for (size_t k = 0; k < std::min<size_t>(10, bad.size()); ++k) {
            std::cout << "  Track " << bad[k].track_id << ": depth " << bad[k].depth << " in camera "
                      << bad[k].image << "\n";
        }
        std::cout << std::defaultfloat;
    }
    return bad;
}

ReprojectionStats compute_reprojection_stats(const Reconstruction& rec,
                                             const KpToTrack& kp_to_track,
                                             const Keypoints& keypoints,
                                             const IntrinsicsMap& intrinsics) {
    ReprojectionStats stats;
    std::vector<double> errors;
    for (const auto& [img, pose] : rec.poses) {
        const Eigen::Matrix3d& K = K_of(intrinsics, img);
        for (const auto& [kp, tid] : kp_to_track[img]) {
            const auto it = rec.points.find(tid);
            if (it == rec.points.end()) continue;
            ++stats.num_observations;
            const Eigen::Vector3d Xc = pose.R * it->second + pose.t;
            if (!(Xc.z() > 0)) {
                ++stats.num_behind;
                continue;
            }
            errors.push_back(((K * Xc).hnormalized() - pixel(keypoints, img, kp)).norm());
        }
    }
    if (errors.empty()) return stats;

    double sum = 0, sum_sq = 0;
    for (double e : errors) {
        sum += e;
        sum_sq += e * e;
        stats.num_over_10px += e > 10.0;
    }
    stats.mean = sum / errors.size();
    stats.rms = std::sqrt(sum_sq / errors.size());
    std::sort(errors.begin(), errors.end());
    auto pct = [&](double q) { return errors[static_cast<size_t>(q * (errors.size() - 1))]; };
    stats.median = pct(0.5);
    stats.p90 = pct(0.9);
    stats.p99 = pct(0.99);
    stats.max = errors.back();
    return stats;
}

}  // namespace sfm
