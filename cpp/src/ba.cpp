#include "sfm/ba.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Geometry>

namespace sfm {

namespace {

struct Obs {
    int image;
    int track;
    Eigen::Vector2d uv;  // undistorted pixel
};

// Pinhole reprojection residual. Camera block = [angle-axis (3), t (3)], world -> camera.
struct ReprojectionError {
    ReprojectionError(const Eigen::Vector2d& uv, const Eigen::Matrix3d& K)
        : u(uv.x()), v(uv.y()), fx(K(0, 0)), fy(K(1, 1)), cx(K(0, 2)), cy(K(1, 2)) {}

    template <typename T>
    bool operator()(const T* cam, const T* point, T* residual) const {
        T p[3];
        ceres::AngleAxisRotatePoint(cam, point, p);
        p[0] += cam[3];
        p[1] += cam[4];
        p[2] += cam[5];
        residual[0] = fx * p[0] / p[2] + cx - u;
        residual[1] = fy * p[1] / p[2] + cy - v;
        return true;
    }

    static ceres::CostFunction* create(const Eigen::Vector2d& uv, const Eigen::Matrix3d& K) {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
            new ReprojectionError(uv, K));
    }

    double u, v, fx, fy, cx, cy;
};

std::array<double, 6> pose_to_params(const Pose& pose) {
    std::array<double, 6> p{};
    // Eigen matrices are column-major, which is what this Ceres overload expects.
    ceres::RotationMatrixToAngleAxis(pose.R.data(), p.data());
    for (int k = 0; k < 3; ++k) p[3 + k] = pose.t(k);
    return p;
}

Pose params_to_pose(const std::array<double, 6>& p) {
    Pose pose;
    ceres::AngleAxisToRotationMatrix(p.data(), pose.R.data());
    pose.t = Eigen::Vector3d(p[3], p[4], p[5]);
    return pose;
}

// Returns +inf for points at or behind the camera so error filters drop them.
double reprojection_error(const Reconstruction& rec, const IntrinsicsMap& intrinsics, const Obs& o) {
    const Pose& pose = rec.poses.at(o.image);
    const Eigen::Vector3d Xc = pose.R * rec.points.at(o.track) + pose.t;
    if (!(Xc.z() > 0)) return std::numeric_limits<double>::infinity();
    const Eigen::Vector2d proj = (intrinsics.at(o.image + 1).K * Xc).hnormalized();
    return (proj - o.uv).norm();
}

}  // namespace

BAResult run_bundle_adjustment(const Reconstruction& rec, const KpToTrack& kp_to_track,
                               const Keypoints& keypoints, const IntrinsicsMap& intrinsics,
                               const BAParams& params) {
    BAResult result;
    result.rec = rec;
    Reconstruction& cur = result.rec;

    const int anchor = params.anchor_image >= 0 ? params.anchor_image
                       : rec.registration_order.empty() ? -1
                                                         : rec.registration_order.front();
    if (!cur.poses.count(anchor)) throw std::runtime_error("BA anchor image has no pose");

    int scale_cam = -1, scale_coord = -1;  // camera and translation component held fixed
    if (params.fix_scale) {
        scale_cam = params.scale_image >= 0 ? params.scale_image
                    : rec.registration_order.size() > 1 ? rec.registration_order[1]
                                                        : -1;
        if (scale_cam == anchor || !cur.poses.count(scale_cam)) {
            throw std::runtime_error("BA scale image must be a registered image other than the anchor");
        }
        cur.poses.at(scale_cam).t.cwiseAbs().maxCoeff(&scale_coord);
    }

    // 1-3. Collect observations, drop ones too close / behind, then gross outliers.
    std::vector<Obs> obs;
    for (const auto& [img, pose] : cur.poses) {
        for (const auto& [kp, tid] : kp_to_track[img]) {
            const auto it = cur.points.find(tid);
            if (it == cur.points.end()) continue;
            if (!((pose.R * it->second + pose.t).z() > params.min_depth)) {
                ++result.dropped_depth;
                continue;
            }
            const cv::Point2f& pt = keypoints[img][kp].pt;
            Obs o{img, tid, Eigen::Vector2d(pt.x, pt.y)};
            if (reprojection_error(cur, intrinsics, o) > params.max_initial_error) {
                ++result.dropped_initial;
                continue;
            }
            obs.push_back(o);
        }
    }
    if (params.verbose) {
        std::cout << "BA: dropped " << result.dropped_depth << " observations for depth, "
                  << result.dropped_initial << " for error > " << params.max_initial_error
                  << " px (anchor image " << anchor;
        if (scale_cam >= 0) std::cout << ", scale fixed by t[" << scale_coord << "] of image " << scale_cam;
        std::cout << ")\n";
    }

    for (int round = 1; round <= params.rounds; ++round) {
        // 5. Only tracks seen at least twice constrain their point.
        std::map<int, int> obs_per_track;
        for (const auto& o : obs) ++obs_per_track[o.track];
        std::vector<Obs> kept;
        for (const auto& o : obs) {
            if (obs_per_track[o.track] >= 2) kept.push_back(o);
        }
        obs.swap(kept);

        // 6-7. Parameter blocks, initialized from the current reconstruction.
        std::map<int, std::array<double, 6>> cam_params;
        std::map<int, std::array<double, 3>> point_params;
        for (const auto& o : obs) {
            if (!cam_params.count(o.image)) cam_params[o.image] = pose_to_params(cur.poses.at(o.image));
            if (!point_params.count(o.track)) {
                const Eigen::Vector3d& X = cur.points.at(o.track);
                point_params[o.track] = {X.x(), X.y(), X.z()};
            }
        }

        // 8-10. Residuals with Huber loss; the anchor camera fixes the world frame.
        ceres::Problem problem;
        ceres::LossFunction* loss = new ceres::HuberLoss(params.huber_delta);  // owned by problem
        for (const auto& o : obs) {
            problem.AddResidualBlock(
                ReprojectionError::create(o.uv, intrinsics.at(o.image + 1).K), loss,
                cam_params[o.image].data(), point_params[o.track].data());
        }
        if (cam_params.count(anchor)) problem.SetParameterBlockConstant(cam_params[anchor].data());
        if (scale_cam >= 0 && cam_params.count(scale_cam)) {
            // Block layout is [angle-axis(3), t(3)], so translation component k is index 3 + k.
            problem.SetManifold(cam_params[scale_cam].data(),
                                new ceres::SubsetManifold(6, {3 + scale_coord}));
        }

        // 11. LM with a dense Schur complement: only 24 cameras, so the reduced system is tiny.
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.max_num_iterations = params.max_iterations;
        options.function_tolerance = params.function_tolerance;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        const auto t0 = std::chrono::steady_clock::now();
        ceres::Solve(options, &problem, &summary);
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        // 12. Write back. Tracks dropped from BA leave the map, as in the Python version.
        for (const auto& [img, p] : cam_params) cur.poses[img] = params_to_pose(p);
        Map3D optimized;
        for (const auto& [tid, p] : point_params) optimized[tid] = Eigen::Vector3d(p[0], p[1], p[2]);
        cur.points.swap(optimized);

        BARoundStats st;
        st.num_observations = static_cast<int>(obs.size());
        st.num_tracks = static_cast<int>(point_params.size());
        st.iterations = static_cast<int>(summary.iterations.size()) - 1;  // entry 0 is the start
        st.initial_cost = summary.initial_cost;
        st.final_cost = summary.final_cost;
        st.python_rmse = std::sqrt(summary.final_cost / std::max(2 * st.num_observations, 1));
        double sum_sq = 0;
        for (const auto& o : obs) {
            const double e = reprojection_error(cur, intrinsics, o);
            if (std::isinf(e)) {
                ++st.num_behind;
                continue;
            }
            sum_sq += e * e;
        }
        st.plain_rms = std::sqrt(sum_sq / std::max(st.num_observations - st.num_behind, 1));
        st.seconds = secs;
        result.rounds.push_back(st);

        if (params.verbose) {
            std::cout << "BA round " << round << ": " << st.num_tracks << " tracks, "
                      << st.num_observations << " observations | cost " << st.initial_cost
                      << " -> " << st.final_cost << " ("
                      << std::fixed << std::setprecision(1)
                      << 100.0 * (1.0 - st.final_cost / std::max(st.initial_cost, 1e-12))
                      << "% lower) in " << st.iterations << " iterations, " << std::setprecision(2)
                      << secs << " s | Python-style RMSE " << std::setprecision(4) << st.python_rmse
                      << " px, plain RMS " << st.plain_rms << " px, " << st.num_behind
                      << " behind camera | "
                      << ceres::TerminationTypeToString(summary.termination_type) << "\n"
                      << std::defaultfloat;
        }

        // 13. Between rounds, drop observations that are still bad under the new solution.
        if (round < params.rounds) {
            std::vector<Obs> good;
            for (const auto& o : obs) {
                if (reprojection_error(cur, intrinsics, o) <= params.max_final_error) {
                    good.push_back(o);
                } else {
                    ++result.dropped_between;
                }
            }
            if (params.verbose) {
                std::cout << "BA: dropped " << obs.size() - good.size() << " observations with error > "
                          << params.max_final_error << " px after round " << round << "\n";
            }
            obs.swap(good);
        }
    }

    // Poses of cameras that lost all observations keep their pre-BA value; report them.
    std::set<int> in_ba;
    for (const auto& o : obs) in_ba.insert(o.image);
    for (const auto& [img, pose] : cur.poses) {
        if (!in_ba.count(img) && params.verbose) {
            std::cout << "BA warning: image " << img << " has no observations left\n";
        }
    }
    return result;
}

}  // namespace sfm
