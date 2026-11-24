// Two-view geometry: F/E estimation, pose recovery, triangulation.
// Everything here is implemented from scratch on top of Eigen (no OpenCV).
#pragma once

#include <array>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace sfm {

using Points2D = std::vector<Eigen::Vector2d>;
using Points3D = std::vector<Eigen::Vector3d>;
using Mat34 = Eigen::Matrix<double, 3, 4>;
using RelativePose = std::pair<Eigen::Matrix3d, Eigen::Vector3d>;  // (R, t)

// P = K [R | t]
Mat34 projection_matrix(const Eigen::Matrix3d& K, const Eigen::Matrix3d& R,
                        const Eigen::Vector3d& t);

// Hartley normalization: centroid to origin, mean distance sqrt(2).
// Returns nullopt for empty or degenerate (all points coincident) input.
std::optional<std::pair<Points2D, Eigen::Matrix3d>> normalize_points(const Points2D& pts);

// Normalized 8-point algorithm. F is rank 2 and scaled to unit Frobenius norm.
std::optional<Eigen::Matrix3d> estimate_fundamental_matrix(const Points2D& pts1,
                                                           const Points2D& pts2);

struct FundamentalResult {
    std::optional<Eigen::Matrix3d> F;  // nullopt if no sample produced a valid F
    std::vector<int> inliers;          // indices into pts1 / pts2
};

// Indices k whose Sampson error (px^2) under F is below threshold.
std::vector<int> sampson_inliers(const Eigen::Matrix3d& F, const Points2D& pts1,
                                 const Points2D& pts2, double threshold);

// RANSAC over 8-point samples, scored by Sampson error (px^2) < threshold.
FundamentalResult ransac_fundamental_matrix(const Points2D& pts1, const Points2D& pts2,
                                            int num_iterations = 5000,
                                            double threshold = 1.0);

// Refit F on all of `result`'s inliers and recount inliers over every correspondence.
// Keeps the original result if the refit fails or would keep fewer inliers.
FundamentalResult refine_fundamental_matrix(const FundamentalResult& result,
                                            const Points2D& pts1, const Points2D& pts2,
                                            double threshold = 1.0);

// E = Kj^T F Ki, projected onto the essential manifold (singular values 1, 1, 0).
Eigen::Matrix3d compute_essential_matrix(const Eigen::Matrix3d& F, const Eigen::Matrix3d& K_i,
                                         const Eigen::Matrix3d& K_j);

// The four (R, t) hypotheses: (R1,t), (R1,-t), (R2,t), (R2,-t).
std::array<RelativePose, 4> estimate_camera_pose(const Eigen::Matrix3d& E);

// Linear (DLT) triangulation. Points at infinity (w ~ 0) come back as NaN so that
// any "depth > x" test rejects them.
Points3D triangulate_points(const Mat34& P1, const Mat34& P2, const Points2D& pts1,
                            const Points2D& pts2);

// Pick the hypothesis with the most points in front of both cameras.
RelativePose cheirality_check(const std::array<RelativePose, 4>& poses,
                              const Eigen::Matrix3d& K_i, const Eigen::Matrix3d& K_j,
                              const Points2D& pts1, const Points2D& pts2);

// Median angle (degrees) between the two camera rays to each triangulated point.
double calculate_median_parallax(const Points2D& pts1, const Points2D& pts2,
                                 const Eigen::Matrix3d& K_i, const Eigen::Matrix3d& K_j,
                                 const Eigen::Matrix3d& R, const Eigen::Vector3d& t);

}  // namespace sfm
