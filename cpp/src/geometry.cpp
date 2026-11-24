#include "sfm/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include <Eigen/Geometry>  // homogeneous()
#include <Eigen/LU>
#include <Eigen/SVD>

#include "sfm/rng.hpp"

namespace sfm {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kRadToDeg = 180.0 / M_PI;

Points2D select(const Points2D& pts, const std::vector<int>& idx) {
    Points2D out;
    out.reserve(idx.size());
    for (int i : idx) out.push_back(pts[i]);
    return out;
}

}  // namespace

Mat34 projection_matrix(const Eigen::Matrix3d& K, const Eigen::Matrix3d& R,
                        const Eigen::Vector3d& t) {
    Mat34 Rt;
    Rt << R, t;
    return K * Rt;
}

std::optional<std::pair<Points2D, Eigen::Matrix3d>> normalize_points(const Points2D& pts) {
    if (pts.empty()) return std::nullopt;

    Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
    for (const auto& p : pts) centroid += p;
    centroid /= static_cast<double>(pts.size());

    double avg_dist = 0.0;
    for (const auto& p : pts) avg_dist += (p - centroid).norm();
    avg_dist /= static_cast<double>(pts.size());
    if (avg_dist < 1e-6) return std::nullopt;

    const double s = std::sqrt(2.0) / avg_dist;
    Eigen::Matrix3d T;
    T << s, 0, -s * centroid.x(),
         0, s, -s * centroid.y(),
         0, 0, 1;

    Points2D normalized;
    normalized.reserve(pts.size());
    for (const auto& p : pts) normalized.push_back(s * (p - centroid));
    return std::make_pair(normalized, T);
}

std::optional<Eigen::Matrix3d> estimate_fundamental_matrix(const Points2D& pts1,
                                                           const Points2D& pts2) {
    const auto n1 = normalize_points(pts1);
    if (!n1) return std::nullopt;
    const auto n2 = normalize_points(pts2);
    if (!n2) return std::nullopt;
    const auto& [p1, T1] = *n1;
    const auto& [p2, T2] = *n2;

    Eigen::MatrixXd A(p1.size(), 9);
    for (size_t k = 0; k < p1.size(); ++k) {
        const double x1 = p1[k].x(), y1 = p1[k].y();
        const double x2 = p2[k].x(), y2 = p2[k].y();
        A.row(k) << x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1.0;
    }

    // Full V is needed: with exactly 8 rows the null vector is not in the thin V.
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_a(A, Eigen::ComputeFullV);
    const Eigen::VectorXd f = svd_a.matrixV().col(8);
    Eigen::Matrix3d F_norm;
    F_norm << f(0), f(1), f(2),
              f(3), f(4), f(5),
              f(6), f(7), f(8);

    // Enforce rank 2 by zeroing the smallest singular value.
    Eigen::JacobiSVD<Eigen::Matrix3d> svd_f(F_norm, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d S = svd_f.singularValues();
    S(2) = 0.0;
    const Eigen::Matrix3d F_rank2 = svd_f.matrixU() * S.asDiagonal() * svd_f.matrixV().transpose();

    const Eigen::Matrix3d F = T2.transpose() * F_rank2 * T1;
    const double norm = F.norm();
    if (norm < 1e-12) return std::nullopt;
    // Unit Frobenius norm instead of F(2,2): avoids blow-up when F(2,2) ~ 0.
    return F / norm;
}

std::vector<int> sampson_inliers(const Eigen::Matrix3d& F, const Points2D& pts1,
                                 const Points2D& pts2, double threshold) {
    std::vector<int> inliers;
    for (size_t k = 0; k < pts1.size(); ++k) {
        const Eigen::Vector3d x1 = pts1[k].homogeneous();
        const Eigen::Vector3d x2 = pts2[k].homogeneous();
        const Eigen::Vector3d a = F * x1;
        const Eigen::Vector3d b = F.transpose() * x2;
        const double den = a(0) * a(0) + a(1) * a(1) + b(0) * b(0) + b(1) * b(1);
        if (den <= 0.0) continue;
        const double r = x2.dot(a);
        if (r * r / den < threshold) inliers.push_back(static_cast<int>(k));
    }
    return inliers;
}

FundamentalResult ransac_fundamental_matrix(const Points2D& pts1, const Points2D& pts2,
                                            int num_iterations, double threshold) {
    FundamentalResult best;
    const int n = static_cast<int>(pts1.size());
    if (n < 8 || pts2.size() != pts1.size()) return best;

    for (int iter = 0; iter < num_iterations; ++iter) {
        const std::vector<int> sample = sample_without_replacement(n, 8);
        const auto F = estimate_fundamental_matrix(select(pts1, sample), select(pts2, sample));
        if (!F) continue;

        std::vector<int> current = sampson_inliers(*F, pts1, pts2, threshold);
        if (current.size() > best.inliers.size()) {
            best.inliers = std::move(current);
            best.F = F;
        }
    }
    return best;
}

FundamentalResult refine_fundamental_matrix(const FundamentalResult& result,
                                            const Points2D& pts1, const Points2D& pts2,
                                            double threshold) {
    if (!result.F || result.inliers.size() < 8) return result;

    const auto F = estimate_fundamental_matrix(select(pts1, result.inliers),
                                               select(pts2, result.inliers));
    if (!F) return result;

    FundamentalResult refined{F, sampson_inliers(*F, pts1, pts2, threshold)};
    // A refit on a contaminated set can occasionally be worse; never lose inliers.
    return refined.inliers.size() >= result.inliers.size() ? refined : result;
}

Eigen::Matrix3d compute_essential_matrix(const Eigen::Matrix3d& F, const Eigen::Matrix3d& K_i,
                                         const Eigen::Matrix3d& K_j) {
    const Eigen::Matrix3d E = K_j.transpose() * F * K_i;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU() * Eigen::Vector3d(1, 1, 0).asDiagonal() * svd.matrixV().transpose();
}

std::array<RelativePose, 4> estimate_camera_pose(const Eigen::Matrix3d& E) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    // Flip signs so both rotations come out with det = +1.
    if (U.determinant() < 0) U.col(2) *= -1;
    if (V.determinant() < 0) V.col(2) *= -1;  // same as negating the last row of V^T

    Eigen::Matrix3d W;
    W << 0, -1, 0,
         1, 0, 0,
         0, 0, 1;

    const Eigen::Matrix3d R1 = U * W * V.transpose();
    const Eigen::Matrix3d R2 = U * W.transpose() * V.transpose();
    const Eigen::Vector3d t = U.col(2);
    return {{{R1, t}, {R1, -t}, {R2, t}, {R2, -t}}};
}

Points3D triangulate_points(const Mat34& P1, const Mat34& P2, const Points2D& pts1,
                            const Points2D& pts2) {
    Points3D out;
    out.reserve(pts1.size());
    for (size_t k = 0; k < pts1.size(); ++k) {
        // Each view gives two rows: u * P.row(2) - P.row(0) and v * P.row(2) - P.row(1).
        Eigen::Matrix4d A;
        A.row(0) = pts1[k].x() * P1.row(2) - P1.row(0);
        A.row(1) = pts1[k].y() * P1.row(2) - P1.row(1);
        A.row(2) = pts2[k].x() * P2.row(2) - P2.row(0);
        A.row(3) = pts2[k].y() * P2.row(2) - P2.row(1);

        Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
        const Eigen::Vector4d X = svd.matrixV().col(3);
        if (std::abs(X(3)) < 1e-12) {
            out.emplace_back(kNaN, kNaN, kNaN);  // point at infinity
        } else {
            out.push_back(X.head<3>() / X(3));
        }
    }
    return out;
}

RelativePose cheirality_check(const std::array<RelativePose, 4>& poses,
                              const Eigen::Matrix3d& K_i, const Eigen::Matrix3d& K_j,
                              const Points2D& pts1, const Points2D& pts2) {
    const Mat34 P1 = projection_matrix(K_i, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());

    int max_positive_depth = -1;
    RelativePose best_pose = poses[0];
    for (const auto& [R, t] : poses) {
        const Points3D X = triangulate_points(P1, projection_matrix(K_j, R, t), pts1, pts2);
        int count = 0;
        for (const auto& p : X) {
            if (p.z() > 0 && (R * p + t).z() > 0) ++count;  // NaN fails both tests
        }
        if (count > max_positive_depth) {
            max_positive_depth = count;
            best_pose = {R, t};
        }
    }
    std::cout << "Cheirality Check: Found valid pose with " << max_positive_depth
              << " points in front of cameras.\n";
    return best_pose;
}

double calculate_median_parallax(const Points2D& pts1, const Points2D& pts2,
                                 const Eigen::Matrix3d& K_i, const Eigen::Matrix3d& K_j,
                                 const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    const Mat34 P1 = projection_matrix(K_i, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    const Mat34 P2 = projection_matrix(K_j, R, t);
    const Points3D X = triangulate_points(P1, P2, pts1, pts2);

    const Eigen::Vector3d C1 = Eigen::Vector3d::Zero();
    const Eigen::Vector3d C2 = -R.transpose() * t;

    std::vector<double> angles;
    angles.reserve(X.size());
    for (const auto& p : X) {
        if (!p.allFinite()) continue;
        const Eigen::Vector3d v1 = p - C1;
        const Eigen::Vector3d v2 = p - C2;
        const double denom = v1.norm() * v2.norm();
        if (denom <= 0.0) continue;
        const double cos_theta = std::clamp(v1.dot(v2) / denom, -1.0, 1.0);
        angles.push_back(std::acos(cos_theta) * kRadToDeg);
    }
    if (angles.empty()) return 0.0;

    // Same convention as np.median: average the two middle values for even counts.
    const size_t mid = angles.size() / 2;
    std::nth_element(angles.begin(), angles.begin() + mid, angles.end());
    double median = angles[mid];
    if (angles.size() % 2 == 0) {
        median = 0.5 * (median + *std::max_element(angles.begin(), angles.begin() + mid));
    }
    return median;
}

}  // namespace sfm
