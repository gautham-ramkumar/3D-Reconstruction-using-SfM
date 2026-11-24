// Module 2 check: synthetic two-view scene with known poses, pixel noise and outliers.
// Every estimate is compared against ground truth.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <set>
#include <string>

#include <Eigen/Geometry>
#include <Eigen/SVD>

#include "sfm/geometry.hpp"
#include "sfm/rng.hpp"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (!ok) ++failures;
}

double rotation_error_deg(const Eigen::Matrix3d& A, const Eigen::Matrix3d& B) {
    return Eigen::AngleAxisd(A.transpose() * B).angle() * 180.0 / M_PI;
}

double angle_between_deg(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    const double c = std::clamp(a.normalized().dot(b.normalized()), -1.0, 1.0);
    return std::acos(c) * 180.0 / M_PI;
}

}  // namespace

int main() {
    sfm::set_random_seed(0);
    std::mt19937 gen(42);  // scene generator, separate from the pipeline RNG
    std::uniform_real_distribution<double> ux(-1.5, 1.5), uz(4.0, 8.0);
    std::uniform_real_distribution<double> upx(0.0, 1080.0), upy(0.0, 1920.0);
    std::normal_distribution<double> noise(0.0, 0.5);

    // Buddha-like intrinsics; camera 2 is rotated 8 deg about y, centered 0.8 to the right.
    Eigen::Matrix3d K;
    K << 1700, 0, 540,
         0, 1700, 960,
         0, 0, 1;
    const Eigen::Matrix3d R_true = Eigen::AngleAxisd(-8.0 * M_PI / 180.0, Eigen::Vector3d::UnitY()).toRotationMatrix();
    const Eigen::Vector3d C2(0.8, 0.0, 0.0);
    const Eigen::Vector3d t_true = -R_true * C2;
    const sfm::Mat34 P1 = sfm::projection_matrix(K, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    const sfm::Mat34 P2 = sfm::projection_matrix(K, R_true, t_true);

    auto project = [](const sfm::Mat34& P, const Eigen::Vector3d& X) {
        return Eigen::Vector2d((P * X.homogeneous()).hnormalized());
    };

    // 400 true correspondences + 100 random outliers.
    sfm::Points3D X_true;
    sfm::Points2D clean1, clean2, pts1, pts2;
    std::set<int> true_inliers;
    while (X_true.size() < 400) {
        const Eigen::Vector3d X(ux(gen), ux(gen), uz(gen));
        const Eigen::Vector2d a = project(P1, X), b = project(P2, X);
        if (a.x() < 0 || a.x() > 1080 || b.x() < 0 || b.x() > 1080) continue;
        X_true.push_back(X);
        clean1.push_back(a);
        clean2.push_back(b);
        true_inliers.insert(static_cast<int>(pts1.size()));
        pts1.push_back(a + Eigen::Vector2d(noise(gen), noise(gen)));
        pts2.push_back(b + Eigen::Vector2d(noise(gen), noise(gen)));
        if (pts1.size() % 4 == 3) {  // interleave outliers
            pts1.emplace_back(upx(gen), upy(gen));
            pts2.emplace_back(upx(gen), upy(gen));
        }
    }
    std::cout << pts1.size() << " correspondences, " << true_inliers.size() << " true inliers\n";

    // Normalization: centroid at origin, mean distance sqrt(2).
    {
        const auto n = sfm::normalize_points(pts1);
        Eigen::Vector2d c = Eigen::Vector2d::Zero();
        double d = 0;
        for (const auto& p : n->first) { c += p; }
        c /= n->first.size();
        for (const auto& p : n->first) d += p.norm();
        d /= n->first.size();
        check(n && c.norm() < 1e-9 && std::abs(d - std::sqrt(2.0)) < 1e-9, "normalize_points: centroid 0, mean dist sqrt(2)");
        check(!sfm::normalize_points(sfm::Points2D(5, Eigen::Vector2d(3, 3))), "normalize_points rejects coincident points");
    }

    // 8-point on noise-free data: epipolar constraint holds exactly.
    {
        const auto F = sfm::estimate_fundamental_matrix(clean1, clean2);
        double max_r = 0;
        for (size_t k = 0; k < clean1.size(); ++k) {
            max_r = std::max(max_r, std::abs(clean2[k].homogeneous().dot(*F * clean1[k].homogeneous())));
        }
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(*F);
        std::cout << "  max |x2^T F x1| = " << max_r << ", sigma3 = " << svd.singularValues()(2) << "\n";
        check(F && max_r < 1e-6, "8-point satisfies epipolar constraint on clean data");
        check(svd.singularValues()(2) < 1e-12 && std::abs(F->norm() - 1.0) < 1e-12, "F is rank 2 with unit norm");
    }

    // RANSAC: recovers the true inliers and rejects the outliers.
    const auto res = sfm::ransac_fundamental_matrix(pts1, pts2, 2000, 1.0);
    int tp = 0;
    for (int i : res.inliers) tp += true_inliers.count(i);
    const double recall = double(tp) / true_inliers.size();
    const double precision = res.inliers.empty() ? 0 : double(tp) / res.inliers.size();
    std::cout << "  RANSAC: " << res.inliers.size() << " inliers, recall " << recall
              << ", precision " << precision << "\n";
    check(res.F.has_value(), "RANSAC found an F");
    // With sigma = 0.5 px, an exact F would keep ~95% of true inliers at 1 px^2. RANSAC's F
    // comes from 8 noisy points and is never refit, so recall lands somewhat lower.
    check(recall > 0.85 && precision > 0.97, "RANSAC recall > 0.85 and precision > 0.97");

    // Refitting F on all inliers should recover close to the ~95% an exact F would keep.
    {
        const auto refined = sfm::refine_fundamental_matrix(res, pts1, pts2, 1.0);
        int tp_r = 0;
        for (int i : refined.inliers) tp_r += true_inliers.count(i);
        const double recall_r = double(tp_r) / true_inliers.size();
        const double precision_r = double(tp_r) / refined.inliers.size();
        std::cout << "  refit: " << refined.inliers.size() << " inliers, recall " << recall_r
                  << ", precision " << precision_r << "\n";
        check(recall_r > recall && recall_r > 0.92 && precision_r > 0.97,
              "refit raises recall above 0.92 and keeps precision > 0.97");
    }

    // E -> 4 poses -> cheirality picks the true one (t only up to scale).
    sfm::Points2D in1, in2;
    for (int i : res.inliers) { in1.push_back(pts1[i]); in2.push_back(pts2[i]); }
    const Eigen::Matrix3d E = sfm::compute_essential_matrix(*res.F, K, K);
    Eigen::JacobiSVD<Eigen::Matrix3d> svd_e(E);
    const Eigen::Vector3d s = svd_e.singularValues();
    check(std::abs(s(0) - 1) < 1e-9 && std::abs(s(1) - 1) < 1e-9 && s(2) < 1e-9, "E has singular values (1, 1, 0)");

    const auto poses = sfm::estimate_camera_pose(E);
    bool all_rotations = true;
    for (const auto& [R, t] : poses) all_rotations &= std::abs(R.determinant() - 1.0) < 1e-9;
    check(all_rotations, "all 4 pose hypotheses have det(R) = +1");

    const auto [R_est, t_est] = sfm::cheirality_check(poses, K, K, in1, in2);
    const double r_err = rotation_error_deg(R_est, R_true);
    const double t_err = angle_between_deg(t_est, t_true);
    std::cout << "  rotation error " << r_err << " deg, translation direction error " << t_err << " deg\n";
    check(r_err < 0.5 && t_err < 3.0, "cheirality recovers R (< 0.5 deg) and t direction (< 3 deg)");

    // Triangulation with true poses on clean pixels reproduces the 3D points.
    {
        const sfm::Points3D X = sfm::triangulate_points(P1, P2, clean1, clean2);
        double max_err = 0;
        for (size_t k = 0; k < X.size(); ++k) max_err = std::max(max_err, (X[k] - X_true[k]).norm());
        std::cout << "  max triangulation error " << max_err << "\n";
        check(max_err < 1e-6, "triangulation is exact on clean data");
    }

    // Parallax: compare with angles computed directly from ground-truth points.
    {
        std::vector<double> gt;
        for (const auto& X : X_true) gt.push_back(angle_between_deg(X, X - C2));
        std::nth_element(gt.begin(), gt.begin() + gt.size() / 2, gt.end());
        const double est = sfm::calculate_median_parallax(clean1, clean2, K, K, R_true, t_true);
        std::cout << "  median parallax " << est << " deg (ground truth ~" << gt[gt.size() / 2] << ")\n";
        check(std::abs(est - gt[gt.size() / 2]) < 0.05, "median parallax matches ground truth");
    }

    std::cout << (failures == 0 ? "All checks passed.\n" : "Some checks FAILED.\n");
    return failures == 0 ? 0 : 1;
}
