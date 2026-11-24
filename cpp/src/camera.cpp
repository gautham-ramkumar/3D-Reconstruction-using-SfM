#include "sfm/camera.hpp"

namespace sfm {

namespace {

// K is [[f, 0, cx], [0, f, cy], [0, 0, 1]] for SIMPLE_RADIAL.
Eigen::Vector2d to_normalized(const Eigen::Vector2d& px, const Intrinsics& intr) {
    return {(px.x() - intr.K(0, 2)) / intr.K(0, 0), (px.y() - intr.K(1, 2)) / intr.K(1, 1)};
}

Eigen::Vector2d to_pixel(const Eigen::Vector2d& xn, const Intrinsics& intr) {
    return {intr.K(0, 0) * xn.x() + intr.K(0, 2), intr.K(1, 1) * xn.y() + intr.K(1, 2)};
}

}  // namespace

Eigen::Vector2d distort_pixel(const Eigen::Vector2d& px, const Intrinsics& intr) {
    const Eigen::Vector2d xu = to_normalized(px, intr);
    return to_pixel(xu * (1.0 + intr.k1 * xu.squaredNorm()), intr);
}

Eigen::Vector2d undistort_pixel(const Eigen::Vector2d& px, const Intrinsics& intr) {
    const Eigen::Vector2d xd = to_normalized(px, intr);
    // Solve x_u = x_d / (1 + k1 |x_u|^2). With k1 * r^2 < ~0.05 here this converges to
    // machine precision in well under 20 iterations.
    Eigen::Vector2d xu = xd;
    for (int iter = 0; iter < 20; ++iter) {
        const Eigen::Vector2d next = xd / (1.0 + intr.k1 * xu.squaredNorm());
        const bool converged = (next - xu).squaredNorm() < 1e-24;
        xu = next;
        if (converged) break;
    }
    return to_pixel(xu, intr);
}

void undistort_keypoints(Keypoints& keypoints, const IntrinsicsMap& intrinsics) {
    for (size_t i = 0; i < keypoints.size(); ++i) {
        const Intrinsics& intr = intrinsics.at(static_cast<int>(i) + 1);
        for (auto& kp : keypoints[i]) {
            const Eigen::Vector2d u = undistort_pixel({kp.pt.x, kp.pt.y}, intr);
            kp.pt = cv::Point2f(static_cast<float>(u.x()), static_cast<float>(u.y()));
        }
    }
}

}  // namespace sfm
