// SIMPLE_RADIAL lens model: x_d = x_u * (1 + k1 * r_u^2), in normalized coordinates.
#pragma once

#include <Eigen/Core>

#include "sfm/types.hpp"

namespace sfm {

// Undistorted pixel -> distorted pixel (what the real camera records).
Eigen::Vector2d distort_pixel(const Eigen::Vector2d& px, const Intrinsics& intr);

// Distorted pixel -> undistorted (pinhole) pixel, by fixed-point iteration.
Eigen::Vector2d undistort_pixel(const Eigen::Vector2d& px, const Intrinsics& intr);

// Replace every keypoint position with its undistorted pixel. Image i uses camera i + 1.
void undistort_keypoints(Keypoints& keypoints, const IntrinsicsMap& intrinsics);

}  // namespace sfm
